#!/usr/bin/env node
// Training host: runs the Training stage on this machine's GPUs for a
// Gausify Orbit page that chose "Brush on the host" or "gsplat on the host"
// (docs/host.md). The page uploads the dataset, follows the run and downloads
// the splat; each job gets a GPU of its own, so two GPUs train two jobs at
// once.
//
//   node host/server.mjs [--port 1441] [--listen 127.0.0.1] [--token <secret>]
//                        [--gpus 0,1] [--jobs-per-gpu 1] [--work <folder>]
//                        [--brush host/bin/brush] [--python host/.venv/bin/python]
//                        [--serve dist-web] [--fake]
//   node host/server.mjs --check     lists GPUs and trainers, then exits
//
// - Listens on loopback unless told otherwise: reach it through an SSH tunnel
//   (`ssh -L 1441:localhost:1441 host`), which also keeps the page a secure
//   context. With --listen 0.0.0.0, set --token.
// - --serve also serves a built app (`pnpm web:build`) from the same port.
// - --fake adds a trainer that needs no GPU (host/fake-trainer.mjs), for
//   developing the integration.

import { spawn, execFile } from "node:child_process";
import { randomUUID, timingSafeEqual } from "node:crypto";
import { createReadStream, createWriteStream, existsSync, mkdirSync, rmSync, statSync } from "node:fs";
import { mkdir, readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { checkVulkanPin, detectGpus, GpuPool, pinEnv, refreshMemory } from "./gpus.mjs";
import { BRUSH_ENV, brushArgs, gsplatArgs, gsplatIgnored, listSnapshots, parseBrushLine, parseGsplatLine } from "./trainers.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const VERSION = JSON.parse(await readFile(join(here, "..", "package.json"), "utf8")).version;

/** A client that stops asking for events (closed tab, lost network) has its
 *  job cancelled, so the GPU doesn't stay taken. */
const ABANDONED_MS = 90_000;
/** A finished job's files are kept this long for the client to fetch. */
const KEEP_DONE_MS = 30 * 60_000;
const LONG_POLL_MS = 20_000;
const MAX_UPLOAD = 2 ** 31;

/**
 * @param {object} [o]
 * @param {import("./gpus.mjs").Gpu[]} [o.gpus] detected when omitted
 */
export async function createHost(o = {}) {
  const opts = {
    token: o.token ?? null,
    work: o.work ?? join(tmpdir(), "gausify-host"),
    brush: o.brush ?? join(here, "bin", process.platform === "win32" ? "brush.exe" : "brush"),
    // The environment host/setup.sh creates, when it is there.
    python: o.python ?? (existsSync(join(here, ".venv", "bin", "python")) ? join(here, ".venv", "bin", "python") : "python3"),
    serve: o.serve ?? null,
    fake: !!o.fake,
    perGpu: o.perGpu ?? 1,
    abandonedMs: o.abandonedMs ?? ABANDONED_MS,
  };
  const gpus = o.gpus ?? (await detectGpus(o.only ?? null));
  const pool = new GpuPool(gpus, opts.perGpu);
  mkdirSync(opts.work, { recursive: true });

  /** @type {Map<string, any>} */
  const jobs = new Map();

  // ---------- trainers ----------

  const run = (cmd, args) =>
    new Promise((res) => execFile(cmd, args, { timeout: 30_000 }, (err, stdout, stderr) => res({ ok: !err, out: `${stdout}${stderr}`.trim() })));

  let backendsCache = null;
  /** Which trainers are installed here (cached; `fresh` rechecks). */
  async function backends(fresh = false) {
    if (backendsCache && !fresh) return backendsCache;
    const b = {};
    b.brush = existsSync(opts.brush)
      ? { ok: true, detail: opts.brush }
      : { ok: false, detail: `${opts.brush} not found: build it with "pnpm host:brush"` };
    const py = await run(opts.python, [
      "-c",
      "import torch, gsplat; print(gsplat.__version__, torch.__version__, torch.cuda.device_count() if torch.cuda.is_available() else 0)",
    ]);
    if (py.ok) {
      const [gs, torch, n] = py.out.split("\n").pop().split(" ");
      b.gsplat = Number(n) > 0 ? { ok: true, detail: `gsplat ${gs}, torch ${torch}, ${n} CUDA device${n === "1" ? "" : "s"}` } : { ok: false, detail: `gsplat ${gs} is installed but torch ${torch} sees no CUDA device` };
    } else {
      b.gsplat = { ok: false, detail: `${opts.python} can't import torch and gsplat: run "pnpm host:setup"` };
    }
    if (opts.fake) b.fake = { ok: true, detail: "no GPU; a stand-in for developing the integration" };
    return (backendsCache = b);
  }

  /** Command, pinned environment and line parser of a job's trainer. */
  function trainerOf(job, gpu) {
    const out = join(job.dir, "out");
    const total = Math.max(1, Math.round(Number(job.options.totalSteps) || 1));
    const every = Math.round(Number(job.options.exportEvery) || 0) || total;
    switch (job.backend) {
      case "brush":
        return { cmd: opts.brush, args: brushArgs(join(job.dir, "dataset"), out, job.brushConfig, every), env: { ...BRUSH_ENV, ...pinEnv("wgpu", gpu) }, parse: parseBrushLine };
      case "gsplat":
        return {
          cmd: opts.python,
          args: ["-u", ...gsplatArgs(join(here, "train_gsplat.py"), join(job.dir, "dataset"), out, job.options, job.masked)],
          env: { ...pinEnv("cuda", gpu), PYTHONUNBUFFERED: "1" },
          parse: parseGsplatLine,
        };
      case "fake":
        return { cmd: process.execPath, args: [join(here, "fake-trainer.mjs"), join(job.dir, "dataset"), out, String(total), String(every)], env: {}, parse: parseGsplatLine };
    }
    throw new Error(`unknown trainer "${job.backend}"`);
  }

  // ---------- jobs ----------

  function emit(job, e) {
    job.events.push(e);
    for (const wake of job.waiters.splice(0)) wake();
  }

  function finish(job, state, e) {
    if (job.state === "done" || job.state === "failed" || job.state === "cancelled") return;
    job.state = state;
    job.endedAt = Date.now();
    pool.release(job.id);
    emit(job, e);
  }

  function remove(job) {
    jobs.delete(job.id);
    const rm = () => rmSync(job.dir, { recursive: true, force: true });
    // A trainer still shutting down would write into the folder again.
    if (job.proc && job.proc.exitCode == null && job.proc.signalCode == null) job.proc.once("close", rm);
    else rm();
  }

  function cancel(job) {
    if (job.proc && job.proc.exitCode == null) {
      job.proc.kill("SIGTERM");
      // A trainer stuck in a GPU call may ignore SIGTERM.
      setTimeout(() => job.proc.exitCode == null && job.proc.kill("SIGKILL"), 5000).unref();
    }
    finish(job, "cancelled", { t: "cancelled" });
  }

  /** New snapshots in the job's output folder; keeps the latest two files
   *  (the previous one may still be downloading). Brush writes the file in
   *  place, so one is only announced once its size has stopped changing
   *  (`settled`: the trainer has exited). */
  function scanSnapshots(job, settled = false) {
    const snaps = listSnapshots(join(job.dir, "out"));
    const last = snaps.at(-1);
    if (last && last.iter > job.lastSnapshot) {
      const size = statSync(last.path).size;
      if (size > 0 && (settled || (job.pendingSnapshot?.iter === last.iter && job.pendingSnapshot.size === size))) {
        job.lastSnapshot = last.iter;
        job.pendingSnapshot = null;
        emit(job, { t: "snapshot", iter: last.iter });
      } else {
        job.pendingSnapshot = { iter: last.iter, size };
      }
    }
    for (const old of snaps.slice(0, -2)) rmSync(old.path, { force: true });
    return snaps;
  }

  async function start(job) {
    if (job.state !== "created") throw httpError(409, "job already started");
    const available = await backends();
    if (!available[job.backend]?.ok) throw httpError(400, available[job.backend]?.detail ?? `unknown trainer "${job.backend}"`);
    if (!existsSync(join(job.dir, "dataset", "sparse", "0", "images.bin"))) throw httpError(400, "the dataset has no sparse/0/images.bin");
    job.state = "queued";
    const gpuPromise = pool.acquire(job.id, job.gpuWanted);
    const ahead = pool.position(job.id);
    if (ahead >= 0) emit(job, { t: "queued", ahead });
    void gpuPromise.then(
      (gpu) => launch(job, gpu),
      () => {}, // cancelled while waiting
    );
  }

  function launch(job, gpu) {
    if (job.state !== "queued") return pool.release(job.id);
    job.state = "running";
    job.gpu = gpu.index;
    job.startedAt = Date.now();
    emit(job, { t: "started", gpu: gpu.index, gpuName: gpu.name, backend: job.backend });
    if (job.backend === "gsplat") {
      const ignored = gsplatIgnored(job.options);
      if (ignored.length) emit(job, { t: "log", line: `not available with gsplat, ignored: ${ignored.join(", ")}` });
    }
    let t;
    try {
      t = trainerOf(job, gpu);
    } catch (e) {
      return finish(job, "failed", { t: "failed", error: e.message });
    }
    mkdirSync(join(job.dir, "out"), { recursive: true });
    const proc = spawn(t.cmd, t.args, { env: { ...process.env, ...t.env }, stdio: ["ignore", "pipe", "pipe"] });
    job.proc = proc;
    const tail = [];
    const onLine = (raw) => {
      const e = t.parse(raw);
      if (!e) return;
      if (e.t === "log") {
        tail.push(e.line);
        if (tail.length > 30) tail.shift();
      }
      if (e.t === "step") {
        // A step line without a splat count keeps the last one known.
        if (e.splats != null) job.splats = e.splats;
        else e.splats = job.splats;
        job.iter = e.iter;
      }
      // Snapshots and the end are read from disk and the exit code instead.
      if (e.t === "snapshot" || e.t === "done") return;
      emit(job, e);
    };
    const watch = setInterval(() => job.state === "running" && scanSnapshots(job), 500);
    createInterface({ input: proc.stdout }).on("line", onLine);
    createInterface({ input: proc.stderr }).on("line", onLine);
    proc.on("error", (e) => {
      clearInterval(watch);
      finish(job, "failed", { t: "failed", error: `couldn't start ${t.cmd}: ${e.message}` });
    });
    proc.on("close", (code, signal) => {
      clearInterval(watch);
      if (job.state !== "running") return;
      const last = scanSnapshots(job, true).at(-1);
      // Both trainers export at the last step; the last file is the result.
      if (code === 0 && last) {
        job.result = last.path;
        return finish(job, "done", { t: "done", iter: last.iter, seconds: (Date.now() - job.startedAt) / 1000, bytes: statSync(last.path).size });
      }
      const why = code === 0 ? "the trainer ended without the final splat" : `the trainer exited with ${signal ?? `code ${code}`}`;
      finish(job, "failed", { t: "failed", error: [why, ...tail.slice(-8)].join("\n") });
    });
  }

  // Abandoned and expired jobs.
  const sweep = setInterval(() => {
    const now = Date.now();
    for (const job of [...jobs.values()]) {
      const ended = job.endedAt != null;
      if (!ended && now - job.seenAt > opts.abandonedMs) cancel(job);
      if (job.endedAt != null && now - Math.max(job.endedAt, job.seenAt) > KEEP_DONE_MS) remove(job);
    }
  }, Math.min(10_000, opts.abandonedMs / 2));
  sweep.unref();

  // ---------- HTTP ----------

  function httpError(status, message) {
    return Object.assign(new Error(message), { status });
  }

  const cors = (req) => ({
    "Access-Control-Allow-Origin": req.headers.origin ?? "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
    // Chrome asks before a public page (orbit.gausify.com) may call loopback.
    "Access-Control-Allow-Private-Network": "true",
    "Access-Control-Max-Age": "600",
    Vary: "Origin",
  });

  function authorized(req) {
    if (!opts.token) return true;
    const given = Buffer.from(/^Bearer (.+)$/.exec(req.headers.authorization ?? "")?.[1] ?? "");
    const want = Buffer.from(opts.token);
    return given.length === want.length && timingSafeEqual(given, want);
  }

  const json = (req, res, status, body) => {
    res.writeHead(status, { ...cors(req), "Content-Type": "application/json", "Cache-Control": "no-store" });
    res.end(JSON.stringify(body));
  };

  async function body(req, limit = 1 << 20) {
    const chunks = [];
    let size = 0;
    for await (const c of req) {
      size += c.length;
      if (size > limit) throw httpError(413, "request too large");
      chunks.push(c);
    }
    return Buffer.concat(chunks);
  }

  /** `rel` inside `root`, or a 400: uploads name their own paths. */
  function inside(root, rel) {
    const clean = normalize(rel).replace(/^([/\\])+/, "");
    const full = resolve(root, clean);
    if (!clean || clean.split(/[/\\]/).some((p) => p === ".." || p === "") || !(full + sep).startsWith(resolve(root) + sep)) throw httpError(400, "invalid path");
    return full;
  }

  function sendFile(req, res, path, type) {
    const size = statSync(path).size;
    res.writeHead(200, { ...cors(req), "Content-Type": type, "Content-Length": size, "Cache-Control": "no-store" });
    createReadStream(path).pipe(res);
  }

  async function info() {
    await refreshMemory(gpus);
    return {
      name: "gausify-host",
      version: VERSION,
      gpus: gpus.map((g) => ({ index: g.index, name: g.name, memoryMb: g.memoryMb, usedMb: g.usedMb, jobs: pool.jobsOn(g.index).length })),
      jobsPerGpu: opts.perGpu,
      backends: await backends(),
      queued: pool.waiting.length,
    };
  }

  const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm", ".png": "image/png", ".svg": "image/svg+xml", ".ico": "image/x-icon", ".webmanifest": "application/manifest+json", ".gz": "application/gzip", ".onnx": "application/octet-stream", ".bin": "application/octet-stream" };

  function serveStatic(req, res, pathname) {
    let file;
    try {
      file = inside(opts.serve, pathname === "/" ? "index.html" : decodeURIComponent(pathname));
    } catch {
      return false;
    }
    if (!existsSync(file) || !statSync(file).isFile()) return false;
    res.writeHead(200, { "Content-Type": MIME[extname(file)] ?? "application/octet-stream", "Content-Length": statSync(file).size });
    createReadStream(file).pipe(res);
    return true;
  }

  async function handle(req, res) {
    const url = new URL(req.url, "http://host");
    const path = url.pathname;
    if (req.method === "OPTIONS") {
      res.writeHead(204, cors(req));
      return res.end();
    }
    if (!path.startsWith("/api/")) {
      if (opts.serve && req.method === "GET" && serveStatic(req, res, path)) return;
      throw httpError(404, "not found");
    }
    if (!authorized(req)) throw httpError(401, "this host needs its access token");

    if (path === "/api/info" && req.method === "GET") return json(req, res, 200, await info());

    if (path === "/api/jobs" && req.method === "POST") {
      const b = JSON.parse((await body(req)).toString("utf8") || "{}");
      if (!["brush", "gsplat", "fake"].includes(b.backend) || (b.backend === "fake" && !opts.fake)) throw httpError(400, `unknown trainer "${b.backend}"`);
      if (!b.options || typeof b.options !== "object") throw httpError(400, "missing training options");
      let gpuWanted;
      try {
        gpuWanted = pool.normalize(b.gpu);
      } catch (e) {
        throw httpError(400, e.message);
      }
      const id = randomUUID();
      const dir = join(opts.work, id);
      await mkdir(join(dir, "dataset"), { recursive: true });
      const job = {
        id, dir, backend: b.backend, gpuWanted, masked: !!b.masked, options: b.options, brushConfig: b.brushConfig && typeof b.brushConfig === "object" ? b.brushConfig : {},
        state: "created", events: [], waiters: [], seenAt: Date.now(), endedAt: null, startedAt: null, gpu: null, proc: null, iter: 0, splats: 0, lastSnapshot: 0, pendingSnapshot: null, result: null,
      };
      // Bad options fail here, not after the upload.
      try {
        if (job.backend === "brush") brushArgs("", "", job.brushConfig, 1);
      } catch (e) {
        rmSync(dir, { recursive: true, force: true });
        throw httpError(400, e.message);
      }
      jobs.set(id, job);
      return json(req, res, 200, { id });
    }

    const m = /^\/api\/jobs\/([0-9a-f-]{36})(\/.*)?$/.exec(path);
    const job = m && jobs.get(m[1]);
    if (!job) throw httpError(404, "no such job (the host forgets jobs it restarted over, or left unattended)");
    job.seenAt = Date.now();
    const sub = m[2] ?? "";

    if (sub === "" && req.method === "DELETE") {
      cancel(job);
      remove(job);
      return json(req, res, 200, {});
    }

    if (sub.startsWith("/files/") && req.method === "PUT") {
      if (job.state !== "created") throw httpError(409, "job already started");
      const dest = inside(join(job.dir, "dataset"), decodeURIComponent(sub.slice("/files/".length)));
      if (Number(req.headers["content-length"] ?? 0) > MAX_UPLOAD) throw httpError(413, "file too large");
      await mkdir(dirname(dest), { recursive: true });
      await new Promise((ok, fail) => {
        const w = createWriteStream(dest);
        req.pipe(w);
        w.on("finish", ok);
        w.on("error", fail);
        req.on("error", fail);
        req.on("aborted", () => fail(httpError(400, "upload interrupted")));
      });
      return json(req, res, 200, {});
    }

    if (sub === "/start" && req.method === "POST") {
      await start(job);
      return json(req, res, 200, { state: job.state });
    }

    if (sub === "/events" && req.method === "GET") {
      const after = Math.max(0, Number(url.searchParams.get("after")) || 0);
      if (job.events.length <= after && job.endedAt == null) {
        await new Promise((wake) => {
          const timer = setTimeout(wake, LONG_POLL_MS);
          job.waiters.push(() => (clearTimeout(timer), wake()));
          req.on("close", wake);
        });
      }
      job.seenAt = Date.now();
      return json(req, res, 200, { events: job.events.slice(after), next: job.events.length, state: job.state });
    }

    if (req.method === "GET" && (sub === "/result" || sub.startsWith("/snapshot/"))) {
      const file = sub === "/result" ? job.result : listSnapshots(join(job.dir, "out")).find((s) => s.iter === Number(sub.slice("/snapshot/".length)))?.path;
      if (!file || !existsSync(file)) throw httpError(404, sub === "/result" ? "the job has no result yet" : "that preview is gone (a newer one replaced it)");
      return sendFile(req, res, file, "application/octet-stream");
    }

    throw httpError(404, "not found");
  }

  const server = createServer((req, res) => {
    handle(req, res).catch((e) => {
      if (res.headersSent) return res.destroy();
      const status = e.status ?? (e instanceof SyntaxError ? 400 : 500);
      if (status >= 500) console.error(e);
      json(req, res, status, { error: e.message });
    });
  });
  // Long polls outlive Node's default request timeouts.
  server.requestTimeout = 0;
  server.headersTimeout = 60_000;

  return {
    server,
    gpus,
    pool,
    jobs,
    info,
    backends,
    close() {
      clearInterval(sweep);
      for (const job of [...jobs.values()]) {
        cancel(job);
        remove(job);
      }
      return new Promise((r) => server.close(r));
    },
  };
}

// ---------- command line ----------

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const opt = (name, def = null) => {
    const i = args.indexOf(`--${name}`);
    return i >= 0 ? args[i + 1] : def;
  };
  const flag = (name) => args.includes(`--${name}`);
  const port = Number(opt("port", 1441));
  const listen = opt("listen", "127.0.0.1");
  const token = opt("token", process.env.GAUSIFY_HOST_TOKEN ?? null);
  const serveDir = opt("serve", existsSync(join(here, "..", "dist-web", "index.html")) ? join(here, "..", "dist-web") : null);
  const host = await createHost({
    token,
    only: opt("gpus")?.split(",").map(Number) ?? null,
    perGpu: Number(opt("jobs-per-gpu", 1)),
    work: opt("work") ?? undefined,
    brush: opt("brush") ?? undefined,
    python: opt("python") ?? undefined,
    serve: serveDir,
    fake: flag("fake"),
  });

  const b = await host.backends();
  console.log(`Gausify Orbit training host ${VERSION}`);
  for (const g of host.gpus) console.log(`  GPU ${g.index}: ${g.name}${g.memoryMb ? ` · ${(g.memoryMb / 1024).toFixed(0)} GB` : ""}${g.pci ? ` · ${g.pci}` : ""}`);
  for (const [name, s] of Object.entries(b)) console.log(`  ${s.ok ? "✓" : "✗"} ${name}: ${s.detail}`);

  if (flag("check")) {
    const pin = await checkVulkanPin(host.gpus);
    console.log(`  ${pin.ok ? "✓" : pin.ok === false ? "✗" : "?"} Vulkan GPU selection (headless Chrome): ${pin.detail}`);
    await host.close();
    process.exit(Object.values(b).some((s) => s.ok) ? 0 : 1);
  }

  const loopback = ["127.0.0.1", "localhost", "::1"].includes(listen);
  if (!loopback && !token) {
    console.error(`Refusing to listen on ${listen} without --token: anyone who can reach the port could start training jobs.`);
    process.exit(2);
  }
  host.server.listen(port, listen, () => {
    console.log(`Listening on http://${listen}:${port}${serveDir ? ` (app from ${serveDir})` : ""}${token ? " · access token required" : ""}`);
    if (loopback) console.log(`From another computer: ssh -L ${port}:localhost:${port} <this host>, then use http://localhost:${port}`);
  });
  const stop = () => void host.close().then(() => process.exit(0));
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
}

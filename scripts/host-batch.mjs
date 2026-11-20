#!/usr/bin/env node
// Turns several orbit videos into splats on a multi-GPU Linux host, without a
// display: one headless Chrome per GPU, each taking the next video from the
// list (docs/host.md). With two GPUs, two videos are always in progress.
//
//   node scripts/host-batch.mjs <video>... --out <folder>
//        [--backend browser|brush|gsplat] [--gpus 0,1] [--app http://localhost:1440]
//        [--host-url http://localhost:1441] [--host-token <secret>]
//        [-- <options of scripts/web-bench/headless.mjs: --preset, --mask, --steps, --export …>]
//
// - --backend browser (default): the whole pipeline, training included, runs
//   in each Chrome, pinned to its GPU through Vulkan.
// - --backend brush | gsplat: frames, masks and poses run in each Chrome and
//   training goes to the training host (`pnpm host`), on the same GPU.
// - Needs the app's dev server with file access: `WEB_FS_ANY=1 pnpm web:dev`.
// - Each video's project (frames, poses, splat.ply, exports) lands in
//   <out>/<video name>/, with the run's output in <out>/<video name>.log.

import { spawn } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync } from "node:fs";
import { basename, dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { detectGpus } from "../host/gpus.mjs";

const argv = process.argv.slice(2);
const split = argv.indexOf("--");
const own = split >= 0 ? argv.slice(0, split) : argv;
const passthrough = split >= 0 ? argv.slice(split + 1) : [];
const VALUED = ["out", "backend", "gpus", "app", "host-url", "host-token"];
const opts = {};
const videos = [];
for (let i = 0; i < own.length; i++) {
  const name = own[i].startsWith("--") ? own[i].slice(2) : null;
  if (name && VALUED.includes(name)) opts[name] = own[++i];
  else if (name) {
    console.error(`unknown option --${name} (options for each run go after "--")`);
    process.exit(2);
  } else videos.push(resolve(own[i]));
}
const backend = opts.backend ?? "browser";
if (!videos.length || !opts.out || !["browser", "brush", "gsplat"].includes(backend)) {
  console.error("usage: host-batch.mjs <video>... --out <folder> [--backend browser|brush|gsplat] [--gpus 0,1] [-- <headless.mjs options>]");
  process.exit(2);
}
for (const v of videos) {
  if (!existsSync(v)) {
    console.error(`no such file: ${v}`);
    process.exit(2);
  }
}

const out = resolve(opts.out);
const app = (opts.app ?? "http://localhost:1440").replace(/\/+$/, "");
const headless = join(dirname(fileURLToPath(import.meta.url)), "web-bench", "headless.mjs");
const gpus = await detectGpus(opts.gpus?.split(",").map(Number) ?? null);
mkdirSync(out, { recursive: true });

try {
  await fetch(app, { signal: AbortSignal.timeout(5000) });
} catch {
  console.error(`The app isn't answering at ${app}: start it with "WEB_FS_ANY=1 pnpm web:dev".`);
  process.exit(1);
}

/** Output names: the video's name, numbered when two share one. */
const taken = new Set();
const jobs = videos.map((video) => {
  const stem = basename(video, extname(video));
  let name = stem;
  for (let n = 2; taken.has(name); n++) name = `${stem}-${n}`;
  taken.add(name);
  return { video, name };
});

console.log(`${jobs.length} video${jobs.length > 1 ? "s" : ""} · training: ${backend} · ${gpus.map((g) => (g.pseudo ? "default GPU" : `GPU ${g.index} (${g.name})`)).join(", ")}`);

const queue = [...jobs];
const results = [];
const children = new Set();

function runOne(job, gpu, slot) {
  const args = [
    headless,
    "video",
    `${app}/?video=/@fs${encodeURI(job.video)}`,
    "--out", join(out, job.name),
    "--backend", backend,
    ...(opts["host-url"] ? ["--host-url", opts["host-url"]] : []),
    ...(opts["host-token"] ? ["--host-token", opts["host-token"]] : []),
    // Without nvidia-smi there is one unnamed GPU and nothing to pin.
    ...(gpu.pseudo ? [] : ["--gpu", String(gpu.index), ...(backend === "browser" ? [] : ["--host-gpu", String(gpu.index)])]),
    ...passthrough,
  ];
  return new Promise((done) => {
    const t0 = Date.now();
    const log = createWriteStream(join(out, `${job.name}.log`));
    // Each Chrome needs a debugging port of its own.
    const child = spawn(process.execPath, args, { env: { ...process.env, CDP_PORT: String(9333 + slot) }, stdio: ["ignore", "pipe", "pipe"] });
    children.add(child);
    let last = "";
    const onData = (b) => {
      log.write(b);
      const lines = b.toString().split("\n").filter((l) => l.trim());
      if (lines.length) last = lines.at(-1).trim();
    };
    child.stdout.on("data", onData);
    child.stderr.on("data", onData);
    child.on("close", (code) => {
      children.delete(child);
      log.end();
      done({ ...job, ok: code === 0, seconds: (Date.now() - t0) / 1000, gpu: gpu.index, last });
    });
  });
}

async function worker(gpu, slot) {
  for (let job = queue.shift(); job; job = queue.shift()) {
    console.log(`[GPU ${gpu.index}] ${job.name} …`);
    const r = await runOne(job, gpu, slot);
    results.push(r);
    console.log(`[GPU ${gpu.index}] ${job.name}: ${r.ok ? `done in ${r.seconds.toFixed(0)} s` : `FAILED after ${r.seconds.toFixed(0)} s: ${r.last}`}`);
  }
}

process.on("SIGINT", () => {
  queue.length = 0;
  for (const c of children) c.kill("SIGTERM");
});

const t0 = Date.now();
await Promise.all(gpus.map((g, slot) => worker(g, slot)));
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${jobs.length} splats in ${((Date.now() - t0) / 1000).toFixed(0)} s → ${out}`);
for (const r of failed) console.log(`  failed: ${r.name} (see ${join(out, `${r.name}.log`)})`);
process.exit(failed.length ? 1 : 0);

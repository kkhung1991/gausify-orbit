// Training host: GPU discovery and scheduling, trainer command lines and
// output parsing, and the HTTP API driven end to end with the stand-in
// trainer (no GPU needed), through the same client the app uses.

import assert from "node:assert/strict";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";
import type { TrainHost, TrainOptions } from "../src/lib/api.ts";
import { parseTrainHost, TrainHostClient, TrainHostError, type HostEvent } from "../src/lib/trainhost.ts";
import type { TrainEvents } from "../web/src/brush.ts";
import { trainRemote, type DatasetFiles } from "../web/src/host/remote-train.ts";
import { trainConfig } from "../web/src/pipeline/train-config.ts";
import { GpuPool, parseNvidiaSmi, pciTag, pinEnv } from "./gpus.mjs";
import { createHost } from "./server.mjs";
import { brushArgs, gsplatArgs, gsplatIgnored, parseBrushLine, parseGsplatLine } from "./trainers.mjs";

const SMI = `0, NVIDIA GeForce RTX 4090, GPU-11111111-2222-3333-4444-555555555555, 00000000:01:00.0, 24564, 512
1, NVIDIA GeForce RTX 4090, GPU-66666666-7777-8888-9999-aaaaaaaaaaaa, 00000000:41:00.0, 24564, 3
`;

const OPTIONS: TrainOptions = { totalSteps: 200, shDegree: 1, maxSplats: 500_000, maxResolution: 960, exportEvery: 100, evalSplitEvery: 0, ssimWeight: 0 };

test("nvidia-smi output becomes GPUs in PCI order", () => {
  const gpus = parseNvidiaSmi(SMI);
  assert.equal(gpus.length, 2);
  assert.deepEqual(
    gpus.map((g) => [g.index, g.ordinal, g.name, g.memoryMb]),
    [
      [0, 0, "NVIDIA GeForce RTX 4090", 24564],
      [1, 1, "NVIDIA GeForce RTX 4090", 24564],
    ],
  );
  assert.equal(parseNvidiaSmi("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n").length, 0);
});

test("each kind of trainer is pinned to its own GPU", () => {
  const [g0, g1] = parseNvidiaSmi(SMI);
  assert.equal(pciTag("00000000:41:00.0"), "pci-0000_41_00_0");
  assert.equal(pciTag("garbage"), null);
  // Two identical cards differ by UUID and PCI address, never by name.
  assert.deepEqual(pinEnv("cuda", g1), { CUDA_DEVICE_ORDER: "PCI_BUS_ID", CUDA_VISIBLE_DEVICES: "GPU-66666666-7777-8888-9999-aaaaaaaaaaaa" });
  assert.deepEqual(pinEnv("wgpu", g1), { CUBECL_WGPU_DEFAULT_DEVICE: "DiscreteGpu(1)" });
  assert.deepEqual(pinEnv("vulkan", g0), { DRI_PRIME: "pci-0000_01_00_0", MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE: "1" });
  assert.notDeepEqual(pinEnv("vulkan", g0), pinEnv("vulkan", g1));
  assert.deepEqual(pinEnv("cuda", { ...g0, pseudo: true }), {});
});

test("two GPUs run two jobs at once and queue the third", async () => {
  const pool = new GpuPool(parseNvidiaSmi(SMI));
  const a = await pool.acquire("a", "auto");
  const b = await pool.acquire("b", "auto");
  assert.deepEqual([a.index, b.index], [0, 1]);
  let c: number | null = null;
  void pool.acquire("c", "auto").then((g) => (c = g.index));
  await Promise.resolve();
  assert.equal(c, null);
  assert.equal(pool.position("c"), 0);
  pool.release("b");
  await new Promise((r) => setImmediate(r));
  assert.equal(c, 1);
  assert.throws(() => pool.normalize(7), /no GPU 7/);
});

test("a job waiting for a busy GPU doesn't hold back one that fits a free GPU", async () => {
  const pool = new GpuPool(parseNvidiaSmi(SMI));
  await pool.acquire("a", 0);
  await pool.acquire("b", 1);
  const got: string[] = [];
  const wantsZero = pool.acquire("c", 0).then(() => got.push("c"));
  const wantsAny = pool.acquire("d", "auto").then(() => got.push("d"));
  pool.release("b");
  await wantsAny;
  assert.deepEqual(got, ["d"]);
  // A cancelled wait rejects and leaves the queue.
  const gone = pool.acquire("e", 0);
  pool.release("e");
  await assert.rejects(gone, /cancelled/);
  pool.release("a");
  await wantsZero;
  assert.deepEqual(got, ["d", "c"]);
});

test("the app's training options become a Brush command line", () => {
  const masked = brushArgs("/data", "/out", trainConfig({ ...OPTIONS, depthDistortionWeight: 10 }, true), 100).join(" ");
  assert.ok(masked.startsWith("/data --total-train-iters 200 --sh-degree 1 --max-splats 500000 --max-resolution 960 --ssim-weight 0"));
  assert.ok(masked.includes("--background-color 0.5,0.5,0.5 --background-noise-strength 0.5"));
  assert.ok(masked.includes("--depth-distortion-weight 10") && masked.includes("--geom-reg-crop true"));
  assert.ok(masked.endsWith("--export-every 100 --export-path /out --export-name splat_{iter}.ply"));
  // Evaluation off: Brush crashes on eval-every 0, so neither option is sent.
  assert.ok(!masked.includes("eval"));
  const evaluated = brushArgs("/data", "/out", trainConfig({ ...OPTIONS, evalSplitEvery: 16 }, false), 100).join(" ");
  assert.ok(evaluated.includes("--eval-split-every 16 --eval-every 100") && !evaluated.includes("background"));
  // Where files go is the host's business, whatever the page sends.
  assert.throws(() => brushArgs("/d", "/o", { "export-path": "/etc" }, 1), /set by the host/);
  assert.throws(() => brushArgs("/d", "/o", { "--bad key": 1 }, 1), /invalid Brush option/);
  assert.throws(() => brushArgs("/d", "/o", { "sh-degree": "3; rm -rf /" }, 1), /invalid value/);
});

test("Brush's log lines become events", () => {
  assert.deepEqual(parseBrushLine("[2026-10-04T19:51:20Z INFO  brush_cli] Refine iter 801, 13799 splats."), { t: "step", iter: 801, splats: 13799 });
  assert.deepEqual(parseBrushLine("[2026-10-04T19:51:20Z INFO  brush_cli] Step iter 825, 20412 splats"), { t: "step", iter: 825, splats: 20412 });
  assert.deepEqual(parseBrushLine("Eval iter 1200: PSNR 32.510826, ssim 0.9530626"), { t: "eval", iter: 1200, psnr: 32.510826, ssim: 0.9530626 });
  assert.deepEqual(parseBrushLine("[x INFO  brush_cli] Loaded dataset with 118 training, 8 eval views"), { t: "dataset", train: 118, eval: 8 });
  assert.deepEqual(parseBrushLine("[x WARN  brush_cli] something odd"), { t: "log", line: "something odd" });
  assert.equal(parseBrushLine("   "), null);
});

test("gsplat gets the options it has, and says which it lacks", () => {
  const args = gsplatArgs("train.py", "/data", "/out", { ...OPTIONS, lrMean: 2e-5 }, true).join(" ");
  assert.equal(args, "train.py --data /data --out /out --steps 200 --sh-degree 1 --max-splats 500000 --max-resolution 960 --ssim-weight 0 --export-every 100 --eval-split-every 0 --lr-mean 0.00002 --masked");
  assert.deepEqual(gsplatIgnored({ ...OPTIONS, depthDistortionWeight: 10, opacBinaryWeight: 0.01 }), ["Reduce haze (depth distortion)", "opacity (binary) regularization"]);
  assert.deepEqual(gsplatIgnored(OPTIONS), []);
  assert.deepEqual(parseGsplatLine('{"t": "step", "iter": 10, "splats": 5}'), { t: "step", iter: 10, splats: 5 });
  assert.deepEqual(parseGsplatLine("Traceback (most recent call last):"), { t: "log", line: "Traceback (most recent call last):" });
});

test("the stored setting survives bad data", () => {
  assert.deepEqual(parseTrainHost(null), { backend: "browser", url: "http://localhost:1441", token: "", gpu: "auto" });
  assert.deepEqual(parseTrainHost({ backend: "gsplat", url: "gpu-box:1441", gpu: 1, token: "s3" }), { backend: "gsplat", url: "gpu-box:1441", token: "s3", gpu: 1 });
  assert.equal(parseTrainHost({ backend: "metal", gpu: -1 }).backend, "browser");
  assert.equal(parseTrainHost({ gpu: 1.5 }).gpu, "auto");
});

// ---------- the HTTP API, with the stand-in trainer ----------

const work = mkdtempSync(join(tmpdir(), "gausify-host-test-"));
let host: Awaited<ReturnType<typeof createHost>>;
let url = "";
const TOKEN = "s3cret";

before(async () => {
  process.env.FAKE_STEP_MS = "5";
  host = await createHost({ gpus: parseNvidiaSmi(SMI), work, fake: true, token: TOKEN, brush: join(work, "no-brush"), python: join(work, "no-python") });
  await new Promise<void>((r) => host.server.listen(0, "127.0.0.1", r));
  url = `http://127.0.0.1:${(host.server.address() as AddressInfo).port}`;
});

after(async () => {
  await host.close();
  rmSync(work, { recursive: true, force: true });
});

const bytes = (s: string) => new Blob([s]);
const dataset = (images = 3): DatasetFiles => {
  const files = new Map<string, Blob>([
    ["sparse/0/cameras.bin", bytes("c")],
    ["sparse/0/images.bin", bytes("i")],
    ["sparse/0/points3D.bin", bytes("p")],
    ...Array.from({ length: images }, (_, i) => [`images/${String(i).padStart(5, "0")}.jpg`, bytes("jpg")] as [string, Blob]),
  ]);
  return { list: async () => [...files.keys()], read: async (p) => files.get(p)! };
};

function recorder() {
  const seen = { status: [] as string[], steps: [] as number[], snapshots: [] as number[], log: [] as string[], dataset: null as number | null };
  const ev: TrainEvents = {
    status: (m) => seen.status.push(m.key),
    dataset: (n) => (seen.dataset = n),
    step: (iter) => seen.steps.push(iter),
    eval: () => {},
    snapshot: (iter, ply) => {
      assert.equal(new TextDecoder().decode(ply.subarray(0, 4)), "ply\n");
      seen.snapshots.push(iter);
    },
    log: (l) => seen.log.push(l),
  };
  return { seen, ev };
}

const fake = (gpu: TrainHost["gpu"] = "auto") => ({ backend: "fake", url, token: TOKEN, gpu }) as unknown as TrainHost;

test("the host reports its GPUs and which trainers are installed", async () => {
  const info = await new TrainHostClient({ url, token: TOKEN }).info();
  assert.deepEqual(info.gpus.map((g) => [g.index, g.name, g.jobs]), [[0, "NVIDIA GeForce RTX 4090", 0], [1, "NVIDIA GeForce RTX 4090", 0]]);
  assert.equal(info.backends.fake.ok, true);
  assert.equal(info.backends.brush.ok, false);
  assert.match(info.backends.brush.detail, /pnpm host:brush/);
  assert.equal(info.backends.gsplat.ok, false);
});

test("a wrong token, a wrong address and a non-host are told apart", async () => {
  const reason = (p: Promise<unknown>) => p.then(() => "ok", (e) => (e instanceof TrainHostError ? e.reason : String(e)));
  assert.equal(await reason(new TrainHostClient({ url, token: "nope" }).info()), "auth");
  assert.equal(await reason(new TrainHostClient({ url, token: "" }).info()), "auth");
  assert.equal(await reason(new TrainHostClient({ url: "http://127.0.0.1:1", token: "" }).info()), "network");
  assert.equal(await reason(new TrainHostClient({ url: `${url}/nothing-here`, token: TOKEN }).info()), "http");
});

test("training on the host: upload, progress, previews and the final splat", async () => {
  const { seen, ev } = recorder();
  const ply = await trainRemote(dataset(), OPTIONS, false, fake(), ev, new AbortController().signal);
  assert.equal(new TextDecoder().decode(ply.subarray(0, 4)), "ply\n");
  assert.ok(ply.length > 3000);
  assert.deepEqual(seen.status.slice(0, 2), ["train.hostConnecting", "train.hostUploading"]);
  assert.equal(seen.status.at(-1), "train.hostDownloading");
  assert.equal(seen.dataset, 3);
  assert.equal(seen.steps.at(-1), 200);
  assert.ok(seen.steps.length >= 2 && seen.steps.every((s, i) => i === 0 || s >= seen.steps[i - 1]));
  assert.ok(seen.log.some((l) => /on GPU 0 \(NVIDIA GeForce RTX 4090\)/.test(l)));
  // The job and its files are gone from the host once the splat is fetched.
  await new Promise((r) => setTimeout(r, 100));
  assert.equal(host.jobs.size, 0);
  assert.equal(host.pool.jobsOn(0).length + host.pool.jobsOn(1).length, 0);
});

test("two jobs train at the same time, one on each GPU", async () => {
  const runs = [recorder(), recorder()];
  const longer = { ...OPTIONS, totalSteps: 600, exportEvery: 600 };
  const both = Promise.all(runs.map((r) => trainRemote(dataset(), longer, false, fake(), r.ev, new AbortController().signal)));
  // Sampled while they run: both GPUs busy at once.
  let together = false;
  const watch = setInterval(() => (together ||= host.pool.jobsOn(0).length === 1 && host.pool.jobsOn(1).length === 1), 5);
  await both;
  clearInterval(watch);
  assert.ok(together, "both GPUs had a job at the same moment");
  const used = runs.map((r) => /on GPU (\d)/.exec(r.seen.log.find((l) => l.includes("on GPU"))!)![1]).sort();
  assert.deepEqual(used, ["0", "1"]);
});

test("a third job waits in the queue for a GPU", async () => {
  const runs = [recorder(), recorder(), recorder()];
  const longer = { ...OPTIONS, totalSteps: 400, exportEvery: 400 };
  await Promise.all(runs.map((r) => trainRemote(dataset(), longer, false, fake(), r.ev, new AbortController().signal)));
  assert.equal(runs.filter((r) => r.seen.status.includes("train.hostQueued")).length, 1);
  assert.ok(runs.every((r) => r.seen.steps.at(-1) === 400));
});

test("cancelling stops the trainer and frees the GPU", async () => {
  const ctl = new AbortController();
  const { ev, seen } = recorder();
  const run = trainRemote(dataset(), { ...OPTIONS, totalSteps: 100_000 }, false, fake(1), { ...ev, step: (i, s) => (ev.step(i, s), ctl.abort()) }, ctl.signal);
  await assert.rejects(run, { name: "AbortError" });
  assert.ok(seen.steps.length >= 1);
  await new Promise((r) => setTimeout(r, 300));
  assert.equal(host.jobs.size, 0);
  assert.equal(host.pool.jobsOn(1).length, 0);
});

test("a trainer that dies reports why", async () => {
  process.env.FAKE_FAIL = "1";
  try {
    await assert.rejects(trainRemote(dataset(), OPTIONS, false, fake(), recorder().ev, new AbortController().signal), /training on the host failed: .*exited with code 1[\s\S]*CUDA out of memory/);
  } finally {
    delete process.env.FAKE_FAIL;
  }
});

test("a trainer that isn't installed is refused before any upload", async () => {
  const brush = { backend: "brush", url, token: TOKEN, gpu: "auto" } as TrainHost;
  let uploads = 0;
  const files = dataset();
  await assert.rejects(trainRemote({ list: files.list, read: (p) => (uploads++, files.read(p)) }, OPTIONS, false, brush, recorder().ev, new AbortController().signal), /pnpm host:brush/);
  assert.equal(uploads, 0);
  assert.equal(host.jobs.size, 0);
});

test("uploads can't leave the job's folder, and unknown GPUs are refused", async () => {
  const client = new TrainHostClient({ url, token: TOKEN });
  const job = { backend: "fake" as const, masked: false, options: OPTIONS, brushConfig: {} };
  await assert.rejects(client.createJob({ ...job, gpu: 5 }), /no GPU 5/);
  const id = await client.createJob({ ...job, gpu: "auto" });
  // Encoded separators survive URL normalization and reach the host as "..".
  const put = (raw: string) => fetch(`${url}/api/jobs/${id}/files/${raw}`, { method: "PUT", body: "x", headers: { Authorization: `Bearer ${TOKEN}` } });
  for (const raw of ["..%2Fescape.txt", "images%2F..%2F..%2Fescape.txt", "%2E%2E%2F%2E%2E%2Fescape.txt", "images%5C..%5C..%5Cescape.txt"]) assert.equal((await put(raw)).status, 400, raw);
  assert.ok(!existsSync(join(work, "escape.txt")) && !existsSync(join(work, id, "escape.txt")));
  // An absolute path is taken as relative to the dataset.
  assert.equal((await put("%2Fetc%2Fpasswd")).status, 200);
  assert.ok(existsSync(join(work, id, "dataset", "etc", "passwd")));
  // Without the model there is nothing to train.
  await client.upload(id, "images/a.jpg", bytes("x"));
  await assert.rejects(client.start(id), /images\.bin/);
  await client.remove(id);
  const events: HostEvent[] = [];
  await assert.rejects(client.events(id, events.length), /no such job/);
});

test("a job nobody follows is cancelled", async () => {
  const quick = await createHost({ gpus: parseNvidiaSmi(SMI), work: join(work, "abandoned"), fake: true, abandonedMs: 300 });
  await new Promise<void>((r) => quick.server.listen(0, "127.0.0.1", r));
  const client = new TrainHostClient({ url: `http://127.0.0.1:${(quick.server.address() as AddressInfo).port}`, token: "" });
  try {
    const id = await client.createJob({ backend: "fake", gpu: 0, masked: false, options: { ...OPTIONS, totalSteps: 1_000_000 }, brushConfig: {} });
    const files = dataset();
    for (const p of await files.list()) await client.upload(id, p, await files.read(p));
    await client.start(id);
    assert.equal(quick.pool.jobsOn(0).length, 1);
    // The page went away: no more requests.
    await new Promise((r) => setTimeout(r, 900));
    assert.equal(quick.pool.jobsOn(0).length, 0);
    assert.equal(quick.jobs.get(id)?.state, "cancelled");
  } finally {
    await quick.close();
  }
});

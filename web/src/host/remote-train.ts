// Training on a host (host/server.mjs, docs/host.md) instead of this
// browser: uploads the dataset, follows the run and brings back the previews
// and the final .ply. Same events as the in-browser training (brush.ts).

import type { TrainHost, TrainOptions } from "../../../src/lib/api";
import { TrainHostClient, TrainHostError, type HostEvent } from "../../../src/lib/trainhost.ts";
import type { TrainEvents } from "../brush";
import { trainConfig } from "../pipeline/train-config.ts";

/** What the client needs from storage: the dataset's files, by path inside
 *  it ("images/00001.jpg", "sparse/0/images.bin"). */
export interface DatasetFiles {
  list(): Promise<string[]>;
  read(path: string): Promise<Blob>;
}

/** A dataset folder in OPFS: images/ and sparse/0/, nothing else that might
 *  sit beside them. */
export function opfsDataset(dir: FileSystemDirectoryHandle): DatasetFiles {
  const at = async (path: string) => {
    const parts = path.split("/");
    let d = dir;
    for (const p of parts.slice(0, -1)) d = await d.getDirectoryHandle(p);
    return d.getFileHandle(parts[parts.length - 1]);
  };
  return {
    async list() {
      const out: string[] = [];
      for (const sub of ["images", "sparse/0"]) {
        let d = dir;
        for (const p of sub.split("/")) d = await d.getDirectoryHandle(p);
        for await (const [name, h] of d.entries()) if (h.kind === "file") out.push(`${sub}/${name}`);
      }
      return out.sort();
    },
    read: async (path) => (await at(path)).getFile(),
  };
}

const UPLOADS_AT_ONCE = 4;
/** The host cancels a job nobody asks about for 90 s; a network blip shorter
 *  than these retries goes unnoticed. */
const RETRY_MS = [1000, 2000, 4000, 8000, 15000];

const cancelled = () => new DOMException("cancelled", "AbortError");
const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => (clearTimeout(t), reject(cancelled())), { once: true });
  });

/** Why the host couldn't be used, in the log's plain English (the stage's
 *  error banner shows it as is). */
function explain(e: unknown, host: TrainHost): Error {
  if (!(e instanceof TrainHostError)) return e instanceof Error ? e : new Error(String(e));
  switch (e.reason) {
    case "network":
      return new Error(`the training host at ${host.url} didn't answer: is host/server.mjs running, and the address (or SSH tunnel) right?`);
    case "auth":
      return new Error("the training host refused the access token");
    case "notHost":
      return new Error(`${host.url} is not a Gausify Orbit training host`);
    case "http":
      return new Error(`training host: ${e.detail}`);
  }
}

/**
 * Trains `files` on the host and returns the final .ply. `masked`: the
 * images carry the mask alpha. Cancelling (or any failure here) also stops
 * the job on the host, freeing its GPU.
 */
export async function trainRemote(
  files: DatasetFiles,
  opts: TrainOptions,
  masked: boolean,
  host: TrainHost,
  ev: TrainEvents,
  signal: AbortSignal,
  client = new TrainHostClient(host),
): Promise<Uint8Array> {
  if (host.backend === "browser") throw new Error("trainRemote needs a host trainer");
  let id: string | null = null;
  // Previews download beside the event loop; training doesn't wait for them.
  const previews = new AbortController();
  signal.addEventListener("abort", () => previews.abort(), { once: true });
  try {
    if (signal.aborted) throw cancelled();
    ev.status({ key: "train.hostConnecting" });
    const info = await client.info(signal);
    const trainer = info.backends[host.backend];
    if (!trainer?.ok) throw new TrainHostError("http", trainer?.detail ?? `it has no "${host.backend}" trainer`);

    const jobId = (id = await client.createJob({ backend: host.backend, gpu: host.gpu, masked, options: opts, brushConfig: trainConfig(opts, masked) }, signal));

    const paths = await files.list();
    let sent = 0;
    ev.status({ key: "train.hostUploading", args: { i: 0, n: paths.length } });
    const queue = [...paths];
    await Promise.all(
      Array.from({ length: Math.min(UPLOADS_AT_ONCE, queue.length) }, async () => {
        for (let path = queue.shift(); path != null; path = queue.shift()) {
          await client.upload(jobId, path, await files.read(path), signal);
          ev.status({ key: "train.hostUploading", args: { i: ++sent, n: paths.length } });
        }
      }),
    );
    await client.start(jobId, signal);

    let next = 0;
    let failures = 0;
    let previewing = false;
    const total = Math.max(opts.totalSteps, 1);
    for (;;) {
      if (signal.aborted) throw cancelled();
      let batch: { events: HostEvent[]; next: number };
      try {
        batch = await client.events(jobId, next, signal);
        failures = 0;
      } catch (e) {
        if (signal.aborted) throw cancelled();
        // The job is still there after a blip; anything else is final.
        if (!(e instanceof TrainHostError) || e.reason !== "network" || failures >= RETRY_MS.length) throw e;
        await sleep(RETRY_MS[failures++], signal);
        continue;
      }
      next = batch.next;
      for (const e of batch.events) {
        switch (e.t) {
          case "queued":
            ev.status({ key: "train.hostQueued", args: { ahead: e.ahead } });
            break;
          case "started":
            ev.log(`training on the host: ${e.backend === "gsplat" ? "gsplat (CUDA)" : e.backend === "brush" ? "Brush (native)" : e.backend} on GPU ${e.gpu} (${e.gpuName})`);
            ev.status({ key: "train.loading" });
            break;
          case "dataset":
            ev.dataset(e.train, e.eval);
            break;
          case "step":
            ev.step(Math.min(e.iter, total), e.splats ?? 0);
            break;
          case "eval":
            ev.eval(e.iter, e.psnr, e.ssim);
            break;
          case "log":
            ev.log(e.line);
            break;
          case "snapshot":
            // The last step's file is the result, fetched below.
            if (e.iter >= total || previewing) break;
            previewing = true;
            client
              .snapshot(jobId, e.iter, previews.signal)
              .then((ply) => !previews.signal.aborted && ev.snapshot(e.iter, ply))
              // A preview replaced by a newer one on the host, or cut short.
              .catch(() => {})
              .finally(() => (previewing = false));
            break;
          case "done": {
            previews.abort();
            ev.status({ key: "train.hostDownloading" });
            return await client.result(jobId, signal);
          }
          case "failed":
            throw new Error(`training on the host failed: ${e.error}`);
          case "cancelled":
            throw new Error("the training host cancelled the job");
        }
      }
    }
  } catch (e) {
    if (signal.aborted) throw cancelled();
    throw explain(e, host);
  } finally {
    previews.abort();
    // Also how a cancelled job stops on the host.
    if (id) void client.remove(id).catch(() => {});
  }
}

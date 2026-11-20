// Training host (host/server.mjs, docs/host.md): the setting that chooses
// where training runs, stored in this browser only, and the HTTP client the
// app talks to the host with.

import { TRAIN_BACKENDS, type TrainHost } from "./api.ts";

const STORE_KEY = "splatforge.trainHost.v1";

export const DEFAULT_TRAIN_HOST: TrainHost = { backend: "browser", url: "http://localhost:1441", token: "", gpu: "auto" };

export function parseTrainHost(raw: unknown): TrainHost {
  const r = (raw && typeof raw === "object" ? raw : {}) as Partial<TrainHost>;
  return {
    backend: TRAIN_BACKENDS.includes(r.backend!) ? r.backend! : DEFAULT_TRAIN_HOST.backend,
    url: typeof r.url === "string" ? r.url : DEFAULT_TRAIN_HOST.url,
    token: typeof r.token === "string" ? r.token : "",
    gpu: typeof r.gpu === "number" && Number.isInteger(r.gpu) && r.gpu >= 0 ? r.gpu : "auto",
  };
}

export function loadTrainHost(): TrainHost {
  try {
    return parseTrainHost(JSON.parse(localStorage.getItem(STORE_KEY) ?? "null"));
  } catch {
    return DEFAULT_TRAIN_HOST;
  }
}

export function saveTrainHost(s: TrainHost) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(s));
  } catch {
    /* storage unavailable */
  }
}

/** Accepts "localhost:1441", "http://host:1441/", etc. */
export function normalizeHostUrl(raw: string): string {
  let url = raw.trim();
  if (!url) return "";
  if (!/^https?:\/\//i.test(url)) url = `http://${url}`;
  return url.replace(/\/+$/, "");
}

export interface HostGpu {
  index: number;
  name: string;
  memoryMb: number | null;
  usedMb: number | null;
  /** Training jobs on it right now. */
  jobs: number;
}

export interface HostInfo {
  name: string;
  version: string;
  gpus: HostGpu[];
  jobsPerGpu: number;
  /** Trainers installed on the host; `detail` says what is missing. */
  backends: Record<string, { ok: boolean; detail: string }>;
  queued: number;
}

/** What the host reports while a job runs (host/server.mjs). */
export type HostEvent =
  | { t: "queued"; ahead: number }
  | { t: "started"; gpu: number; gpuName: string; backend: string }
  | { t: "dataset"; train: number; eval: number }
  | { t: "step"; iter: number; splats: number }
  | { t: "eval"; iter: number; psnr: number; ssim: number }
  | { t: "snapshot"; iter: number }
  | { t: "log"; line: string }
  | { t: "done"; iter: number; seconds: number; bytes: number }
  | { t: "failed"; error: string }
  | { t: "cancelled" };

/** "network": the host didn't answer (not running, wrong address, tunnel
 *  down, or the browser blocked the request). "auth": missing or wrong token.
 *  "notHost": something else answers at that address. "http": the host
 *  refused, with its reason in `detail`. */
export type TrainHostErrorReason = "network" | "auth" | "notHost" | "http";

export class TrainHostError extends Error {
  readonly reason: TrainHostErrorReason;
  readonly detail: string;
  constructor(reason: TrainHostErrorReason, detail = "") {
    super(detail || reason);
    this.reason = reason;
    this.detail = detail;
  }
}

export interface NewHostJob {
  backend: "brush" | "gsplat" | "fake";
  gpu: number | "auto";
  /** The images carry the mask in their alpha. */
  masked: boolean;
  /** The app's TrainOptions. */
  options: object;
  /** The same options in Brush's own keys (trainConfig). */
  brushConfig: Record<string, unknown>;
}

export class TrainHostClient {
  readonly base: string;
  private readonly token: string;
  constructor(host: Pick<TrainHost, "url" | "token">) {
    this.base = normalizeHostUrl(host.url);
    this.token = host.token.trim();
  }

  private async req(path: string, init: RequestInit = {}): Promise<Response> {
    if (!this.base) throw new TrainHostError("network", "no address");
    let res: Response;
    try {
      res = await fetch(`${this.base}${path}`, {
        ...init,
        headers: { ...(init.headers as Record<string, string>), ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}) },
      });
    } catch (e) {
      if ((e as Error).name === "AbortError" || (e as Error).name === "TimeoutError") throw e;
      throw new TrainHostError("network", String((e as Error).message ?? e));
    }
    if (res.status === 401) throw new TrainHostError("auth");
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      let detail = text;
      try {
        detail = (JSON.parse(text) as { error?: string }).error ?? text;
      } catch {
        /* not JSON: another server */
      }
      throw new TrainHostError("http", detail || `HTTP ${res.status}`);
    }
    return res;
  }

  async info(signal?: AbortSignal): Promise<HostInfo> {
    const res = await this.req("/api/info", { signal });
    const info = (await res.json().catch(() => null)) as HostInfo | null;
    if (info?.name !== "gausify-host" || !Array.isArray(info.gpus)) throw new TrainHostError("notHost");
    return info;
  }

  async createJob(job: NewHostJob, signal?: AbortSignal): Promise<string> {
    const res = await this.req("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(job), signal });
    return ((await res.json()) as { id: string }).id;
  }

  /** `path` inside the dataset, e.g. "images/00001.jpg". */
  async upload(id: string, path: string, data: Blob, signal?: AbortSignal): Promise<void> {
    await this.req(`/api/jobs/${id}/files/${path.split("/").map(encodeURIComponent).join("/")}`, { method: "PUT", body: data, signal });
  }

  async start(id: string, signal?: AbortSignal): Promise<void> {
    await this.req(`/api/jobs/${id}/start`, { method: "POST", signal });
  }

  /** Events from index `after` on; waits on the host while there are none. */
  async events(id: string, after: number, signal?: AbortSignal): Promise<{ events: HostEvent[]; next: number }> {
    return (await this.req(`/api/jobs/${id}/events?after=${after}`, { signal })).json();
  }

  async snapshot(id: string, iter: number, signal?: AbortSignal): Promise<Uint8Array> {
    return new Uint8Array(await (await this.req(`/api/jobs/${id}/snapshot/${iter}`, { signal })).arrayBuffer());
  }

  async result(id: string, signal?: AbortSignal): Promise<Uint8Array> {
    return new Uint8Array(await (await this.req(`/api/jobs/${id}/result`, { signal })).arrayBuffer());
  }

  /** Stops the job and frees its GPU and files. */
  async remove(id: string): Promise<void> {
    await this.req(`/api/jobs/${id}`, { method: "DELETE" });
  }
}

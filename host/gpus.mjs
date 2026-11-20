// GPUs of the training host: discovery (nvidia-smi), pinning a process to one
// of them, and the pool that hands each training job a GPU of its own.

import { execFile } from "node:child_process";

/**
 * @typedef {object} Gpu
 * @property {number} index    nvidia-smi index (PCI bus order)
 * @property {string} name
 * @property {string | null} uuid  "GPU-…", as CUDA_VISIBLE_DEVICES takes it
 * @property {string | null} pci   "00000000:01:00.0"
 * @property {number | null} memoryMb
 * @property {number | null} usedMb
 * @property {number} ordinal  position among all of the machine's NVIDIA GPUs
 * @property {boolean} [pseudo] no nvidia-smi: whatever GPU the trainer picks
 */

const QUERY = "index,name,uuid,pci.bus_id,memory.total,memory.used";

/** `nvidia-smi --query-gpu=… --format=csv,noheader,nounits` → GPUs.
 *  @returns {Gpu[]} */
export function parseNvidiaSmi(csv) {
  const gpus = [];
  for (const line of csv.split("\n")) {
    const f = line.split(",").map((s) => s.trim());
    if (f.length < 6 || !/^\d+$/.test(f[0])) continue;
    const num = (s) => (/^\d+(\.\d+)?$/.test(s) ? Number(s) : null);
    gpus.push({
      index: Number(f[0]),
      name: f[1],
      uuid: f[2].startsWith("GPU-") ? f[2] : null,
      pci: /^[0-9a-f]+:[0-9a-f]+:[0-9a-f]+\.[0-9a-f]$/i.test(f[3]) ? f[3] : null,
      memoryMb: num(f[4]),
      usedMb: num(f[5]),
      ordinal: 0,
    });
  }
  gpus.sort((a, b) => a.index - b.index);
  gpus.forEach((g, i) => (g.ordinal = i));
  return gpus;
}

/** nvidia-smi's "00000000:01:00.0" → Mesa's device tag "pci-0000_01_00_0". */
export function pciTag(busId) {
  const m = /^([0-9a-f]+):([0-9a-f]+):([0-9a-f]+)\.([0-9a-f])$/i.exec(busId ?? "");
  if (!m) return null;
  const hex = (s, n) => parseInt(s, 16).toString(16).padStart(n, "0");
  return `pci-${hex(m[1], 4)}_${hex(m[2], 2)}_${hex(m[3], 2)}_${parseInt(m[4], 16)}`;
}

const PSEUDO = { index: 0, name: "default GPU", uuid: null, pci: null, memoryMb: null, usedMb: null, ordinal: 0, pseudo: true };

function smi() {
  return new Promise((resolve) => {
    execFile("nvidia-smi", [`--query-gpu=${QUERY}`, "--format=csv,noheader,nounits"], { timeout: 10_000 }, (err, stdout) =>
      resolve(err ? null : stdout),
    );
  });
}

/** The machine's NVIDIA GPUs, or one pseudo GPU (no pinning) without
 *  nvidia-smi. `only`: indices to keep (`--gpus 0,1`).
 *  @returns {Promise<Gpu[]>} */
export async function detectGpus(only = null) {
  const out = await smi();
  const all = out ? parseNvidiaSmi(out) : [];
  if (!all.length) return [{ ...PSEUDO }];
  const kept = only ? all.filter((g) => only.includes(g.index)) : all;
  if (!kept.length) throw new Error(`--gpus ${only.join(",")} matches none of the GPUs nvidia-smi lists (${all.map((g) => g.index).join(", ")})`);
  return kept;
}

/** Fresh memory figures for /api/info; the list itself doesn't change. */
export async function refreshMemory(gpus) {
  if (gpus[0]?.pseudo) return;
  const out = await smi();
  if (!out) return;
  const now = new Map(parseNvidiaSmi(out).map((g) => [g.index, g]));
  for (const g of gpus) {
    const n = now.get(g.index);
    if (n) [g.memoryMb, g.usedMb] = [n.memoryMb, n.usedMb];
  }
}

/**
 * Environment that makes a process use only `gpu`.
 * - "cuda" (gsplat): CUDA_VISIBLE_DEVICES by UUID, which doesn't depend on
 *   CUDA's device order.
 * - "wgpu" (native Brush): cubecl-wgpu's own override, the n-th discrete GPU
 *   in Vulkan's enumeration. NVIDIA enumerates in PCI order, like nvidia-smi.
 * - "vulkan" (Chrome, anything else on Vulkan): Mesa's device-select layer
 *   (mesa-vulkan-drivers) hides every device but the one with this PCI
 *   address. CUDA_VISIBLE_DEVICES does nothing for Vulkan.
 * @param {"cuda" | "wgpu" | "vulkan"} kind
 * @param {Gpu} gpu
 */
export function pinEnv(kind, gpu) {
  if (gpu.pseudo) return {};
  switch (kind) {
    case "cuda":
      return { CUDA_DEVICE_ORDER: "PCI_BUS_ID", CUDA_VISIBLE_DEVICES: gpu.uuid ?? String(gpu.index) };
    case "wgpu":
      return { CUBECL_WGPU_DEFAULT_DEVICE: `DiscreteGpu(${gpu.ordinal})` };
    case "vulkan": {
      const tag = pciTag(gpu.pci);
      return tag ? { DRI_PRIME: tag, MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE: "1" } : {};
    }
  }
  return {};
}

/** First device of `vulkaninfo --summary` under an environment. */
function firstVulkanDevice(env) {
  return new Promise((resolve) => {
    execFile("vulkaninfo", ["--summary"], { timeout: 15_000, env: { ...process.env, ...env } }, (err, stdout) => {
      if (err) return resolve(null);
      const uuid = /deviceUUID\s*=\s*([0-9a-f-]+)/i.exec(stdout)?.[1];
      const name = /deviceName\s*=\s*(.+)/.exec(stdout)?.[1]?.trim();
      resolve(uuid ? { uuid: uuid.toLowerCase(), name: name ?? "" } : null);
    });
  });
}

/**
 * Whether the "vulkan" pin really selects each GPU: under each GPU's
 * environment, Vulkan's first device must be that GPU (NVIDIA's deviceUUID is
 * the nvidia-smi UUID) and differ from the others'.
 * @returns {Promise<{ ok: boolean | null, detail: string }>} ok null = couldn't check
 */
export async function checkVulkanPin(gpus) {
  if (gpus[0]?.pseudo) return { ok: null, detail: "no nvidia-smi: GPUs can't be told apart, nothing is pinned" };
  const seen = [];
  for (const g of gpus) {
    const dev = await firstVulkanDevice(pinEnv("vulkan", g));
    if (!dev) return { ok: null, detail: "vulkaninfo not available (install vulkan-tools to verify)" };
    seen.push(dev.uuid);
    const want = g.uuid?.replace(/^GPU-/, "").toLowerCase();
    if (want && dev.uuid !== want) return { ok: false, detail: `GPU ${g.index}: Vulkan's first device is ${dev.name} (${dev.uuid}), expected ${want}` };
  }
  if (new Set(seen).size !== seen.length) return { ok: false, detail: "every GPU's environment selects the same Vulkan device (is mesa-vulkan-drivers installed?)" };
  return { ok: true, detail: `${gpus.length} GPU${gpus.length > 1 ? "s" : ""} selectable by PCI address` };
}

/**
 * Hands out GPUs: at most `perGpu` jobs on each, first come first served. A
 * request is for a given GPU index or "auto" (the free GPU with the fewest
 * jobs).
 */
export class GpuPool {
  /** @param {Gpu[]} gpus */
  constructor(gpus, perGpu = 1) {
    this.gpus = gpus;
    this.perGpu = Math.max(1, perGpu);
    /** @type {Map<number, Set<string>>} */
    this.load = new Map(gpus.map((g) => [g.index, new Set()]));
    /** @type {{ id: string, want: number | "auto", resolve: (g: Gpu) => void, reject: (e: Error) => void }[]} */
    this.waiting = [];
  }

  /** "auto", or an index this pool has; anything else is an error. */
  normalize(want) {
    if (want == null || want === "auto") return "auto";
    const n = Number(want);
    if (!this.load.has(n)) throw new Error(`no GPU ${want} on this host (available: ${this.gpus.map((g) => g.index).join(", ")})`);
    return n;
  }

  /** @returns {Gpu | null} */
  pick(want) {
    const free = this.gpus.filter((g) => this.load.get(g.index).size < this.perGpu && (want === "auto" || g.index === want));
    free.sort((a, b) => this.load.get(a.index).size - this.load.get(b.index).size || a.index - b.index);
    return free[0] ?? null;
  }

  /** Resolves with the job's GPU, right away or when one frees up.
   *  @returns {Promise<Gpu>} */
  acquire(id, want) {
    const w = this.normalize(want);
    // Whoever is already waiting goes first.
    const gpu = this.waiting.length ? null : this.pick(w);
    if (gpu) {
      this.load.get(gpu.index).add(id);
      return Promise.resolve(gpu);
    }
    return new Promise((resolve, reject) => this.waiting.push({ id, want: w, resolve, reject }));
  }

  /** Jobs ahead of `id` in the queue (0 = next); -1 if it isn't waiting. */
  position(id) {
    return this.waiting.findIndex((w) => w.id === id);
  }

  /** Frees the job's GPU, or drops it from the queue; then serves the queue. */
  release(id) {
    const i = this.waiting.findIndex((w) => w.id === id);
    if (i >= 0) this.waiting.splice(i, 1)[0].reject(new Error("cancelled"));
    for (const jobs of this.load.values()) jobs.delete(id);
    // In order, but a job waiting for a busy GPU doesn't hold back one that
    // can use a free GPU.
    for (let k = 0; k < this.waiting.length; ) {
      const w = this.waiting[k];
      const gpu = this.pick(w.want);
      if (!gpu) {
        k++;
        continue;
      }
      this.waiting.splice(k, 1);
      this.load.get(gpu.index).add(w.id);
      w.resolve(gpu);
    }
  }

  jobsOn(index) {
    return [...(this.load.get(index) ?? [])];
  }
}

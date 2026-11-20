// The two trainers the host runs: native Brush (the pinned commit with the
// app's patches, on Vulkan) and gsplat (CUDA, host/train_gsplat.py). Command
// lines and the events read back from their output.

import { readdirSync } from "node:fs";
import { join } from "node:path";

/** Brush options the app may not set: where files are written, the viewer. */
const BRUSH_RESERVED = /^(export-|eval-save-to-disk$|rerun|with-viewer$|start-iter$)/;

/**
 * `brush-cli` arguments for a config in Brush's own keys (kebab-case, what
 * `trainConfig` in web/src/pipeline/train-config.ts produces): `null` keeps
 * Brush's default, arrays become comma lists.
 * @param {Record<string, unknown>} config
 */
export function brushArgs(dataset, outDir, config, exportEvery) {
  const args = [dataset];
  for (const [key, value] of Object.entries(config)) {
    if (!/^[a-z][a-z0-9-]*$/.test(key)) throw new Error(`invalid Brush option "${key}"`);
    if (BRUSH_RESERVED.test(key)) throw new Error(`Brush option "${key}" is set by the host`);
    if (value == null) continue;
    const scalar = (v) => {
      if (typeof v === "number" && Number.isFinite(v)) return String(v);
      if (typeof v === "boolean") return String(v);
      throw new Error(`invalid value for Brush option "${key}"`);
    };
    args.push(`--${key}`, Array.isArray(value) ? value.map(scalar).join(",") : scalar(value));
  }
  args.push("--export-every", String(Math.max(1, Math.round(exportEvery))), "--export-path", outDir, "--export-name", "splat_{iter}.ply");
  return args;
}

/** Brush only prints progress bars on a TTY; these modules log the lines
 *  parsed below. */
export const BRUSH_ENV = { RUST_LOG: "brush_process=info,brush_cli=info,brush_train=info", NO_COLOR: "1" };

/** Drops env_logger's "[2026-10-04T19:51:15Z INFO  module] " prefix. */
function stripLogger(line) {
  if (line.startsWith("[")) {
    const i = line.indexOf("] ");
    if (i >= 0) return line.slice(i + 2);
  }
  return line;
}

/**
 * One line of Brush output → event, or null for noise.
 * @returns {{ t: "step", iter: number, splats?: number } | { t: "eval", iter: number, psnr: number, ssim: number }
 *   | { t: "dataset", train: number, eval: number } | { t: "log", line: string } | null}
 */
export function parseBrushLine(raw) {
  const line = stripLogger(raw.trimEnd());
  if (!line.trim()) return null;
  let m = /Refine iter (\d+), (\d+) splats/.exec(line);
  if (m) return { t: "step", iter: Number(m[1]), splats: Number(m[2]) };
  // From host/brush/progress.patch: refine lines stop once growth does.
  m = /^Step iter (\d+)(?:, (\d+) splats)?/.exec(line);
  if (m) return m[2] ? { t: "step", iter: Number(m[1]), splats: Number(m[2]) } : { t: "step", iter: Number(m[1]) };
  m = /Eval iter (\d+): PSNR ([\d.eE+-]+), ssim ([\d.eE+-]+)/.exec(line);
  if (m) return { t: "eval", iter: Number(m[1]), psnr: Number(m[2]), ssim: Number(m[3]) };
  m = /Loaded dataset with (\d+) training, (\d+) eval views/.exec(line);
  if (m) return { t: "dataset", train: Number(m[1]), eval: Number(m[2]) };
  if (line.includes("load_scene_img") || line.includes("Loaded COLMAP image")) return null;
  return { t: "log", line };
}

/** `splat_<iter>.ply` in a folder, by ascending iteration. Brush zero-pads
 *  the iteration to the total's digits, so the folder is scanned. */
export function listSnapshots(dir) {
  let names = [];
  try {
    names = readdirSync(dir);
  } catch {
    return [];
  }
  return names
    .map((n) => ({ m: /^splat_(\d+)\.ply$/.exec(n), n }))
    .filter((x) => x.m)
    .map((x) => ({ iter: Number(x.m[1]), path: join(dir, x.n) }))
    .sort((a, b) => a.iter - b.iter);
}

/** What train_gsplat.py takes from the app's TrainOptions; the rest is
 *  Brush-only and listed in `gsplatIgnored`. */
const num = (v, def) => (typeof v === "number" && Number.isFinite(v) ? v : def);

/** @param {Record<string, unknown>} o the app's TrainOptions */
export function gsplatArgs(script, dataset, outDir, o, masked) {
  const steps = Math.max(1, Math.round(num(o.totalSteps, 7000)));
  const args = [
    script,
    "--data", dataset,
    "--out", outDir,
    "--steps", String(steps),
    "--sh-degree", String(Math.min(3, Math.max(0, Math.round(num(o.shDegree, 1))))),
    "--max-splats", String(Math.max(1000, Math.round(num(o.maxSplats, 2_000_000)))),
    "--max-resolution", String(Math.max(64, Math.round(num(o.maxResolution, 1920)))),
    "--ssim-weight", String(Math.max(0, num(o.ssimWeight, 0.2))),
    "--export-every", String(Math.round(num(o.exportEvery, 0)) || steps),
    "--eval-split-every", String(Math.max(0, Math.round(num(o.evalSplitEvery, 0)))),
  ];
  if (o.lrMean != null) args.push("--lr-mean", String(num(o.lrMean, 1.6e-4)));
  if (o.growthStopIter != null) args.push("--growth-stop-iter", String(Math.round(num(o.growthStopIter, steps / 2))));
  if (masked) args.push("--masked");
  return args;
}

/** Options set in the app that gsplat training doesn't have (they're Brush
 *  patches): reported in the log instead of silently dropped. */
export function gsplatIgnored(o) {
  const on = (v) => typeof v === "number" && v > 0;
  return [
    on(o.depthDistortionWeight) && "Reduce haze (depth distortion)",
    on(o.contribPruneMinViews) && "contribution pruning",
    on(o.opacSolidWeight) && "opacity (solid) regularization",
    on(o.opacBinaryWeight) && "opacity (binary) regularization",
    on(o.volumeRegWeight) && "volume regularization",
  ].filter(Boolean);
}

/** train_gsplat.py prints one JSON object per line; anything else is a log
 *  line (Python warnings, tracebacks). */
export function parseGsplatLine(raw) {
  const line = raw.trimEnd();
  if (!line.trim()) return null;
  if (line.startsWith("{")) {
    try {
      const e = JSON.parse(line);
      if (e && typeof e.t === "string") return e;
    } catch {
      /* a log line that happens to start with a brace */
    }
  }
  return { t: "log", line };
}

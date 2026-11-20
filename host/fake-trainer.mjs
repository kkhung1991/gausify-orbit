#!/usr/bin/env node
// Stand-in trainer for developing and testing the host without a GPU
// (`node host/server.mjs --fake`): speaks train_gsplat.py's output protocol
// and writes a small valid splat at each export.
//
//   node host/fake-trainer.mjs <dataset> <out> <steps> <export-every>
//
// FAKE_STEP_MS: delay per 10 steps (default 20). FAKE_FAIL=1: dies midway.

import { existsSync, mkdirSync, readdirSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [dataset, out, stepsArg, everyArg] = process.argv.slice(2);
const steps = Number(stepsArg);
const every = Number(everyArg) || steps;
const stepMs = Number(process.env.FAKE_STEP_MS ?? 20);
const say = (e) => console.log(JSON.stringify(e));

/** A ring of `n` Gaussians in the 3DGS .ply layout (SH degree 0). */
function ply(n) {
  const props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"];
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${n}\n${props.map((p) => `property float ${p}`).join("\n")}\nend_header\n`;
  const data = new Float32Array(n * props.length);
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    data.set([Math.cos(a), Math.sin(a), 0, 0.5, 0.2, -0.2, 2, -3, -3, -3, 1, 0, 0, 0], i * props.length);
  }
  return Buffer.concat([Buffer.from(header), Buffer.from(data.buffer)]);
}

if (!existsSync(join(dataset, "sparse", "0", "images.bin"))) {
  console.error("no sparse/0/images.bin in the dataset");
  process.exit(1);
}
const images = readdirSync(join(dataset, "images")).length;
mkdirSync(out, { recursive: true });
say({ t: "dataset", train: images, eval: 0 });

for (let iter = 10; ; iter += 10) {
  const at = Math.min(iter, steps);
  await new Promise((r) => setTimeout(r, stepMs));
  if (process.env.FAKE_FAIL && at >= steps / 2) {
    console.error("RuntimeError: CUDA out of memory (simulated)");
    process.exit(1);
  }
  const splats = 1000 + at * 3;
  say({ t: "step", iter: at, splats });
  if (at % every === 0 || at === steps) {
    const file = join(out, `splat_${String(at).padStart(String(steps).length, "0")}.ply`);
    writeFileSync(`${file}.tmp`, ply(64));
    renameSync(`${file}.tmp`, file);
    say({ t: "snapshot", iter: at });
  }
  if (at === steps) break;
}
say({ t: "done", iter: steps });

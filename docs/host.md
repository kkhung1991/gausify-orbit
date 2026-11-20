# Training on a GPU host

Gausify Orbit runs in the browser, on the GPU of the machine the browser is
on. A Linux machine with NVIDIA GPUs, including one you only reach over SSH,
can do the training instead, in three ways. **Train on** (advanced settings →
Training) chooses between them:

| Train on | Where training runs | GPU API | The host needs |
|---|---|---|---|
| **This browser (WebGPU)** | Brush as WebAssembly, in the page | WebGPU (Vulkan on Linux) | Chrome on the host, for its GPUs to be the ones used |
| **Host: Brush (Vulkan)** | the same Brush, native, started by the training host | Vulkan | `pnpm host:setup --brush` |
| **Host: gsplat (CUDA)** | [gsplat](https://github.com/nerfstudio-project/gsplat), started by the training host | CUDA | `pnpm host:setup --gsplat` |

With the two host options, frames, cutout and poses still run in the browser
you are using (your laptop is fine); when the Training stage starts, the page
uploads the dataset to the host, shows its progress and previews, and
downloads the splat. Everything after that (export, mesh, retouch, share)
works as usual.

None of them splits one training run across GPUs: an orbit is a small scene,
and one GPU trains it in minutes. Two GPUs train **two videos at once**: the
host gives each job its own GPU. In the app, "Process all" still handles one
video after another, so to keep both GPUs busy open the app in two tabs with
different videos, or use the [headless batch](#without-a-display-one-chrome-per-gpu).

## The training host

`host/server.mjs` is a small HTTP server (Node, no dependencies) that runs on
the machine with the GPUs, takes training jobs from the page and gives each
one a GPU.

```bash
pnpm host:setup --check     # what the machine has and what is missing
pnpm host:setup             # builds native Brush, the gsplat environment and the app; tests the trainers
pnpm host                   # the host and the app, on http://127.0.0.1:1441
```

`host/setup.sh` sets up what lives in this folder and tells you about the
rest:

| | Set up by `pnpm host:setup` | You install (it prints the command) |
|---|---|---|
| Both | the app's packages (`pnpm install`) | NVIDIA driver, Node 22.18+, pnpm, a C/C++ compiler |
| Native Brush | `host/bin/brush`, built from the pinned commit and patches (~15 min, once) | Rust (rustup) |
| gsplat | `host/.venv` with a PyTorch build matching the machine's CUDA toolkit, gsplat and its CUDA code compiled | Python 3 with venv, **the CUDA toolkit (`nvcc`)** |
| The app | `dist-web`, the page with the **Train on** setting (Brush as WebAssembly, ~25 min, once; fetches `wasm-pack` with cargo) | Rust (rustup) |
| Headless Chrome per GPU | | Chrome, `mesa-vulkan-drivers`, `vulkan-tools` |

The driver is not enough for gsplat: it compiles its CUDA kernels on the
machine the first time it runs, which takes the CUDA toolkit. Native Brush
needs no CUDA at all, only the driver's Vulkan. `--apt` lets the script run
`sudo apt-get install` for the missing system packages (never the driver);
`--brush`, `--gsplat` and `--app` set up one part only. At the end it renders a
test image with gsplat on the GPU and prints what `node host/server.mjs
--check` sees. "Ready" means gsplat rendered on the GPU and the Brush binary
is built; Brush itself first touches the GPU in its first job.

Either trainer is enough; the page says which ones the host has. Native Brush
is built from the same pinned commit and patches as the browser's
(`scripts/build-brush-web.mjs`), so it takes every training option of the
app. `host/brush/*.patch` only goes into the native build: a plain progress
line, because Brush draws its progress bar on a terminal only.

### Reaching it from your computer

The host listens on loopback. From another computer, forward the port:

```bash
ssh -L 1441:localhost:1441 user@gpu-host
```

and open `http://localhost:1441`: once the app is built, the host serves it
on the same port, so the page and the host share an address and **Host
address** can stay at `http://localhost:1441`. The tunnel also keeps the page
a secure context, which WebGPU and the browser's storage require.

The **Train on** setting is part of this repository's app, so use a page
built from it: the one the host serves, `pnpm web:dev`, or your own
deployment. A page served over HTTPS can reach the host too, but only at
`http://localhost` (through the tunnel); browsers block plain HTTP to any
other address from a secure page, and Chrome may ask once for permission to
reach local devices. `--listen 0.0.0.0` exposes the port directly and
requires `--token <secret>`, entered as **Access token**.

### GPUs

GPUs come from `nvidia-smi`. **Host GPU** is *Automatic* (the first free GPU)
or a specific one. Each job has a GPU to itself (`--jobs-per-gpu` changes
that), and a job that finds them all busy waits in a queue, in order. Closing
the page, cancelling, or losing the connection for 90 s stops the job and
frees its GPU. `--gpus 0,1` restricts the host to some of the machine's GPUs,
for instance to leave one to ComfyUI.

Each trainer is confined to its GPU its own way, since two identical cards
only differ by address:

| Trainer | Pinned with |
|---|---|
| gsplat | `CUDA_VISIBLE_DEVICES=<GPU UUID>` |
| Native Brush | `CUBECL_WGPU_DEFAULT_DEVICE=DiscreteGpu(n)`, the n-th discrete GPU Vulkan lists (PCI order on NVIDIA, like `nvidia-smi`) |
| Chrome (`--gpu`, below) | Mesa's device-select layer: `DRI_PRIME=pci-0000_01_00_0` with `MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1`, from `mesa-vulkan-drivers` |

`CUDA_VISIBLE_DEVICES` has no effect on Vulkan, which is why Chrome and Brush
need the other two. `node host/server.mjs --check` verifies the Chrome one
with `vulkaninfo`: under each GPU's environment, Vulkan's first device must
be that GPU. To see which GPU a job landed on, watch `nvidia-smi` while it
trains.

### With gsplat

`host/train_gsplat.py` follows gsplat's reference trainer (3DGS defaults, the
default densification strategy) with the schedule scaled to the number of
steps. It takes the app's steps, SH degree, maximum splats, maximum
resolution, SSIM, preview interval and PSNR evaluation, and trains cutouts
over a background colour drawn every step, like the app's Brush. The options
that are patches to Brush don't exist there: **Reduce haze** and the
experimental opacity, volume and contribution options are ignored, and the
log says so. The pose check, mesh export and retouch fine-tuning keep using
Brush in the browser whatever this setting says.

## Without a display: one Chrome per GPU

`scripts/web-bench/headless.mjs` drives the app in headless Chrome, and on
Linux it starts Chrome on Vulkan. `--gpu N` runs that Chrome on the N-th GPU,
and `scripts/host-batch.mjs` uses it to work through a list of videos with
one Chrome per GPU:

```bash
WEB_FS_ANY=1 pnpm web:dev &               # the app, with access to local files
node scripts/host-batch.mjs ~/orbits/*.mp4 --out ~/splats -- --preset balanced --mask sam2
```

Each video's project (frames, poses, `splat.ply`) lands in
`<out>/<video name>/`. `--backend brush` or `--backend gsplat` sends the
training of each one to the training host (`pnpm host` running), on the same
GPU as its Chrome; the default trains in Chrome itself. Options after `--` go
to `headless.mjs` (preset, masking, steps, `--export`…). Chrome comes from
`CHROME`, or `google-chrome`/`chromium` on the `PATH`.

To use the app interactively with Chrome on the host, any remote desktop
works. If `chrome://gpu` shows WebGPU disabled, start Chrome with
`--enable-unsafe-webgpu --enable-features=Vulkan`.

## What has been run, and what hasn't

Developed on a machine without a GPU, on software Vulkan (Mesa's llvmpipe,
Chrome's SwiftShader):

- **Host: Brush.** The whole path ran: the app in headless Chrome on Linux
  did frames and poses (40/40 cameras on a synthetic orbit), uploaded the
  dataset, native Brush trained it and the splat came back with its previews
  and PSNR. Native Brush builds from the pinned commit with every patch.
- **Host: gsplat.** `train_gsplat.py` ran end to end on CPU with gsplat's
  rasterizer replaced by a stand-in (data loading, cutouts, densification,
  evaluation, `.ply` export, and the app's reader loading the result). The
  CUDA rasterizer in the training loop has never run.
- **This browser, on Linux.** Headless Chrome with the Vulkan flags gets a
  WebGPU adapter and the pipeline runs to the end, `host-batch.mjs` included.
  On SwiftShader the in-browser training ends with a wrong result, though
  (PSNR 7 dB where native Brush reaches 18 dB with the same poses and
  settings). That training path is unchanged and had only ever run on Metal,
  so whether it is the software renderer or Linux Vulkan in general is open.
- **The host itself:** API, queue, cancellation and abandoned jobs, with a
  stand-in trainer (`pnpm test:web`, `host/host.test.ts`).
- **Not run at all:** anything on an NVIDIA GPU. GPU selection on a real
  two-GPU machine (all three mechanisms) and every speed figure are
  untested.

`host/setup.sh` ran here as far as a machine without a GPU allows: the
checks, the Brush build, and the gsplat environment up to its GPU test.

So on the real machine: run `pnpm host:setup`, then train one
video with each option and compare the three splats and their PSNR (turn on
**Measure quality**) before relying on any of them.

`node host/server.mjs --fake` adds a trainer that needs no GPU, for working
on the integration.

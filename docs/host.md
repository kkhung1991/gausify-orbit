# Training on a GPU host

Gausify Orbit runs in the browser, on the GPU of the machine the browser is
on. A Linux machine with NVIDIA GPUs, including one you only reach over SSH,
can do the training instead, in three ways. **Train on** (advanced settings →
Training) chooses between them:

| Train on | Where training runs | GPU API | The host needs |
|---|---|---|---|
| **This browser (WebGPU)** | Brush as WebAssembly, in the page | WebGPU (Vulkan on Linux) | Chrome on the host, for its GPUs to be the ones used |
| **Host: Brush (Vulkan)** | the same Brush, native, started by the training host | Vulkan | `pnpm host:brush` |
| **Host: gsplat (CUDA)** | [gsplat](https://github.com/nerfstudio-project/gsplat), started by the training host | CUDA | `pip install -r host/requirements.txt` |

With the two host options, frames, cutout and poses still run in the browser
you are using (your laptop is fine); when the Training stage starts, the page
uploads the dataset to the host, shows its progress and previews, and
downloads the splat. Everything after that (export, mesh, retouch, share)
works as usual.

None of them splits one training run across GPUs: an orbit is a small scene,
and one GPU trains it in minutes. Two GPUs train **two videos at once**.

## The training host

`host/server.mjs` is a small HTTP server (Node, no dependencies) that runs on
the machine with the GPUs, takes training jobs from the page and gives each
one a GPU.

```bash
pnpm install
pnpm host:brush                           # native Brush → host/bin/brush (Rust; ~15 min, once)
pip install -r host/requirements.txt      # gsplat (a CUDA build of PyTorch)
node host/server.mjs --check              # lists the GPUs and which trainers are ready
pnpm host                                 # http://127.0.0.1:1441
```

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

and leave **Host address** at `http://localhost:1441`. The tunnel is also
what makes it work from `https://orbit.gausify.com`: browsers let a secure
page call `http://localhost`, but not plain HTTP anywhere else (Chrome may
ask once for permission to reach local devices). `--listen 0.0.0.0` exposes
the port directly and requires `--token <secret>`, entered as **Access
token**; that only works for a page that is itself served over HTTP on
localhost.

If the app was built on the host (`pnpm web:wasm && pnpm web:build`), the
host serves it too: open `http://localhost:1441` through the same tunnel.

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

Developed on a machine without a GPU:

- **Run:** the host's API, queue and cancellation, with a stand-in trainer
  (`pnpm test:web`, `host/host.test.ts`); native Brush built from the pinned
  commit with all patches and training a small synthetic orbit through the
  host on a software Vulkan device; `train_gsplat.py` end to end on CPU with
  gsplat's rasterizer replaced by a stand-in (data loading, densification,
  evaluation, `.ply` export); headless Chrome on Linux getting a WebGPU
  adapter with the Vulkan flags.
- **Not run:** anything on an NVIDIA GPU. That leaves gsplat's CUDA
  rasterizer in the training loop, GPU selection on a real two-GPU machine
  (all three mechanisms), and every speed figure. Run
  `node host/server.mjs --check` first, and compare a first splat from each
  trainer against the browser's on the same video.

`node host/server.mjs --fake` adds a trainer that needs no GPU, for working
on the integration.

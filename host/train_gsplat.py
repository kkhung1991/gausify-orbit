#!/usr/bin/env python3
"""gsplat (CUDA) trainer of the Gausify Orbit training host (docs/host.md).

Trains a COLMAP dataset as the app lays it out (images/ + sparse/0, RGBA
images when the subject is cut out) and writes splats in the 3DGS .ply layout
the app reads. host/server.mjs starts it with one GPU visible
(CUDA_VISIBLE_DEVICES) and reads its output: one JSON object per line.

  python host/train_gsplat.py --data <dataset> --out <folder> --steps 7000
      [--sh-degree 1] [--max-splats 2000000] [--max-resolution 1920]
      [--ssim-weight 0.2] [--export-every 2500] [--eval-split-every 0]
      [--lr-mean 1.6e-4] [--growth-stop-iter N] [--masked] [--seed 42]

Follows gsplat's examples/simple_trainer.py (3DGS defaults, DefaultStrategy),
with the schedule scaled to the run's length. Masked runs train over a
background colour drawn every step, like the app's Brush (see
MASKED_BACKGROUND in web/src/pipeline/train-config.ts): on a fixed colour,
semi-transparent layers outside the silhouette cost nothing.
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np


def say(**event):
    print(json.dumps(event), flush=True)


# ---------- COLMAP binary model ----------

# model id → (name, number of parameters, number of focal lengths)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3, 1),
    1: ("PINHOLE", 4, 2),
    2: ("SIMPLE_RADIAL", 4, 1),
    3: ("RADIAL", 5, 1),
    4: ("OPENCV", 8, 2),
}


class Reader:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        self.o = 0

    def take(self, fmt):
        size = struct.calcsize(fmt)
        values = struct.unpack_from(fmt, self.data, self.o)
        self.o += size
        return values

    def string(self):
        end = self.data.index(b"\0", self.o)
        s = self.data[self.o:end].decode("utf-8")
        self.o = end + 1
        return s


def read_cameras(path):
    """camera id → dict(width, height, fx, fy, cx, cy)."""
    r = Reader(path)
    cameras = {}
    (count,) = r.take("<Q")
    for _ in range(count):
        cam_id, model = r.take("<Ii")
        width, height = r.take("<QQ")
        if model not in CAMERA_MODELS:
            raise ValueError(f"camera model {model} is not supported (pinhole and radial models only)")
        name, n_params, n_focal = CAMERA_MODELS[model]
        p = r.take(f"<{n_params}d")
        fx, fy = (p[0], p[0]) if n_focal == 1 else (p[0], p[1])
        cx, cy = p[n_focal], p[n_focal + 1]
        distortion = p[n_focal + 2:]
        if any(abs(k) > 1e-6 for k in distortion):
            say(t="log", line=f"camera {cam_id} ({name}) has lens distortion, which is ignored: undistort the images for best results")
        cameras[cam_id] = dict(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    return cameras


def read_images(path):
    """[(name, camera id, world→camera 4×4)], by name."""
    r = Reader(path)
    images = []
    (count,) = r.take("<Q")
    for _ in range(count):
        (_image_id,) = r.take("<I")
        qw, qx, qy, qz = r.take("<4d")
        t = r.take("<3d")
        (cam_id,) = r.take("<I")
        name = r.string()
        (n_points,) = r.take("<Q")
        r.o += 24 * n_points
        n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ]
        m[:3, 3] = t
        images.append((name, cam_id, m))
    images.sort(key=lambda im: im[0])
    return images


def read_points(path):
    """(xyz [N,3] float32, rgb [N,3] uint8); empty arrays without the file."""
    if not os.path.exists(path):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    r = Reader(path)
    (count,) = r.take("<Q")
    xyz = np.zeros((count, 3), np.float32)
    rgb = np.zeros((count, 3), np.uint8)
    for i in range(count):
        (_point_id,) = r.take("<Q")
        xyz[i] = r.take("<3d")
        rgb[i] = r.take("<3B")
        (_error,) = r.take("<d")
        (track,) = r.take("<Q")
        r.o += 8 * track
    return xyz, rgb


# ---------- dataset ----------


def load_dataset(data, max_resolution, masked):
    """Views of the dataset: name, uint8 image [H,W,3 or 4], K [3,3], world→camera."""
    from PIL import Image

    model = os.path.join(data, "sparse", "0")
    cameras = read_cameras(os.path.join(model, "cameras.bin"))
    views = []
    for name, cam_id, w2c in read_images(os.path.join(model, "images.bin")):
        path = os.path.join(data, "images", name)
        if not os.path.exists(path):
            say(t="log", line=f"{name} is in the model but not in images/: skipped")
            continue
        cam = cameras[cam_id]
        with Image.open(path) as im:
            im.load()
            # Premultiplied while resizing, so transparent pixels don't bleed in.
            im = im.convert("RGBa") if masked and "A" in im.getbands() else im.convert("RGB")
            scale = min(1.0, max_resolution / max(im.size))
            if scale < 1.0:
                im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
            pixels = np.array(im.convert("RGBA") if im.mode == "RGBa" else im)
        height, width = pixels.shape[:2]
        # The images may be smaller than the model's camera (downscaled datasets).
        sx, sy = width / cam["width"], height / cam["height"]
        K = np.array([[cam["fx"] * sx, 0, cam["cx"] * sx], [0, cam["fy"] * sy, cam["cy"] * sy], [0, 0, 1]], np.float32)
        views.append(dict(name=name, image=pixels, K=K, w2c=w2c.astype(np.float32)))
    if len(views) < 3:
        raise ValueError(f"only {len(views)} posed images found in {data}")
    return views


def split_views(views, eval_split_every):
    """Holds out one view in every `eval_split_every` (0 or 1 = none)."""
    if eval_split_every <= 1:
        return views, []
    train = [v for i, v in enumerate(views) if i % eval_split_every != 0]
    held = [v for i, v in enumerate(views) if i % eval_split_every == 0]
    return (train, held) if len(train) >= 3 else (views, [])


def camera_extent(views):
    """Largest distance of a camera from the cameras' centre."""
    centres = np.stack([-v["w2c"][:3, :3].T @ v["w2c"][:3, 3] for v in views])
    return float(np.linalg.norm(centres - centres.mean(0), axis=1).max()), centres


# ---------- model ----------

SH_C0 = 0.28209479177387814


def knn_scale(points, k=3, chunk=4096):
    """Mean distance to the k nearest neighbours, per point."""
    import torch

    out = torch.empty(points.shape[0], device=points.device)
    for i in range(0, points.shape[0], chunk):
        d = torch.cdist(points[i:i + chunk], points)
        out[i:i + chunk] = d.topk(min(k + 1, points.shape[0]), largest=False).values[:, 1:].mean(1)
    return out


def create_splats(xyz, rgb, sh_degree, extent, centres, device, seed):
    import torch

    g = torch.Generator().manual_seed(seed)
    if len(xyz) < 16:
        # No sparse points: a cloud inside the cameras' orbit.
        n = 20_000
        centre = torch.from_numpy(centres.mean(0)).float()
        xyz_t = centre + (torch.rand(n, 3, generator=g) - 0.5) * extent
        rgb_t = torch.rand(n, 3, generator=g)
    else:
        xyz_t = torch.from_numpy(xyz).float()
        rgb_t = torch.from_numpy(rgb).float() / 255.0
    xyz_t = xyz_t.to(device)
    n = xyz_t.shape[0]
    scales = torch.log(knn_scale(xyz_t).clamp_min(1e-7))[:, None].repeat(1, 3)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((n,), 0.1, device=device))
    sh0 = ((rgb_t.to(device) - 0.5) / SH_C0)[:, None, :]
    shN = torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3, device=device)
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(xyz_t),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    })


def write_ply(path, params):
    """3DGS .ply: logit opacity, log scales, SH with all of a channel's
    coefficients together. Written beside the target and renamed, so a
    reader never sees half a file."""
    import torch

    with torch.no_grad():
        n = params["means"].shape[0]
        rest = params["shN"].permute(0, 2, 1).reshape(n, -1)
        quats = torch.nn.functional.normalize(params["quats"], dim=-1)
        cols = torch.cat([params["means"], params["sh0"].reshape(n, 3), rest, params["opacities"][:, None], params["scales"], quats], dim=1)
        data = cols.detach().float().cpu().numpy().astype("<f4")
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(rest.shape[1])]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + "".join(f"property float {p}\n" for p in names) + "end_header\n"
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    os.replace(tmp, path)


# ---------- losses ----------


def ssim(a, b, window=11, sigma=1.5):
    """Mean SSIM of two [1,3,H,W] images in 0–1."""
    import torch
    import torch.nn.functional as F

    window = min(window, a.shape[-1], a.shape[-2])
    window -= 1 - window % 2
    x = torch.arange(window, device=a.device, dtype=a.dtype) - window // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(3, 1, window, window).contiguous()
    blur = lambda t: F.conv2d(t, kernel, groups=3)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))).mean()


def target_over(image, background):
    """[H,W,3] float target from a uint8 view image; RGBA goes over `background`."""
    pixels = image.float() / 255.0
    if pixels.shape[-1] == 4:
        alpha = pixels[..., 3:4]
        return pixels[..., :3] * alpha + background * (1.0 - alpha)
    return pixels


# ---------- training ----------


def train(args, rasterize=None, device="cuda"):
    import torch

    if rasterize is None:
        from gsplat.rendering import rasterization as rasterize
    from gsplat.strategy import DefaultStrategy

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    steps = max(1, args.steps)

    views = load_dataset(args.data, args.max_resolution, args.masked)
    train_views, eval_views = split_views(views, args.eval_split_every)
    say(t="dataset", train=len(train_views), eval=len(eval_views))
    extent, centres = camera_extent(views)
    scene_scale = max(extent, 1e-3) * 1.1
    for v in views:
        v["image"] = torch.from_numpy(v["image"]).to(device)
        v["K"] = torch.from_numpy(v["K"]).to(device)
        v["w2c"] = torch.from_numpy(v["w2c"]).to(device)

    xyz, rgb = read_points(os.path.join(args.data, "sparse", "0", "points3D.bin"))
    params = create_splats(xyz, rgb, args.sh_degree, scene_scale, centres, device, args.seed)

    lrs = {"means": args.lr_mean * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    means_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / steps))

    # 3DGS grows for the first half of 30k steps and resets opacity every 3k;
    # shorter runs keep the proportions.
    grow_stop = args.growth_stop_iter if args.growth_stop_iter is not None else steps // 2
    grow_stop = max(1, min(grow_stop, steps))
    strategy = DefaultStrategy(
        verbose=False,
        refine_start_iter=min(500, max(1, grow_stop // 5)),
        refine_stop_iter=grow_stop,
        refine_every=100 if grow_stop >= 1000 else max(1, grow_stop // 10),
        reset_every=3000 if grow_stop > 3000 else grow_stop + 1,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    grow_threshold = strategy.grow_grad2d
    sh_interval = min(1000, max(1, steps // 10))

    def render(view, sh_degree, background=None):
        height, width = view["image"].shape[:2]
        return rasterize(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=torch.cat([params["sh0"], params["shN"]], dim=1),
            viewmats=view["w2c"][None],
            Ks=view["K"][None],
            width=width,
            height=height,
            sh_degree=sh_degree,
            backgrounds=background[None] if background is not None else None,
            packed=False,
        )

    def evaluate(step):
        if not eval_views:
            return
        psnrs, ssims = [], []
        with torch.no_grad():
            black = torch.zeros(3, device=device)
            for v in eval_views:
                colors, _, _ = render(v, args.sh_degree, black if args.masked else None)
                out = colors[0].clamp(0, 1)
                gt = target_over(v["image"], black)
                mse = torch.mean((out - gt) ** 2).clamp_min(1e-10)
                psnrs.append(float(-10 * torch.log10(mse)))
                ssims.append(float(ssim(out.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
        say(t="eval", iter=step, psnr=sum(psnrs) / len(psnrs), ssim=sum(ssims) / len(ssims))

    def export(step):
        write_ply(os.path.join(args.out, f"splat_{step:0{len(str(steps))}d}.ply"), params)
        say(t="snapshot", iter=step)

    started = time.time()
    last_report = 0.0
    order = []
    for step in range(1, steps + 1):
        if not order:
            order = list(np.random.permutation(len(train_views)))
        view = train_views[order.pop()]
        background = torch.rand(3, device=device) if args.masked else None
        colors, _alphas, info = render(view, min(args.sh_degree, (step - 1) // sh_interval), background)
        out = colors[0]
        gt = target_over(view["image"], background)
        loss = torch.abs(out - gt).mean()
        if args.ssim_weight > 0:
            loss = (1 - args.ssim_weight) * loss + args.ssim_weight * (1 - ssim(out.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))

        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_schedule.step()
        # At the cap the strategy only prunes.
        strategy.grow_grad2d = grow_threshold if params["means"].shape[0] < args.max_splats else float("inf")
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)

        if not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"the loss became {float(loss.detach())} at step {step}")
        now = time.time()
        if step == steps or (step % 10 == 0 and now - last_report >= 0.2):
            last_report = now
            say(t="step", iter=step, splats=int(params["means"].shape[0]))
        if step % args.export_every == 0 or step == steps:
            evaluate(step)
            export(step)
    say(t="done", iter=steps, seconds=time.time() - started)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=7000)
    p.add_argument("--sh-degree", type=int, default=1)
    p.add_argument("--max-splats", type=int, default=2_000_000)
    p.add_argument("--max-resolution", type=int, default=1920)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--export-every", type=int, default=0)
    p.add_argument("--eval-split-every", type=int, default=0)
    p.add_argument("--lr-mean", type=float, default=1.6e-4)
    p.add_argument("--growth-stop-iter", type=int, default=None)
    p.add_argument("--masked", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.export_every <= 0:
        args.export_every = args.steps
    return args


def main():
    args = parse_args()
    try:
        import torch
        import gsplat  # noqa: F401
    except ImportError as e:
        say(t="log", line=f"{e}: pip install -r host/requirements.txt")
        return 1
    if not torch.cuda.is_available():
        say(t="log", line="torch sees no CUDA device: gsplat trains on NVIDIA GPUs only")
        return 1
    say(t="log", line=f"gsplat {gsplat.__version__} on {torch.cuda.get_device_name(0)}")
    try:
        train(args)
    except torch.cuda.OutOfMemoryError:
        say(t="log", line="out of GPU memory: lower the max resolution or the maximum number of splats")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# Sets up the training host on a Linux machine with NVIDIA GPUs (docs/host.md):
# checks what the machine has, builds native Brush and creates the Python
# environment of the gsplat trainer in host/.venv, then tests each on the GPU.
#
#   pnpm host:setup              check the machine, set up both trainers and build the app
#   pnpm host:setup --brush      only native Brush (Vulkan)
#   pnpm host:setup --gsplat     only gsplat (CUDA)
#   pnpm host:setup --app        only the app the host serves (dist-web)
#   pnpm host:setup --check      only report what is there and what is missing
#   pnpm host:setup --apt        also install the missing system packages (sudo apt-get)
#
# It never installs the NVIDIA driver, and without --apt it changes nothing
# outside this folder, except for the Rust toolchains and wasm-pack that the
# Brush builds fetch into ~/.cargo and ~/.rustup: for system packages it
# prints the command to run. TORCH_INDEX_URL overrides where PyTorch comes
# from (default: the build for this machine's CUDA toolkit,
# https://download.pytorch.org/whl/cuXYZ).

set -u
cd "$(dirname "$0")/.."

want_brush=0 want_gsplat=0 want_app=0 check_only=0 use_apt=0
for arg in "$@"; do
  case "$arg" in
    --brush) want_brush=1 ;;
    --gsplat) want_gsplat=1 ;;
    --app) want_app=1 ;;
    --check) check_only=1 ;;
    --apt) use_apt=1 ;;
    *) echo "unknown option $arg (see the top of host/setup.sh)"; exit 2 ;;
  esac
done
# Nothing chosen: everything.
if [ $((want_brush + want_gsplat + want_app)) = 0 ]; then want_brush=1 want_gsplat=1 want_app=1; fi

if [ -t 1 ]; then G=$'\e[32m' R=$'\e[31m' Y=$'\e[33m' B=$'\e[1m' N=$'\e[0m'; else G= R= Y= B= N=; fi
ok()   { echo "  ${G}✓${N} $*"; }
miss() { echo "  ${R}✗${N} $*"; }
note() { echo "  ${Y}!${N} $*"; }
section() { echo; echo "${B}$*${N}"; }
has()  { command -v "$1" >/dev/null 2>&1; }

apt_missing=()
failed=()

# ---------- the machine ----------

section "Machine"
if has nvidia-smi && gpus=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null) && [ -n "$gpus" ]; then
  while IFS= read -r line; do ok "GPU $line"; done <<<"$gpus"
  ok "NVIDIA driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
  have_gpu=1
else
  miss "no NVIDIA driver (nvidia-smi doesn't answer). On Pop!_OS: sudo apt install system76-driver-nvidia, then reboot."
  have_gpu=0
fi

if has node; then
  node_major=$(node -p 'process.versions.node.split(".")[0]')
  if [ "$node_major" -ge 22 ]; then ok "Node $(node -v)"; else miss "Node $(node -v): 22.18 or newer is needed (https://nodejs.org, or nvm install 22)"; fi
else
  miss "Node isn't installed: 22.18 or newer (https://nodejs.org, or nvm install 22)"
fi
if has pnpm; then ok "pnpm $(pnpm -v)"; else miss "pnpm isn't installed: corepack enable (comes with Node), or npm install -g pnpm"; fi

if ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so\.1'; then ok "Vulkan loader"; else miss "Vulkan loader (native Brush and Chrome need it)"; apt_missing+=(libvulkan1); fi
if has vulkaninfo; then ok "vulkaninfo"; else note "vulkaninfo missing: only used to verify GPU selection for Chrome"; apt_missing+=(vulkan-tools); fi
if ls /usr/share/vulkan/implicit_layer.d/*MESA_device_select* >/dev/null 2>&1; then
  ok "Mesa device-select layer (one headless Chrome per GPU)"
else
  note "Mesa device-select layer missing: only needed to pin headless Chrome to a GPU"
  apt_missing+=(mesa-vulkan-drivers)
fi
chrome=$(command -v google-chrome || command -v google-chrome-stable || command -v chromium || command -v chromium-browser || true)
if [ -n "$chrome" ]; then ok "Chrome at $chrome"; else note "no Chrome or Chromium: only needed for the headless batches (https://www.google.com/chrome)"; fi

if has cc || has gcc; then ok "C compiler"; else miss "no C/C++ compiler (both trainers' builds need one)"; apt_missing+=(build-essential); fi

# ---------- what each trainer needs ----------

section "Native Brush (Vulkan)"
brush_ready=1
if has cargo && has rustup; then ok "Rust $(rustc --version 2>/dev/null | cut -d' ' -f2)"; else
  miss "Rust isn't installed: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  brush_ready=0
fi
if has git; then :; else miss "git isn't installed"; apt_missing+=(git); brush_ready=0; fi
if [ -x host/bin/brush ]; then ok "host/bin/brush is built"; else note "host/bin/brush not built yet"; fi

section "gsplat (CUDA)"
gsplat_ready=1
if has python3 && python3 -c 'import venv, ensurepip' 2>/dev/null; then ok "Python $(python3 -c 'import platform; print(platform.python_version())') with venv"; else
  miss "python3 with the venv module"
  apt_missing+=(python3-venv python3-pip)
  gsplat_ready=0
fi
# gsplat compiles its CUDA kernels on the machine the first time it runs:
# that takes the CUDA toolkit (nvcc), not just the driver.
nvcc_bin=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc 2>/dev/null || true)
if [ -n "$nvcc_bin" ]; then
  cuda_ver=$("$nvcc_bin" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -n1)
  ok "CUDA toolkit $cuda_ver ($nvcc_bin)"
else
  miss "CUDA toolkit (nvcc): gsplat compiles its CUDA code on first use, the driver alone isn't enough"
  apt_missing+=(nvidia-cuda-toolkit)
  cuda_ver=
  gsplat_ready=0
fi
if has g++; then :; else miss "g++ (compiles gsplat's CUDA code)"; apt_missing+=(build-essential); gsplat_ready=0; fi
if [ -x host/.venv/bin/python ]; then ok "host/.venv exists"; else note "host/.venv not created yet"; fi

# ---------- system packages ----------

if [ ${#apt_missing[@]} -gt 0 ]; then
  pkgs=$(printf '%s\n' "${apt_missing[@]}" | sort -u | tr '\n' ' ')
  section "System packages"
  if [ $use_apt = 1 ] && [ $check_only = 0 ]; then
    echo "  sudo apt-get install -y $pkgs"
    if sudo apt-get update -qq && sudo apt-get install -y $pkgs; then
      ok "installed; run pnpm host:setup again to continue with them in place"
      exit 0
    fi
    miss "apt-get failed"
    exit 1
  fi
  echo "  Missing: $pkgs"
  echo "  Install them with:  sudo apt install $pkgs"
  echo "  (or run this again with --apt). nvidia-cuda-toolkit is Ubuntu's CUDA toolkit;"
  echo "  NVIDIA's own repository has newer ones: https://developer.nvidia.com/cuda-downloads"
fi

if [ $check_only = 1 ]; then
  echo
  exit 0
fi

# ---------- app dependencies ----------

if has pnpm && has node; then
  section "App dependencies"
  if pnpm install --frozen-lockfile >/dev/null 2>&1; then ok "pnpm install"; else miss "pnpm install failed (run it by hand to see why)"; failed+=("pnpm install"); fi
fi

# ---------- native Brush ----------

if [ $want_brush = 1 ]; then
  section "Building native Brush"
  if [ $brush_ready = 0 ]; then
    miss "skipped: see what is missing above"
    failed+=("native Brush")
  else
    # Rebuilt when the pinned commit or a patch changes.
    stamp=$(cat scripts/build-brush-web.mjs web/brush/*.patch host/brush/*.patch | sha256sum | cut -d' ' -f1)
    if [ -x host/bin/brush ] && [ "$(cat host/bin/brush.rev 2>/dev/null)" = "$stamp" ]; then
      ok "already built from this commit and patches"
    elif node scripts/build-brush-web.mjs --native; then
      echo "$stamp" > host/bin/brush.rev
      ok "host/bin/brush"
    else
      miss "the build failed (output above)"
      failed+=("native Brush")
    fi
  fi
fi

# ---------- the app ----------

# The page with the "Train on" setting, which the host serves beside its API.
if [ $want_app = 1 ]; then
  section "Building the app"
  if [ $brush_ready = 0 ] || ! has pnpm; then
    miss "skipped: see what is missing above"
    failed+=("the app")
  else
    # Brush as WebAssembly: rebuilt when the pinned commit or a patch changes.
    stamp=$(cat scripts/build-brush-web.mjs web/brush/*.patch | sha256sum | cut -d' ' -f1)
    wasm_ok=1
    if [ -f web/pkg/brush_js.js ] && [ "$(cat web/pkg/.rev 2>/dev/null)" = "$stamp" ]; then
      ok "Brush for the browser already built from this commit and patches"
    else
      has wasm-pack || cargo install wasm-pack --locked || wasm_ok=0
      if [ $wasm_ok = 1 ] && node scripts/build-brush-web.mjs; then
        echo "$stamp" > web/pkg/.rev
        ok "web/pkg (Brush for the browser)"
      else
        wasm_ok=0
      fi
    fi
    if [ $wasm_ok = 1 ] && pnpm web:build >/dev/null 2>&1; then
      ok "dist-web: pnpm host serves it at the host's address"
    else
      miss "the build failed (pnpm web:wasm and pnpm web:build show why)"
      failed+=("the app")
    fi
  fi
fi

# ---------- gsplat ----------

if [ $want_gsplat = 1 ]; then
  section "Setting up gsplat"
  if [ $gsplat_ready = 0 ]; then
    miss "skipped: see what is missing above"
    failed+=("gsplat")
  else
    py=host/.venv/bin/python
    [ -x $py ] || python3 -m venv host/.venv || { miss "couldn't create host/.venv"; failed+=("gsplat"); py=; }
    if [ -n "$py" ]; then
      $py -m pip install -q --upgrade pip
      # PyTorch built for this machine's CUDA toolkit, so the code gsplat
      # compiles with nvcc matches the CUDA libraries PyTorch brings.
      index=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu${cuda_ver/./}}
      echo "  PyTorch from $index"
      if ! $py -m pip install -q torch --index-url "$index"; then
        note "no PyTorch build there for CUDA $cuda_ver; taking PyPI's default build instead"
        $py -m pip install -q torch || failed+=("gsplat")
      fi
      $py -m pip install -q -r host/requirements.txt || failed+=("gsplat")
      torch_cuda=$($py -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null)
      if [ -n "$torch_cuda" ] && [ "${torch_cuda%%.*}" != "${cuda_ver%%.*}" ]; then
        note "PyTorch is built for CUDA $torch_cuda and the toolkit is $cuda_ver: if the test below fails,"
        note "install a matching pair (TORCH_INDEX_URL=https://download.pytorch.org/whl/cuXYZ pnpm host:setup --gsplat)"
      fi
      echo "  Compiling gsplat's CUDA code and rendering a test image (a few minutes the first time)…"
      if out=$($py host/train_gsplat.py --self-test 2>&1); then
        ok "$(echo "$out" | tail -n1)"
      else
        miss "gsplat doesn't work on this machine yet:"
        echo "$out" | tail -n 12 | sed 's/^/      /'
        failed+=("gsplat")
      fi
    fi
  fi
fi

# ---------- result ----------

section "Training host"
node host/server.mjs --check 2>&1 | sed '1d'
echo
if [ ${#failed[@]} -gt 0 ]; then
  echo "${R}Not ready:${N} $(printf '%s\n' "${failed[@]}" | sort -u | paste -sd, - | sed 's/,/, /g')."
  [ $have_gpu = 0 ] && echo "This machine has no working NVIDIA driver, which both trainers need."
  exit 1
fi
echo "${G}Ready.${N} Start it with:  pnpm host"

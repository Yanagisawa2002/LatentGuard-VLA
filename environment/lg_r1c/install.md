# LG-R1c remote environment

LG-R1c runs only on the authorized remote GPU host. It reuses the already
frozen LeRobot 0.6.0 runtime through an isolated virtual-environment overlay;
the LG-R0/LG-R1 environment is never upgraded in place.

Every remote shell must first run:

```bash
source /etc/network_turbo
```

Python packages use the Aliyun PyPI mirror by default. Large public model
weights prefer Alibaba ModelScope when a byte-identical official mirror is
available. Every mirrored shard must still match the exact size and SHA-256
frozen from the canonical model revision. Exact Hugging Face revision metadata
and processor/tokenizer/configuration files remain content-bound because the
revision and LFS identities are part of the evidence contract. If no validated
mirror exists, the exact canonical snapshot is used instead.

The execution environment is created outside the checkout:

```bash
python -m venv --system-site-packages "$LG_R1C_ENV_ROOT/.venv-lg-r1c"
base_site="$("$LATENTGUARD_REMOTE_PYTHON" -c \
  'import site; print(site.getsitepackages()[0])')"
overlay_site="$("$LG_R1C_ENV_ROOT/.venv-lg-r1c/bin/python" -c \
  'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$base_site" > \
  "$overlay_site/latentguard_lg_r1c_frozen_base.pth"
"$LG_R1C_ENV_ROOT/.venv-lg-r1c/bin/python" -m pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  --no-build-isolation --no-deps -e "$LATENTGUARD_REMOTE_REPO"
```

The `.pth` exposes the already frozen base environment read-only because one
virtual environment does not inherit another virtual environment's
site-packages through `--system-site-packages`. It does not install into or
upgrade the base environment.

The overlay must report LeRobot `0.6.0` and the integration commit
`30da8e687a6dfc617fcd94afc367ac7071c376ce`. Model snapshots are stored
outside Git. Loading fails closed on a missing revision, missing
processor/tokenizer, unexpected checkpoint key, or LFS SHA-256 mismatch.

The completed environment used Python 3.12.3, PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128, Transformers 5.5.4, Hugging Face Hub 1.22.0,
LeRobot 0.6.0, and one NVIDIA GeForce RTX 5090. The older LG-R0/LG-R1
environment was not upgraded.

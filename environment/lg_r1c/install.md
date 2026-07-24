# LG-R1c remote environment

LG-R1c runs only on the authorized remote GPU host. It reuses the already
frozen LeRobot 0.6.0 runtime through an isolated virtual-environment overlay;
the LG-R0/LG-R1 environment is never upgraded in place.

Every remote shell must first run:

```bash
source /etc/network_turbo
```

Python packages use the Aliyun PyPI mirror by default. Hugging Face model
snapshots are the exception: they are downloaded from their canonical model
repositories at the exact commit recorded in
`configs/lg_r1c/reward_stack.yaml`, because the Hub revision and LFS hashes
are part of the evidence contract.

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

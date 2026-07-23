# LG-R0 isolated environment

LG-R0 uses a dedicated Linux environment and does not upgrade the LatentGuard
development environment.

## Frozen source and resolver

- Python: 3.12 (required by LeRobot 0.6.0)
- LeRobot source: tag `v0.6.0`, commit
  `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- Resolver: `uv 0.11.31`
- Lock source: the exact upstream `uv.lock` at that commit
- Enabled extras: `vla_jepa`, `libero`

The checked-in `requirements-lock.txt` was generated from the exact upstream
lock with:

```bash
uv export --frozen --no-dev --no-emit-project \
  --extra vla_jepa --extra libero \
  --output-file requirements-lock.txt
```

Two attempted direct hash installs were rejected rather than weakened:

1. default multi-index resolution selected an incompatible source for
   `certifi`;
2. `--index-strategy unsafe-best-match` then exposed an upstream
   `torchcodec` hash/export incompatibility.

The installed environment was therefore synchronized from the exact upstream
lock:

```bash
LG_R0_ENV=/absolute/runtime/path/latentguard-lg-r0
UV_PROJECT_ENVIRONMENT="$LG_R0_ENV" \
  uv sync --frozen --no-dev --extra vla_jepa --extra libero
```

The remote runtime uses the ordinary PyPI LeRobot 0.6.0 wheel. Its archive was
downloaded once, inspected to contain the VLA-JEPA implementation, and
content-bound before installation:

```text
lerobot-0.6.0-py3-none-any.whl
sha256=b38a564fbc441d98380576863bf68635dde5fc2c42ddc2a39d0486640dc9e9a8
```

`environment-manifest.json` records every installed distribution. The runtime
validator additionally rejects an editable LeRobot install, missing local
snapshots, unavailable CUDA, and revision/version drift.

The server could not clone GitHub reliably, so the installed wheel itself
does not provide a cryptographic VCS binding to
`30da8e687a6dfc617fcd94afc367ac7071c376ce`. That commit remains the reviewed
source identity, while the exact wheel hash above is the remote runtime
identity. This limitation is preserved rather than describing the wheel as a
verified source checkout.

The separately downloaded PyTorch wheel is:

```text
torch-2.11.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl
sha256=d252cf975fb18c94a85336323ad425f473df56dab35a44b00399bd70c7a3b997
```

LeRobotDataset video reload also requires a complete FFmpeg shared-library
runtime. The accepted remote run used Ubuntu's FFmpeg
`4.4.2-0ubuntu0.22.04.1` and TorchCodec `0.11.1+cpu`; a real 77-frame MP4 was
decoded before the four-dataset reload gate was repeated.

## Snapshot binding

Set `LG_R0_MODEL_ROOT` only when the three exact local snapshot directories are
stored outside the default cache root. The expected directory names are:

```text
vla-jepa-libero-735d9f6
qwen3-vl-2b-8964489
vjepa2-vitl-b3c1679
```

Every download uses an exact full revision. Validation and inference run with
offline/local-only loading. The policy config is overridden in memory to point
the nested Qwen and V-JEPA identifiers at these content-validated local
snapshots; official checkpoint files are not edited.

LIBERO's separately downloaded simulator assets are also frozen:
`lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.
`LG_R0_LIBERO_ASSET_ROOT` points to that pinned local snapshot. The runtime
sets hf-libero's in-process asset cache to this explicit path before creating
an environment; it does not invoke the upstream unpinned fallback downloader.

The exact snapshots can be prepared outside the Git checkout with:

```bash
python scripts/lg_r0_prepare_external_assets.py \
  --manifest artifacts/lg_r0/base_model_manifest.json \
  --model-root "$LG_R0_MODEL_ROOT" \
  --libero-asset-root "$LG_R0_LIBERO_ASSET_ROOT" \
  --output "$LG_R0_OUTPUT_ROOT/external-assets.json"
```

## Validation command

```bash
python scripts/lg_r0_validate_environment.py \
  --output artifacts/lg_r0/environment_validation.json \
  --manifest environment/lg_r0/environment-manifest.json
```

GPU inference and LIBERO execution run only on the user-authorized remote
server from an exact pushed LatentGuard commit. The server is not used for
tracked source edits or training. Local validation remains CPU-only.

## Remote download bootstrap

On the authorized AutoDL image, load the network accelerator once after every
server boot and before any network download:

```bash
source /etc/network_turbo
```

Future package downloads default to an Aliyun mirror. Exact revisions and
hashes remain mandatory, and the resolved endpoint is recorded in the run
audit. An alternate endpoint may be used only when the exact artifact is
unavailable from Aliyun; that exception must be recorded. The completed LG-R0
run predates this default: model snapshots used `hf-mirror.com`, while the
system FFmpeg packages came from the server's configured Huawei Cloud Ubuntu
mirror.

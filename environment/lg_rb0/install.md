# LG-RB0 remote environment

This environment is execution-only. Tracked LatentGuard files are created,
reviewed, committed, and pushed locally before the remote checkout is
synchronized to that exact revision.

## Frozen stack

- Python 3.11
- RoboLab `v0.2.1`, peeled commit
  `0aef241fb088ca21bb4ebd24448940ed56620d17`
- Isaac Sim `5.1.0`
- Isaac Lab `2.3.2.post1`
- one headless CUDA environment

RoboLab's official replay guide states that faithful replay requires the same
software stack and `num_envs=1`. Its official state validator uses absolute
tolerance `0.01`; LG-RB0 records that gate without relaxing it. The stricter
`1e-6` tolerance is a separate LatentGuard takeover-repeatability gate and does
not replace the official validator.

## Download and storage policy

At every server boot, run `source /etc/network_turbo`. Put the environment,
temporary files, package caches, RoboLab checkout, assets, recordings, videos,
and complete run outputs under `/root/autodl-tmp/latentguard-lg-rb0`. Use the
Aliyun PyPI mirror by default. NVIDIA's package index and PyTorch's official
CUDA index are permitted only where the pinned RoboLab stack requires packages
that the mirror does not provide.

Do not install into the system Python or the tracked LatentGuard checkout.
Do not commit external source, assets, recordings, simulator states, caches,
videos, or checkpoints.

## Gate order

1. Verify the clean remote LatentGuard checkout at the expected pushed SHA.
2. Verify the RoboLab tag peels to the frozen commit.
3. Create the isolated Python 3.11 environment on the data disk.
4. Install and import the pinned stack.
5. Record the exact package, driver, GPU, disk, and source identities.
6. Launch one headless environment and run a one-step smoke test.
7. Only then generate the preregistered mechanics recordings and execute
   faithful, prefix, branch, and isolation validation.

Any earlier failure stops the later gates. The server remains on and SSH-ready
after completion unless the user explicitly requests shutdown.

## Observed installation result

The validated execution environment used Python `3.11.15`, RoboLab `0.2.1`,
Isaac Sim distribution `5.1.0.0`, Isaac Lab `2.3.2.post1`, PyTorch
`2.7.0+cu128`, and CUDA runtime `12.8`. Isaac Sim's four-component package
version is PEP 440-equivalent to the preregistered `5.1.0` requirement.

The Aliyun mirror remained the default. It did not provide `uv` or the pinned
`gymnasium==1.2.0`, so those packages used the public Python index; the pinned
NVIDIA and PyTorch CUDA indexes were also required by the official dependency
set. RoboLab assets were fetched with Git LFS after the network accelerator was
sourced. Runtime installation and assets remained under the data disk.

Both registered one-step GPU task smokes passed with a single standard NVIDIA
Vulkan ICD. The available RTX 5090 exposed `34190917632` bytes of GPU memory,
below RoboLab's recommended 48 GiB, but no OOM occurred. These launch results
validate the environment only; they do not imply faithful replay or policy
success.

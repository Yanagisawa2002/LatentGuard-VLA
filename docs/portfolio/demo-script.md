# Portfolio demo script (78 seconds)

The release demo is deterministic motion graphics generated only from committed,
audited result values. It does not run a simulator, load a checkpoint, or present
synthetic animation as robot footage.

## Timeline

| Time | Beat | On-screen evidence |
| --- | --- | --- |
| 0:00-0:09 | Question | Can a verifier stop a plausible robot action from failing? Exact PickCube scope and no general-safety claim. |
| 0:09-0:21 | Exact replay | Content-bound state, independent baseline/candidate sessions, 60 trajectories, 360 anchors, and 2,880 strong outcomes. |
| 0:21-0:33 | Verifier | Untouched-test failure AUPRC 0.8986; risk rises from 3.38% at 80% coverage to 17.9% at full coverage. |
| 0:33-0:45 | One-shot | Temporal 98.33% versus random 88.89%; visual 98.89%; frozen blind candidate pools. |
| 0:45-0:58 | Closed-loop failure | Distilled visual 73.33% versus fixed primary 100%, with 93.48% intervention. |
| 0:58-1:12 | Conservative redesign | Clean 100%/0% intervention; fault success 80%; fault intervention 1.17%; override recall only 4.03%. |
| 1:12-1:18 | Takeaway | Preserve the positive, negative, and partial result in one reproducible release. |

## Build

The MP4 is a GitHub Release asset, not a tracked repository file. Install the
optional rendering tools into an isolated environment and run:

```bash
python -m pip install Pillow imageio-ffmpeg
python scripts/build_portfolio_demo.py
```

This writes:

- `artifacts/latentguard-vla-78s-demo.mp4`
- `artifacts/latentguard-vla-demo-poster.png`

Both paths are ignored by Git. The tracked
[`assets/latentguard-demo-poster.svg`](assets/latentguard-demo-poster.svg) is the
accessible README poster and links to the stable release asset.

## Claim and asset boundary

- Source values come from `docs/release/results.json` and
  `docs/portfolio/key-results.md`.
- The demo must not imply cross-task, cross-robot, real-robot, VLA, or general
  safety validation.
- Exact replay is described as physically validated only for the pinned
  ManiSkill PickCube adapter and its authorized state-restoration contract.
- The closed-loop regression and low fault override recall remain visible; they
  are not replaced by the stronger one-shot result.
- Raw replay videos, RGB datasets, simulator states, checkpoints, and caches
  remain outside Git.

# LG-RB0.1 RoboLab replay compatibility plan

## Objective

LG-RB0.1 makes one bounded compatibility correction against RoboLab `v0.2.1`
at commit `0aef241fb088ca21bb4ebd24448940ed56620d17`. Recorded configuration may
restore declarative values, while executable code, callables, and
runtime-owned metadata remain supplied by the live source tree.

The milestone reuses the ten content-bound LG-RB0 recordings. It does not
record new episodes, alter task or physics code, integrate a policy, generate
candidates, train a model, or access final seeds.

## Observed blockers

LG-RB0 recorded every subtask predicate as a lossy `functools.partial(...)`
string. The old overlay replaced live lists containing callable objects with
those JSON lists. The first subtask update then attempted to call a string.

The strict state tree also observed two live-only empty mappings:
`/deformable_object` and `/gripper`. Their numeric sibling leaves matched the
recording exactly. LG-RB0.1 may canonicalize only these exact namespaces and
only while they are strictly empty.

## Ordered gates

1. Bind the exact upstream commit, real recording digests, live config types,
   old overlay branch, call site, instruction metadata, and empty mappings.
2. Demonstrate the old failure with CPU-safe overlay regression cases.
3. Apply the exact content-bound patch to a clean external checkout.
4. Pass patched regression tests, upstream tests, patch provenance checks, and
   local LatentGuard tests.
5. Push the LatentGuard revision, synchronize the remote checkout, and verify
   the exact revision before simulator execution.
6. Run 30 faithful replays: two tasks, ten fixed recordings, three repeats.
7. Only after a complete faithful pass, run early/middle/late prefix replay,
   same-suffix checkpoints at 1/5/10 steps, and A-B-A/B-A-B isolation.
8. Classify the result using the frozen A/B/C/D stop rules.

## Compatibility design

The patch preserves a live callable target when the recorded import path
resolves to the same identity. Unimportable or different identities fail
closed. A live list or tuple containing callable code is preserved as one
runtime unit because JSON cannot represent `functools.partial` safely.
Recorded-only keys are accepted only in explicit declarative parameter or
renderer-settings namespaces. Missing code keys and unknown schema additions
are skipped with fatal classifications.

`/_instruction_variants` remains runtime-owned and is never restored from
lossy JSON. The recorded resolved `/instruction` remains authoritative for the
replay instruction.

## State and visual comparison

Official RoboLab state validation remains at `0.01` and is reported
separately. Every LatentGuard strict numeric comparison uses `1e-6`. Exact
allowlisted empty-map canonicalization happens symmetrically and records
before/after paths and digests. Unknown empty mappings and all non-empty
mappings fail.

Observation comparison remains byte-exact through content hashes; pixel
tolerance is zero.

## Stop rules

- Result A requires 30/30 faithful completion, zero fatal or unexpected
  overlay skips, zero strict state/terminal/success mismatches, complete prefix
  and branch reproducibility, and passing isolation.
- Result B stops the RoboLab counterfactual route if faithful replay passes but
  any prefix or branch gate fails.
- Result C stops the route if the one compatibility patch still cannot
  complete faithful replay.
- Result D stops before execution if a safe patch would require task changes,
  `eval`, `exec`, arbitrary string execution, or hard-coded task repair.

No second compatibility-patch stage is authorized.

## Remote cost and ETA

The source audit and patch tests should take about 10–20 minutes of mostly
CPU-bound remote time. The faithful gate should take about 25–45 GPU minutes
on the existing RTX 5090. Prefix and branch gates run only after faithful
promotion and are expected to require another 60–120 GPU minutes. Total
wall-clock ETA, including environment verification and artifact review, is
approximately 3–5 hours if every gate passes; a failed faithful gate stops
earlier.

The server remains powered on and SSH-ready after completion unless the user
explicitly authorizes shutdown.

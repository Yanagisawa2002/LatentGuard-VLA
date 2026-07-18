# Receding-horizon visual action shield

M4C evaluates frozen failure verifiers inside a repeated PickCube selection
loop. It is an evaluation protocol, not a newly trained controller. A successful
M4C result applies only to this fixed official-planner, candidate-pool,
simulator, camera, and domain contract.

## Boundary transaction

At a committed physical boundary the runtime captures the complete simulator
state, the 38-value verifier state, and (for visual selectors) three RGB views
without advancing physics. It verifies the post-render full state using the
adapter-bound `tolerance_verified_full_state_v1` comparison and verifies the
task projection and verifier bytes exactly.

The candidate factory then derives the same eight horizon-16 candidates from
the fixed nominal source plan. It rejects clipping, repair, an exact source
candidate, a changed source suffix, or any selector-specific proposal pool. The
selector sees only its allowlisted current-boundary inputs. The pool and final
ranking are written before execution. Four actions are executed unless a task
or safety terminal occurs earlier, then the next state is committed.

The immutable chronology is:

```text
restore/observe -> build pool -> persist decision -> execute <= 4 actions
-> evaluate each action -> commit next boundary or terminal episode
```

On recovery, finalized decisions are loaded and verified by content digest. A
partially executed stride is discarded by restoring the previous committed
boundary; the exact selected candidate is reused and no verifier is called.
Already terminal episodes execute zero simulator, renderer, or model work.

## Fixed visual inference

The camera order and rendering domains are inherited from accepted M4A/M4B
artifacts. One boundary supplies exactly three `uint8` RGB224 images. Slots
0, 1, and 2 of a fixed 128-image backbone batch contain those images; the
remaining slots repeat them round-robin. Only the first three feature rows are
consumed. Variable batch sizes and privileged state inputs are rejected for the
visual selectors.

## Selector matrix and outcomes

Every source runs four nonvisual selectors once and the two visual selectors in
three domains, giving ten paired episodes per source. Fixed primary and
deterministic random are non-learned baselines; action-only and privileged
structured use their accepted five-seed M3B ensembles; direct and distilled
visual use their accepted three-seed M4B ensembles.

Success, task failure, unsafe termination, horizon exhaustion, and execution
error remain separate. Abstention is not synthesized, runtime errors are not
task failures, and horizon exhaustion is not silently converted into failure.
Metrics use source-trajectory pairing and the frozen 2,000-resample bootstrap.
Wall time is stored in a runtime-only sidecar and does not affect any identity.

## Limitations

The nominal plan is fixed and finite. M4C selects among deterministic
perturbations of its current window; it does not generate a new plan, learn a
policy online, use future state, or establish visual-language, cross-task,
general-safety, or real-robot performance.

## M4C full result

The accepted full run completed 60 new sources and all 600 selector/domain
episodes with zero execution errors. The primary distilled visual ensemble
improved canonical success over deterministic random (`0.7333` versus
`0.4500`) but underperformed fixed primary (`1.0000`), action-only (`0.9333`),
direct visual (`0.9000`), and privileged structured (`0.9167`). The overall
predeclared targets therefore failed. Strong-camera success matched canonical
at `0.7333`; strong-lighting success was `0.8667`. These results are preserved
without post-outcome tuning in the compact
[`reports/m4c`](../reports/m4c/20260718T204934Z_m4c-full_2864b30_seed420000/review.md)
review.

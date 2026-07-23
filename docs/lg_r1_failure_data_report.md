# LG-R1 natural-failure and stagnation report

## Result B failure-data verdict

The standard frozen protocol produced 159 successes and one natural failure
across 160 episodes. This is enough to evaluate the progress baseline, but not
enough to train or validate a failure head.

| Suite/task | Episodes | Success | Natural failure | Mean frames |
| --- | ---: | ---: | ---: | ---: |
| `libero_10` task 1 | 20 | 20 | 0 | 230.10 |
| `libero_10` task 3 | 20 | 20 | 0 | 237.75 |
| `libero_goal` task 3 | 20 | 19 | 1 | 184.50 |
| `libero_goal` task 7 | 20 | 20 | 0 | 77.00 |
| `libero_object` task 1 | 20 | 20 | 0 | 124.50 |
| `libero_object` task 7 | 20 | 20 | 0 | 129.60 |
| `libero_spatial` task 1 | 20 | 20 | 0 | 105.95 |
| `libero_spatial` task 4 | 20 | 20 | 0 | 129.25 |

Seven tasks were 20/20. The two nominally long-horizon `libero_10` tasks were
also 20/20, so their longer horizons did not create a useful natural-failure
population under this policy and seed schedule. `libero_goal` task 3 was
19/20 and supplied the only natural failure.

## The one natural failure

`libero_goal-task3-seed2017` was a 300-frame horizon exhaustion for “open the
top drawer and put the bowl inside.” The fixture opened, object contact and a
short transport phase occurred, then the trajectory regressed and finished
with the placement goal false. Timeout was not relabeled as a synthetic
terminal-failure stage.

The SARM-style mean prediction for this episode rose from near zero to about
`0.53` in the middle, fell, and plateaued near `0.30`; the final supervised
progress was `0.275`. The mean pre-terminal delta was `0.000404`. In the
bounded feature sequence, the terminal plateau averaged 51 sampled points and
a regression preceded timeout by 67 sampled points. These are sampled-feature
counts, not control-step lead times.

Test temporal recall was `0.8980` for stagnation but only `0.25` for
regression/negative progress. The curve is consistent with a detectable
plateau and regression in this one episode; it does not establish sensitivity,
specificity, failure prediction, or safety value.

## Success/failure curve comparison

The 159 successful episodes show a population mean progress curve increasing
from roughly `0.057` in the first normalized bin to `0.694` in the last bin.
The single failed episode rises, regresses, and ends on a low plateau. Because
the failure curve has `n=1`, it is a case study only. It must not be presented
as a statistically supported difference between successful and failed
episodes.

## LG-R2 gate

| Requirement | Observed | Pass |
| --- | ---: | --- |
| valid episodes ≥ 100 | 160 | yes |
| tasks ≥ 4 | 8 | yes |
| suites ≥ 2 | 4 | yes |
| natural failed episodes ≥ 10 | 1 | **no** |
| failure windows ≥ 100 | 118 | yes, but from one episode |
| successful windows ≥ 100 | 8,760 | yes |
| stage annotation | pass | yes |
| SARM evaluation | complete | yes |
| episode/seed leakage | 0 | yes |
| processor/checkpoint identity | 100% | yes |

The formal status is `FAILURE_DATA_GATE_NOT_MET` and
`LG_R2_AUTHORIZED=false`. Passing the failure-window count cannot compensate
for those windows coming from only one episode.

## Evidence-grounded expansion recommendation

No expansion is authorized by LG-R1 itself. If the user later creates an
LG-R1b data-expansion milestone, it should freeze new seeds and untouched
standard tasks before observing their outcomes. Structure-based candidates
from the installed official LIBERO registry are:

- `libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`;
- `libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`;
- `libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`;
- `libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`.

These are proposed because they add sequential articulation or distinct
multi-object goals, not because their outcomes were inspected. The exact
task/seed registry and sample size would require a separate local plan,
commit, push, and explicit authorization. The existing eight tasks should not
be selectively rerun on observed “bad” seeds, final seeds must remain sealed,
and synthetic action corruption must not be used to satisfy the gate.

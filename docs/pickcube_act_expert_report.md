# PickCube native expert report

Status: **authorized for demonstration collection**.

The implementation reuses the installed ManiSkill PickCube Panda
motion-planning solution, wrapped by project-owned phase and action auditing.
The accepted report is retained outside Git at
`artifacts/pickcube_act/expert_evaluation.json` and was produced by remote run
`20260721T094113Z_wm-v0-p0-expert-rrt-fallback_1e633f6_seed600000` from exact
source commit `1e633f6477c32cf1fe75cfbd6579abc1e594f733`.

The frozen gate is 100 independent seeds starting at 600000, at least 95%
success, zero simulator errors, zero non-finite actions, zero out-of-bounds
actions, zero workspace violations, complete failure taxonomy, and observed
execution in every declared expert phase.

An initial diagnostic run reached final task success on 100/100 seeds but used
67--93 actions and therefore crossed the native 50-step TimeLimit. That result
is retained as invalid timeout-semantics evidence and does not authorize data
collection. The accepted gate stops on the first native termination/truncation,
uses the official planner at fixed velocity/acceleration scales 3.0, repeats
only the final transport target for real controller settling, and invokes
official RRTConnect only if screw planning fails before a transport action
executes.

The accepted result is exactly 95/100 successes (95%). All four declared
phases executed, with zero simulator errors, non-finite actions,
out-of-contract actions, and workspace violations. The five retained failures
are one `expert_phase_incomplete` and four native `timeout` outcomes. Earlier
2.0, 3.0-without-settling, and 3.0-settling-without-RRT candidates remain
negative diagnostic evidence and are not used as the authorization result.

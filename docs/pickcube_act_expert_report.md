# PickCube native expert report

Status: **pending remote 100-seed validation**.

The implementation reuses the installed ManiSkill PickCube Panda
motion-planning solution, wrapped by project-owned phase and action auditing.
No demonstration collection or ACT training is authorized until
`artifacts/pickcube_act/expert_evaluation.json` records all fixed gates passing.

The frozen gate is 100 independent seeds starting at 600000, at least 95%
success, zero simulator errors, zero non-finite actions, zero out-of-bounds
actions, zero workspace violations, complete failure taxonomy, and observed
execution in every declared expert phase.

An initial diagnostic run reached final task success on 100/100 seeds but used
67--93 actions and therefore crossed the native 50-step TimeLimit. That result
is retained as invalid timeout-semantics evidence and does not authorize data
collection. The corrected gate stops on the first termination/truncation and
pins the official planner velocity/acceleration scales to 2.0 before rerunning
the same 100 seeds.

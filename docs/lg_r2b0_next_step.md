# LG-R2b0 next-step decision

Status: **Result C; LG-R2b1 is not authorized**.

- Result A: permit design review for LG-R2b1; do not start it automatically.
- Result B: preserve the real-candidate evidence, do not train a ranker, and
  review whether another separately authorized real source or task/stage design
  is justified.
- Result C: stop because native multi-candidate sampling or complete state
  restoration is not viable; do not substitute synthetic corruption.

The real native multi-candidate source exists, so the stop is not caused by
candidate collapse. The blocker is the inability to replay a complete
same-state continuation deterministically under the frozen LIBERO/MuJoCo
contract. Adding render-derived arrays made the anchor image nearly exact but
caused the next physical step to diverge, demonstrating that copying derived
state is not a valid substitute for a canonical simulator restore boundary.

Do not:

- train LG-R2b1 or any ranker;
- reinterpret candidate-source diversity as outcome identifiability;
- relax the registered numeric or pixel tolerances after seeing the failures;
- replace real candidates with synthetic perturbations;
- access sealed final seeds or tune on terminal outcomes.

A future route requires separate authorization and a new milestone. The
smallest technically defensible prerequisite would be an upstream-supported,
version-bound simulator serialization/restoration mechanism whose fresh-session
continuations pass the same 10-state by 5-repeat contract without restoring
derived transforms manually. Repeating this pilot on more anchors or with a
larger model is not justified.

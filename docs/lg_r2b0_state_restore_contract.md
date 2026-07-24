# LG-R2b0 state restore contract

The normative contract is
[the pre-registered complete-state design](plans/lg-r2b0-state-restore-contract.md).

The implementation binds complete exposed simulator arrays, controller state,
runtime latches, RNG state, and the MuJoCo body/geometry/camera/site transforms
actually consumed by rendering. Numeric comparison uses the frozen `1e-6`
tolerance, while structure and runtime metadata are exact. The compatibility
gate initially permits zero rendered-array error and records maximum, mean, and
differing-value statistics; every render is followed by another complete-state
comparison. Task predicates, termination, and success are exact.

The untouched 10-state by 5-repeat gate **failed** at execution commit
`498a3153dfe7f1023510bf07e7f07ae415840280`. It recorded 36 repeated-trace
failures, one immediate render mismatch, three terminal-result mismatches, and
zero predicate mismatches. The maximum repeated-trace numeric error was
`0.467782997760434916`; the maximum pixel absolute error was 194, maximum mean
absolute pixel error was `1.4887030984268708`, and maximum different-value
ratio was `0.018006616709183673`.

Immediate restoration itself was exact for the completed comparisons, and
post-render complete-state comparisons had zero mismatches. The decisive
failure appeared after applying the same next action: 36 of 36 recorded
cross-repeat trace comparisons failed, generally at rendered step 0. Two
LIBERO-goal control probes also changed between successful and unsuccessful
terminal outcomes. This is a physical continuation inconsistency, not a
reporting-only pixel issue.

The first exact-pixel compatibility attempt showed that the step-returned
observation is not the canonical render of the archived boundary even though
all 5,700+ captured state components restored exactly. That attempt is retained
as calibration evidence. The implementation now canonicalizes the observation
directly from the captured state. A second disjoint calibration exposed that
MuJoCo's derived body, geometry, camera, and site transforms were
render-relevant but absent from the first snapshot inventory. They are now
captured, restored, and compared explicitly. A third disjoint set of ten source
seeds was the untouched validation gate. The added derived arrays made nearly
all immediate renders exact, but it did not make stepped trajectories
deterministic and introduced no basis for relaxing the frozen tolerance. Pixel
tolerances remain exactly zero. The content-bound final evidence is
`artifacts/lg_r2b0/state_restore_validation.json`; the two earlier failed
calibrations remain in the same artifact directory.

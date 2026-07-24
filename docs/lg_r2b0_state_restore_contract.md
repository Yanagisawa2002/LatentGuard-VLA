# LG-R2b0 state restore contract

The normative contract is
[the pre-registered complete-state design](plans/lg-r2b0-state-restore-contract.md).

The implementation binds complete exposed simulator arrays, controller state,
runtime latches, and RNG state. Numeric comparison uses the frozen `1e-6`
tolerance, while structure and runtime metadata are exact. The compatibility
gate initially permits zero rendered-array error and records maximum, mean, and
differing-value statistics; every render is followed by another complete-state
comparison. Task predicates, termination, and success are exact.

The completed 10-state by 5-repeat audit will cite
`artifacts/lg_r2b0/state_restore_validation.json`. Until that artifact passes,
no same-state branch may be described as reproducible or exact enough for this
pilot.

The first exact-pixel compatibility attempt showed that the step-returned
observation is not the canonical render of the archived boundary even though
all 5,700+ captured state components restored exactly. That attempt is retained
as calibration evidence. The implementation now canonicalizes the observation
directly from the captured state, proves rendering does not change complete
state, and uses ten disjoint source seeds for the untouched validation gate.
Pixel tolerances remain exactly zero.

# LG-R2b0 state restore contract

The normative contract is
[the pre-registered complete-state design](plans/lg-r2b0-state-restore-contract.md).

The implementation binds complete exposed simulator arrays, controller state,
runtime latches, and RNG state. Numeric comparison uses the frozen `1e-6`
tolerance, while structure, runtime metadata, rendered observation, task
predicates, termination, and success are exact.

The completed 10-state by 5-repeat audit will cite
`artifacts/lg_r2b0/state_restore_validation.json`. Until that artifact passes,
no same-state branch may be described as reproducible or exact enough for this
pilot.

# LG-R2b0 candidate source audit

Status before remote execution: protocol frozen; result pending.

The only admitted source is native flow-matching inference from the frozen
VLA-JEPA checkpoint and official processor/postprocessor. Candidate generation
resets the policy, seeds the recorded global Torch CPU/CUDA generators, encodes
the identical current observation and instruction, calls
`predict_action_chunk` in inference mode, and records both normalized and
native chunks.

The remote hard gate will establish fixed-seed byte reproduction, identical
preprocessed input, and at least two different content hashes across four
registered seeds. If this fails, the result is
`REAL_MULTI_CANDIDATE_SOURCE_NOT_AVAILABLE` and no branch is run. Synthetic
actions and alternate unregistered sources are not fallbacks.

The completed audit will cite
`artifacts/lg_r2b0/candidate_source_manifest.json`.

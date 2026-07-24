# LG-R2b0 candidate source audit

Status: **passed** on the frozen VLA-JEPA source.

The only admitted source is native flow-matching inference from the frozen
VLA-JEPA checkpoint and official processor/postprocessor. Candidate generation
resets the policy, seeds the recorded global Torch CPU/CUDA generators, encodes
the identical current observation and instruction, calls
`predict_action_chunk` in inference mode, and records both normalized and
native chunks.

At execution commit
`498a3153dfe7f1023510bf07e7f07ae415840280`, fixed-seed generation reproduced
byte-identically, preprocessing was deterministic, and the four registered
inference seeds produced four unique native seven-action chunks. The
policy-generated ratio was 1.0 and the synthetic ratio was 0.0. No optimizer
step or backward call occurred, and no sealed final seed was accessed.

This establishes that Source A is technically available. It does not establish
state-restoration validity, useful outcome diversity, rankability, or task
success. The content-bound evidence is
`artifacts/lg_r2b0/candidate_source_manifest.json`.

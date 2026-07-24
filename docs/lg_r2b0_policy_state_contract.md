# LG-R2b0 policy-state isolation contract

## Candidate generation

The frozen VLA-JEPA policy is reset before every candidate. This empties its
action queue. The current anchor observation and instruction are encoded again;
no earlier candidate observation, normalized action, native action, or queue
entry is reused. Torch CPU and CUDA RNG state is set from the candidate's
registered inference seed immediately before native flow sampling.

Candidate generation may advance process RNG state, but it may not alter the
simulator. Every candidate starts from the same saved observation bytes and
content-bound simulator snapshot. Input and post-generation observation/state
hashes are checked.

## Candidate execution

Candidate execution does not consult the policy action queue. The saved native
seven-action chunk is applied directly from a freshly restored anchor.
Candidate action alignment is checked byte-for-byte against the manifest.

## Continuation

After the candidate chunk, the policy is reset again, so no candidate-generation
or source-rollout action remains queued. The continuation re-encodes the actual
current observation. Every candidate at one anchor uses the same frozen
checkpoint, processor, reset rule, maximum remaining horizon, and registered
continuation seed.

Continuation randomness is therefore common by rule, not inherited from the
candidate seed. Dataset validation rejects multiple continuation identities
within an anchor.

## Failure rule

If reset does not clear the queue, fixed-seed candidate generation is not
reproducible, the current observation changes during inference, or repeated
restored execution differs, the corresponding hard gate fails. The protocol
does not repair or approximate hidden policy state.

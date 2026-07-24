# LG-R1c candidate-time limitations

ROBOMETER and TOPReward consume observed video windows. Their outputs are
therefore `EXECUTED_TRAJECTORY_REQUIRED`: they are available only after the
observations in the scored window exist. Neither model accepts a proposed
numeric action chunk or predicts the video that an unexecuted candidate would
produce.

Consequently, these rewards cannot directly rank or select unexecuted
LatentGuard candidates. In LG-R2 they may serve as post-hoc evaluators,
auxiliary supervision targets, or calibration baselines. Calling them an
online world-model score would be incorrect.

The current VLA representation is `CURRENT_STATE_ONLY`. A numeric candidate
action chunk is `CANDIDATE_ACTION_CONDITIONED`. The existing VLA-JEPA
training-style predictor is conditioned on policy-produced Qwen action-token
hidden states, not arbitrary numeric candidates; its temporal outputs are
therefore conservatively marked
`NOT_USABLE_FOR_PRE_EXECUTION_SELECTION` until a separately reviewed adapter
proves candidate conditioning. The complete typed classification appears in
`artifacts/lg_r1c/lg_r2_feature_contract.json`.

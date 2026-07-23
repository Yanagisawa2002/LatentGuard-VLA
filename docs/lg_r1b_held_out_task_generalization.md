# LG-R1b held-out-task SARM generalization

Status: frozen zero-shot protocol; remote evaluation pending.

The first result will always be the unchanged LG-R1 validation-selected SARM
checkpoint on every active unseen task with zero optimizer steps. It will be
reported beside the committed LG-R1 in-distribution test metrics. Any optional
adaptation result is secondary and cannot replace or overwrite zero-shot
evidence.

# M4B Result Review

The M4B staged visual-verifier workflow completed on one RTX 5090. Exactly four
families were screened at seed 0. Validation promoted distilled multi-view,
direct multi-view, and single-view; only those families received seeds 1 and 2.
No seeds 3 or 4 ran. The distilled ensemble remained the validation winner and
was frozen before internal test and M3C external image access.

Internal and external targets were met on the accepted fixed datasets. External
selection manifests were finalized without outcome inputs, and existing M3C
outcomes were joined afterward without simulator replay. The visual selector's
advantage over the fixed action-only selector is small and its paired intervals
include zero; this result is not a general visual-safety claim.

The fixed batch-128 raw-RGB feature path reproduced cached features and final
student probabilities exactly. A separate batch-3 probe differed from the
batch-128 cache by a maximum feature absolute error of 0.0023941993713378906.
Therefore only the fixed validated batching path is equivalence-verified; cross
batch-shape equivalence is a known limitation and no tolerance was authorized.

No renderer, simulator, VLM, LangMani, distributed process, multi-GPU run, or
new replay was launched. The server remains powered on and all environments,
caches, checkpoints, logs, and run roots remain preserved.

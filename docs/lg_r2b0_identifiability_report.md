# LG-R2b0 identifiability report

Status: **Result C — same-state counterfactual restoration is not viable under
the frozen contract**.

The pre-registered analysis is descriptive and within-anchor only. It reports
action diversity, 7/21/49-step outcome variation, local-event disagreement,
terminal composition, non-task-6 divergence, and rank ties. It trains no
predictor or ranker.

The native candidate hypothesis was supported narrowly: four requested Source
A samples were all unique and fixed-seed replay was byte-identical. The
identifiability pilot nevertheless stopped before data collection because the
same saved simulator state could not produce a deterministic stepped
continuation across five repeats.

The final untouched gate covered ten states across six task/suite pairs. It
reported:

- 36 repeated-trace failures;
- one immediate exact-render mismatch;
- zero post-render complete-state mismatches;
- three terminal-result mismatches;
- zero task-predicate mismatches.

The terminal disagreements occurred in the LIBERO-goal task-6 control probes:
identical registered continuation experiments varied between success and
non-success. Therefore any observed difference between candidate branches
would be confounded by restoration/continuation nondeterminism.

No 60-anchor data exist. Progress-spread, local-event disagreement,
mixed-terminal, non-task-6 divergence, and rank-tie metrics are unavailable,
not zero. Result A and Result B are both inapplicable. The gate artifact sets
`LG_R2B1_AUTHORIZED=false`.

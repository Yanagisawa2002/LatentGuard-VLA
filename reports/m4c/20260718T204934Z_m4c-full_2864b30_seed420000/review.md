# M4C result review

M4C completed the predeclared 60-source, 600-episode closed-loop PickCube
benchmark with no execution errors. The strict resume executed zero work for
all 600 completed identities. Every result was produced with frozen M3B/M4B
ensembles, the fixed eight-candidate pool, horizon 16, stride 4, and the
accepted fixed-batch-128 visual path. No model training, tuning, VLM, LangMani,
distributed execution, or multi-GPU execution occurred.

The primary claim is a mixed, ultimately negative result. The distilled visual
ensemble improved canonical success over deterministic random by 0.2833
(95% trajectory-bootstrap interval 0.1167 to 0.4333) and reduced unsuccessful
episodes by 51.5% relative to random. It did not improve the stronger fixed
primary baseline: distilled canonical success was 0.7333 versus 1.0000. It was
also below action-only (0.9333), direct visual (0.9000), and privileged
structured (0.9167). Therefore the predeclared overall acceptance targets were
not met, and M4C does not support a claim that the accepted distilled verifier
improves this fixed nominal proposal policy.

Domain robustness was not the failure mode. Distilled success was 0.7333 under
strong camera shift and 0.8667 under strong lighting shift, with zero execution
errors. The lighting result exceeded canonical, while camera matched canonical.
These domain comparisons do not rescue the failed fixed-primary and privileged
comparisons.

The benchmark recorded 501 successes, 98 horizon exhaustions, one narrow-proxy
unsafe termination, zero task failures, and zero execution errors. All selector
groups averaged 15.33 decisions per episode. Distilled canonical intervened on
93.48% of boundaries and averaged 14.33 non-primary choices per episode; all
60 distilled-canonical episodes intervened at least once.

State collection performed 78 fresh-session full-state verifications. The
maximum observed 70-component restoration error was
`1.1920928955078125e-07`, below the fixed `1e-6` adapter tolerance; verifier
state was byte-exact across all 38 components. Runtime recovery count was zero.
Crash-selection preservation is covered by the deterministic local recovery
tests, while the accepted remote run exercised completed zero-work resume; no
native-signal crash was injected into the completed full benchmark.

Per-episode wall time was recorded and averaged 3.88 seconds for distilled
canonical versus 2.40 seconds for fixed primary. A sampled `nvidia-smi`
observation during the benchmark was 3016 MiB; it is not a peak-memory
measurement. Dedicated model-inference latency was not independently
instrumented, so this report does not relabel episode wall time as inference
latency.

This is receding-horizon selection over a fixed nominal plan, not arbitrary
replanning, learned proposal generation, policy learning, general robot safety,
or real-robot validation.

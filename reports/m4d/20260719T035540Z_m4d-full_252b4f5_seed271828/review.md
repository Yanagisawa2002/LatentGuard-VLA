# M4D result review

M4D completed the frozen 60-source, 1,560-episode PickCube benchmark on
`0f24befbfc84a502c1af60e06bbeb8cd734a7863`. It recorded 1,393 successes and
167 horizon exhaustions, with zero task failures, unsafe outcomes, or execution
errors. No model training, calibration refit, feature-cache rebuild, new seed,
VLM, LangMani, distributed execution, multi-GPU execution, or final-set tuning
occurred.

The M4C diagnosis remained retrospective association evidence. Learned M4C
selectors intervened at 88.15% to 93.48% of decision boundaries while fixed
primary intervened at 0%; distilled visual canonical was highest at 93.48%.
Direct-versus-distilled canonical candidate-definition disagreement was 31.63%,
and action-only versus direct visual was 71.41%. These records motivate a
conservative gate, but physical trajectories may already differ at matched
decision ordinals and the accepted M4C schema did not persist ensemble
uncertainty, so the diagnosis is not causal proof.

The one-time 12-source development run completed 384/384 episodes: 360 success,
24 horizon exhaustion, and zero task failure, unsafe outcome, or execution
error. The development-only freeze selected the balanced profile for action-only,
direct visual, and distilled visual, and the conservative profile for privileged
structured. The immutable gate-selection digest is
`sha256:fd7694cdc23096d8efb6f2039b53f9176201fc2b0e0752d37d404b0ace65e577`.

The final set accepted 60 sources in 60 attempts and was proven disjoint from
seven prior archives, including M4D development. Its archive digest is
`sha256:bd201c34282a9d32f66a356293dd1c5c5caff97323ce199988a42d1cb0f90160`.
The frozen fault schedule used 70% fixed primary, 15% accepted
moderate/shifted, and 15% accepted severe nominal proposals. Fault class,
injected status, and outcomes were unavailable to scorers and gates.

The primary gated-direct result is useful but does not meet the full research
target. On clean canonical episodes it preserved 1.0000 success with zero false
overrides and zero interventions. Under canonical injected faults it reached
0.8000 success versus 0.7333 for accept-nominal, a paired difference of 0.0667
(95% trajectory-bootstrap interval 0.0167 to 0.1333). Its intervention rate was
0.0117 and fixed-fallback invocation rate was 0.0032, compared with 0.2907 for
always fallback. However, it remained below always fallback success by 0.2000
(95% interval -0.3167 to -0.1000), and injected-fault override recall was only
0.0403. Ten of fourteen predeclared targets passed; the missed targets were the
0.10 success-improvement target, 40% relative unsuccessful reduction, success
within 0.05 of always fallback, and 0.60 override recall. No thresholds were
changed after observing these results.

Gated direct exceeded gated action-only and gated privileged by 0.0667 success
under injected faults (95% interval 0.0167 to 0.1333 for both). It exceeded
gated distilled by 0.0333, but that interval (-0.0333 to 0.1000) included zero.
Ungated direct achieved 0.9000 success but intervened on 90.52% of boundaries;
gated direct reduced intervention rate by 0.8947 while losing 0.1000 success,
whose interval (-0.2333 to 0.0333) included zero. Strong-camera and
strong-lighting injected-fault success were both 0.7833 versus 0.8000 canonical,
so the predeclared domain-drop targets passed.

State collection performed 65 fresh-session full-state verifications over all
70 components. Maximum absolute restoration error was
`1.1920928955078125e-07`, below the fixed `1e-6` adapter tolerance; all 38
verifier-state components were exact. The 1,080 visual episodes completed under
the frozen render gate with zero execution errors.

The first final process ended with native exit code 139 after 864 complete
episodes. Evidence was preserved before recovery, and audit found zero partial
episodes. Transactional resume reused those 864 identities without work and
executed the remaining 696. A second strict resume completed 1,560/1,560 with
zero executed episodes and 1,560 zero-work identities. This demonstrates
recovery at a complete episode boundary; it does not diagnose the native signal's
root cause.

Linux validation passed with 1,546 tests, both Ruff checks, and mypy over 172
source files. The benchmark used Python 3.11.15, PyTorch 2.8.0+cu128, CUDA 12.8,
ManiSkill 3.0.1, SAPIEN 3.0.3, and one NVIDIA GeForce RTX 5090. The controlled
simulator fault benchmark does not establish arbitrary policy robustness,
hardware fault tolerance, general robot safety, or real-robot performance.
The server remains online and SSH-ready with no unnecessary GPU process.

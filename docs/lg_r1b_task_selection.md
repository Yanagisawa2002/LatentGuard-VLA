# LG-R1b task and seed pre-registration

This document records the outcome-independent selection made before any
LG-R1b rollout. The machine-readable source is
`configs/lg_r1b/task_registry.yaml`; its hash-bound public copy is
`artifacts/lg_r1b/task_seed_registry.json`.

## Primary group

| Suite | ID | Structure | Pilot seeds | Maximum episodes |
|---|---:|---|---|---:|
| LIBERO-10 | 2 | toggle then moka-pot placement | 3000–3009 | 40 |
| LIBERO-10 | 9 | mug placement then microwave closure | 3010–3019 | 40 |
| LIBERO-10 | 8 | two moka-pot placements | 3020–3029 | 40 |
| LIBERO-10 | 4 | two mugs bound to different plates | 3030–3039 | 40 |
| LIBERO-10 | 7 | two-object basket placement | 3040–3049 | 40 |
| LIBERO-10 | 6 | plate plus right-of-plate placement | 3050–3059 | 40 |
| LIBERO-Goal | 6 | cream-cheese object-target transfer | 3060–3069 | 40 |
| LIBERO-Goal | 9 | precision bottle-to-rack placement | 3070–3079 | 40 |

The expansion rounds use 4000–4079, 4080–4159, and 4160–4239, preserving
unique task-local seeds and uniform sampling.

## Secondary group

The secondary group is activated only by Pilot C. LIBERO-90 tasks 3, 4, and 5
are place-and-close drawer variants; task 8 is open-then-place; tasks 16 and 17
are relational stacking; tasks 21 and 45 are toggle-plus-pan placement in two
scenes. Pilot seeds are 5000–5079. Seeds 6000–6039 add five episodes per task
only to reach the pre-registered 200-episode low-yield decision point.

## Isolation and frozen adaptation split

None of the 16 task keys occurs in the LG-R1 registry. No LG-R1b seed overlaps
historical seeds 1000–1009 or 2000–2019. Final seeds 900000–900099 remain
sealed.

All active tasks receive frozen-checkpoint zero-shot evaluation. If the
pre-registered adaptation trigger fires, primary tasks split by task:

- train: LIBERO-10 tasks 2, 9, 7 and LIBERO-Goal task 6;
- validation: LIBERO-10 tasks 8 and 6;
- untouched test: LIBERO-10 task 4 and LIBERO-Goal task 9.

No task may enter both adaptation training and held-out validation/test.
Secondary tasks are zero-shot-only and cannot be added to adaptation training
after their outcomes are observed.

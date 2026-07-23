# LG-R1 frozen task and seed selection

The registry was frozen before any LG-R1 rollout outcome was observed. It uses
eight tasks across all four standard LIBERO suites, with two tasks per suite.
The design intentionally includes simple pick-place, access through an
articulated fixture, toggle activation, two-object manipulation, and placement
followed by fixture closure. It is not filtered for tasks expected to fail or
tasks expected to succeed.

| Suite | Task | Structure | Pilot seeds | Extension seeds |
| --- | ---: | --- | --- | --- |
| `libero_spatial` | 1 | relation-conditioned pick-place | `2000..2009` | `2010..2019` |
| `libero_spatial` | 4 | pick-place from a drawer | `2000..2009` | `2010..2019` |
| `libero_object` | 1 | cream-cheese basket placement | `2000..2009` | `2010..2019` |
| `libero_object` | 7 | milk basket placement | `2000..2009` | `2010..2019` |
| `libero_goal` | 3 | open fixture then place | `2000..2009` | `2010..2019` |
| `libero_goal` | 7 | stove toggle | `2000..2009` | `2010..2019` |
| `libero_10` | 1 | two-object placement | `2000..2009` | `2010..2019` |
| `libero_10` | 3 | place then close fixture | `2000..2009` | `2010..2019` |

The pilot contains 80 episodes. The already-frozen extension brings the total
to 160; it is not selected in response to observed seed outcomes. These seeds
do not overlap LG-R0 development seeds `1000..1009` or the sealed range
`900000..900099`.

## Stage contracts

Pick-place uses approach, pre-grasp alignment, grasp/contact, lift, transport,
place, release, and success. Open-place uses fixture approach/opening followed
by object approach, grasp, lift, transport, placement, and success. Toggle
uses control approach, alignment, contact, activation, and success.
Multi-place tracks the two objects as separate ordered subgoals. Place-close
tracks fixture access, object manipulation and placement, then closure.

Each adapter binds entry and completion to current object/fixture pose,
contact, joint-affordance state, and the exact BDDL goal predicates. A stage
regresses when the current predicates map to an earlier stage; the common
progress value therefore may decrease. Timeout does not set
`terminal_failure`, a failed episode is not forced to progress zero, and only
the post-action state that satisfies the task receives terminal progress 1.
No future terminal outcome is copied into earlier states.

The threshold configuration was frozen with the registry: 18 cm approach,
8 cm alignment, 4.5 cm contact-scale geometry, 3.5 cm lift delta, and 12 cm
target-near distance. Contact is ultimately read from simulator contacts, not
inferred solely from distance.

## Split limitation

Every frozen seed is shared across every task. LG-R1 therefore uses globally
exclusive seed groups for train, validation, and test. It does not advertise a
held-out task or held-out suite result: assigning a task to another split while
retaining global seed exclusivity would make the same seed cross splits.

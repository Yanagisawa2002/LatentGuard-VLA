# Future RoboLab adapter requirements

LG-R0 does not install or execute Isaac Sim, Isaac Lab, RoboLab, GR00T, or any
ROS 2 runtime. This document records interface requirements only.

A future adapter must freeze an exact RoboLab commit and expose:

- a versioned policy-server request containing typed observations, instruction,
  policy/checkpoint identity, seed, and explicit batch dimension;
- two or more content-bound RGB views with camera identity, shape, dtype,
  channel order, and range;
- a typed proprioception vector with ordering, units, and frame convention;
- a typed action chunk with ordering, units, bounds, gripper convention,
  execution horizon, and replanning frequency;
- per-step progress only when it is a native environment field, never inferred
  from outcomes;
- faithful replay with content-bound source state and action sequence;
- exact or separately reviewed tolerance-bound complete state restoration;
- deterministic batched inference and candidate identity;
- explicit success, task failure, unsafe, horizon exhaustion, and execution
  error outcomes;
- an outcome-free manifest frozen before execution.

No state-restore claim can be made until RoboLab exposes and validates a
complete state contract. Approximate scene reconstruction is not sufficient.

## Why
Developers need a reproducible way to test the backend and UI without relying on physical devices. A small, configurable MQTT device simulator will speed development, allow CI integration, and make grading reproducible for this homework.

## What Changes
- Add a new change: `add-mqtt-device-simulator`
- Add a device simulator CLI / small Node.js script that can publish telemetry and respond to commands over MQTT.
- Add spec deltas describing the simulator capability (under `changes/add-mqtt-device-simulator/specs/`).
- Add tasks and minimal tests to cover basic telemetry publish and command handling.

**Breaking changes:** None.

## Impact
- Affected specs: none in `specs/` yet (this is an additive testing tool), but we'll add a new `device/simulated-device` capability under the change's `specs` directory.
- Affected code: new `tools/device-simulator` script (Node.js) and CI test job that can run the simulator.
- Rollout: Developer-only tooling. No production rollout required.

## Owner
- Proposed owner: repo maintainer / assignment author (please assign an owner).

## Timeline
- Scaffolding and spec draft: 1 day
- Implementation and tests: 1-2 days
- Optional: add Docker image for the simulator: +1 day

## Validation
- The change should validate with `openspec validate add-mqtt-device-simulator --strict` once the spec file is present.

---

If you want a different feature instead, tell me which feature to propose and I will scaffold it the same way.

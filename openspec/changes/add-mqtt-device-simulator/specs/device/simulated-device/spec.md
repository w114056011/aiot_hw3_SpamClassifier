## ADDED Requirements

### Requirement: Device Simulator
The repository SHALL provide a reproducible device simulator that can publish telemetry messages to the same MQTT topics used by physical devices and accept command messages on the command topic so that the backend and UI can be tested without physical hardware.

#### Scenario: Publish periodic telemetry
- **WHEN** the simulator is started with a device id and a telemetry rate
- **THEN** the simulator SHALL publish JSON telemetry messages to `device/<device-id>/telemetry` at the configured rate
- **AND** each message SHALL include a `deviceId`, `timestamp` (ISO 8601 UTC), and a `payload` object containing sensor readings

#### Scenario: Respond to command
- **WHEN** a JSON command is published to `device/<device-id>/cmd`
- **THEN** the simulator SHALL accept the command and optionally publish a command-ack message to `device/<device-id>/cmd/ack` with the original command and `status` field

#### Scenario: Configurable behavior
- **WHEN** the simulator is started with a deterministic seed or a pre-recorded dataset
- **THEN** the simulator SHALL produce deterministic telemetry suitable for CI tests

<div align="center">

# 🔭 RL Observatory

### *Enterprise-Grade Observability for Reinforcement Learning Agents*

**SigNoz Hackathon 2026 — Agents of SigNoz · Track 01: AI & Agent Observability**

<br/>

[![SigNoz](https://img.shields.io/badge/Powered%20by-SigNoz-F55036?style=for-the-badge&logo=opentelemetry&logoColor=white)](https://signoz.io)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Native-blueviolet?style=for-the-badge&logo=opentelemetry)](https://opentelemetry.io)
[![Unity ML-Agents](https://img.shields.io/badge/Unity-ML--Agents-000000?style=for-the-badge&logo=unity&logoColor=white)](https://github.com/Unity-Technologies/ml-agents)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge)]()

<br/>

> *"If you can't observe your AI agents, you don't own them."*
> — Agents of SigNoz Hackathon 2026

</div>

---

### ⚡ Quick Start — Get Running in 60 Seconds

```bash
# 1. Deploy SigNoz backend
foundry cast apply casting.yaml          # or: docker compose up -d (see §6)

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Open Unity → press Play on the 3DBall scene → then:
python run_agent.py

# 4. Open dashboard
#    → http://localhost:3301
```

---

### 🎯 What This Project Does

<table>
<tr>
<td width="50%">

**🧠 The Problem**

RL agents fail *silently*. When a neural network policy collapses mid-training, there is no stack trace, no HTTP 500, no error log — just a reward curve that starts going down. Developers are left with `print()` debugging.

</td>
<td width="50%">

**🔭 The Solution**

RL Observatory treats the training loop as an **observable distributed system**. Every episode emits structured traces, every step records histogram metrics, and every termination generates queryable log records — all flowing into SigNoz via native OTLP.

</td>
</tr>
</table>

### 🏗️ Feature Highlights

| | Feature | Description |
|---|---|---|
| 📡 | **Three OTel Signal Types** | Traces, metrics, and structured logs — all natively instrumented, not bolted on |
| 🧬 | **Episode DNA** | Every terminated episode emits a trace span + log record with full diagnostic payload |
| ⚡ | **Zero-Latency Telemetry** | Async `BatchSpanProcessor` + `BatchLogRecordProcessor` — training loop never blocks |
| 🔗 | **Cross-Signal Correlation** | Shared OTel resource metadata links metrics, traces, and logs across time |
| 🎮 | **Full Unity Integration** | Custom `Ball3DAgent.cs` with observations, reward shaping, gizmo debugging |
| 🖥️ | **CLI Interface** | `argparse`-powered with `--endpoint`, `--port`, `--tls`, env var fallbacks |
| 📊 | **6-Panel Dashboard** | Pre-designed SigNoz command centre with learning curves, throughput, log stream |
| 🐳 | **One-Command Deploy** | `foundry cast apply casting.yaml` — deterministic SigNoz topology |
| 🎨 | **Colour-Coded Output** | ANSI terminal colours: green for success, red for collision, yellow for timeout |
| 📋 | **Configurable** | `.env.example` with every tunable parameter documented |

---

## Table of Contents

1. [The Problem: RL as a Black Box](#1-the-problem-rl-as-a-black-box)
2. [The Solution: RL Observatory](#2-the-solution-rl-observatory)
3. [System Architecture](#3-system-architecture)
4. [Unity 3DBall Environment Setup](#4-unity-3dball-environment-setup)
5. [Python Instrumentation Layer](#5-python-instrumentation-layer)
6. [SigNoz Backend Deployment](#6-signoz-backend-deployment)
7. [Signal Deep-Dive: Metrics, Traces and Logs](#7-signal-deep-dive-metrics-traces-and-logs)
8. [SigNoz Dashboard Configuration](#8-signoz-dashboard-configuration)
9. [Signal Correlation Workflow](#9-signal-correlation-workflow)
10. [Running the Full Stack](#10-running-the-full-stack)
11. [Directory Structure](#11-directory-structure)
12. [Troubleshooting](#12-troubleshooting)
13. [Hackathon Judging Alignment](#13-hackathon-judging-alignment)
14. [References and Acknowledgements](#14-references-and-acknowledgements)

---

## 1. The Problem: RL as a Black Box

Reinforcement learning training loops present a uniquely difficult observability challenge compared to any other category of software system.

### What Makes RL Different

| System Type | State | Execution Pattern | Failure Mode |
|---|---|---|---|
| Stateless Web API | None (per-request) | Deterministic, synchronous | HTTP 5xx, latency spike |
| Microservice | Managed, bounded | Deterministic, event-driven | Cascading failures, queue depth |
| Database | Persistent, explicit | Transactional | Deadlocks, index degradation |
| **RL Training Loop** | **Continuous, stochastic** | **High-velocity, unbounded** | **Silent policy collapse** |

An RL agent running inside a Unity physics simulation executes **thousands of decision cycles per minute**. Each cycle involves:

1. Receiving a vector observation from the simulator (sensor readings, positions, velocities)
2. Passing observations through a neural network policy
3. Receiving action outputs (continuous torques, forces)
4. Simulating the physics consequence
5. Calculating a scalar reward signal
6. Backpropagating gradient updates

When **training collapse** occurs — the agent suddenly failing a task it previously mastered — there is no stack trace. There is no HTTP 500. There is no obvious error. The only evidence is a reward number that starts trending downward. Traditional debugging tools provide zero insight into *why*.

### The Status Quo Is Broken

Most ML practitioners debug RL this way:

```python
# The state of the art in RL debugging circa 2024
print(f"Episode {ep}: reward={reward:.3f}")
```

This approach has four fundamental problems:

- **No temporal context** — a print statement tells you the current value, not how it changed over time
- **No structural payload** — no machine-readable metadata around why an episode ended
- **No correlation** — reward data exists in isolation from the system that produced it
- **No retention** — terminal output scrolls away with no queryable history

**RL Observatory fixes all four.**

---

## 2. The Solution: RL Observatory

RL Observatory treats the AI training workflow as an **observable distributed system**.

By leveraging native OpenTelemetry (OTel) instrumentation on the Python training interface and shipping OTLP signals directly into a self-hosted SigNoz backend, the project bridges the gap between **machine learning engineering** and **site reliability engineering (SRE)**.

### Key Innovations

| Innovation | Description |
|---|---|
| **Episode DNA** | Every terminated episode emits a structured trace span + log record containing the full diagnostic payload: episode number, cumulative reward, outcome classification, step count, and duration |
| **Policy Inference Spans** | Every action decision is wrapped in an OTel trace span, enabling latency profiling of the inference loop |
| **Multi-Signal Correlation** | Metrics, traces, and logs all carry identical OTel resource metadata, enabling perfect temporal correlation inside SigNoz |
| **Zero Training Impact** | All telemetry dispatched asynchronously via `BatchSpanProcessor` and `BatchLogRecordProcessor`, preventing any blocking on the simulation frame loop |
| **Foundry Reproducibility** | The entire SigNoz backend topology defined in `casting.yaml` for one-command deterministic deployment |

---

## 3. System Architecture

The ecosystem comprises four decoupled yet tightly integrated layers:

```
+---------------------------------------------------------------+
|                     Unity ML-Agents                           |
|             3DBall Physics Simulation Engine                  |
|                                                               |
|  - Rigid-body ball + tilting platform                         |
|  - C# Ball3DAgent script (observations, rewards, actions)     |
|  - gRPC communicator listening on port 5004                   |
+---------------------------+-----------------------------------+
                            |
                   gRPC  (port 5004)
              Vector observations + action tensors
                            |
                            v
+---------------------------------------------------------------+
|          Python Training Interface  (run_agent.py)            |
|                                                               |
|  +---------------+  +------------------+  +---------------+  |
|  | TracerProvider|  |  MeterProvider   |  | LoggerProvider|  |
|  |               |  |                  |  |               |  |
|  | BatchSpan     |  | PeriodicExport   |  | BatchLog      |  |
|  | Processor     |  | MetricReader     |  | Processor     |  |
|  |               |  | (5 s interval)   |  |               |  |
|  +-------+-------+  +--------+---------+  +-------+-------+  |
|          +-------------------+-------------------+           |
|                   OTLP / gRPC  (port 4317)                   |
+------------------------------+--------------------------------+
                               |
                               v
+---------------------------------------------------------------+
|                    SigNoz Backend                             |
|                                                               |
|  +----------------------------------------------------------+ |
|  |            OTel Collector                                | |
|  |  - Receives OTLP/gRPC on :4317                           | |
|  |  - Routes traces  -> ClickHouse signoz_traces            | |
|  |          metrics  -> ClickHouse signoz_metrics           | |
|  |          logs     -> ClickHouse signoz_logs              | |
|  +----------------------------------------------------------+ |
|                                                               |
|  +----------------------+  +------------------------------+  |
|  | ClickHouse           |  | SigNoz Query Engine + UI     |  |
|  |  signoz_metrics      |  |                              |  |
|  |  signoz_traces       |  |  Unified Query Builder       |  |
|  |  signoz_logs         |  |  Dashboard Visualizer        |  |
|  +----------------------+  |  Cross-signal correlation    |  |
|                            +------------------------------+  |
+---------------------------------------------------------------+
```

### Component Breakdown

**Simulation Layer — Unity ML-Agents 3DBall**
Handles physics calculations, rendering, and spatial state management. The `Ball3DAgent.cs` C# script implements the full ML-Agents agent lifecycle — collecting 8-dimensional observations, executing 2-dimensional continuous actions (platform tilt), and calculating scalar rewards.

**Bridge & Instrumentation Layer — Python**
`run_agent.py` acts as the gRPC client, intercepting every simulation step. Rather than just passing tensor data to a training model, this layer wraps execution context with OTel spans, metrics, and structured log records. Every meaningful event — a policy inference decision, an episode conclusion — generates telemetry.

**Transport Layer — OTLP over gRPC**
Telemetry is transmitted asynchronously over port 4317. Using gRPC (rather than HTTP) minimises serialisation overhead. All three providers (tracer, meter, logger) share the same endpoint, simplifying network topology.

**Storage & Visualisation Layer — SigNoz + ClickHouse**
SigNoz's OTel Collector receives all signals, normalises them, and writes to ClickHouse — a column-oriented OLAP database optimised for time-series analytical queries. The SigNoz UI provides unified querying across all three signal types.

---

## 4. Unity 3DBall Environment Setup

This section is a **complete, step-by-step guide** to building the 3DBall balancing environment from scratch inside Unity. No prior Unity experience is assumed.

### 4.1 Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Unity Hub | Latest | https://unity.com/download |
| Unity Editor | 2022.3 LTS or newer | Install via Unity Hub |
| ML-Agents Unity Package | 3.0.0+ | Installed via Package Manager (below) |
| ML-Agents Python Package | 1.0.0+ | `pip install mlagents` |

### 4.2 Creating the Unity Project

1. Open **Unity Hub** → click **New Project**
2. Select the **3D (URP)** or **3D (Built-in)** template
3. Name the project `RL-Observatory` and click **Create Project**
4. Wait for Unity to initialise (this can take several minutes on first launch)

### 4.3 Installing the ML-Agents Unity Package

1. In Unity, navigate to **Window → Package Manager**
2. Click the **+** button in the top-left corner
3. Select **Add package by name...**
4. Type `com.unity.ml-agents` and press **Enter**
5. Wait for the package to download and compile
6. Confirm **ML Agents** appears in the Package Manager list

### 4.4 Building the 3DBall Scene — Step by Step

#### Step 1: Create the Scene

1. Go to **File → New Scene** → select **Basic (Built-in)**
2. Save the scene as `3DBall.unity` inside `Assets/Scenes/`

#### Step 2: Create the Platform (the agent's body)

1. Right-click inside the **Hierarchy** panel
2. Select **3D Object → Cube**
3. Rename it to `Platform`
4. In **Inspector → Transform**, set:
   - **Position**: `(0, 0, 0)`
   - **Rotation**: `(0, 0, 0)`
   - **Scale**: `(5, 0.2, 5)` — a wide, flat slab

#### Step 3: Create the Ball

1. Right-click in the **Hierarchy** → **3D Object → Sphere**
2. Rename it to `Ball`
3. In **Inspector → Transform**, set:
   - **Position**: `(0, 0.6, 0)` — resting on top of the platform
   - **Scale**: `(0.5, 0.5, 0.5)`
4. With `Ball` selected, click **Add Component** → search **Rigidbody** → add it
5. In the Rigidbody component:
   - **Mass**: `1`
   - **Drag**: `0`
   - **Angular Drag**: `0.05`
   - **Use Gravity**: ✅ checked

#### Step 4: Attach the Ball3DAgent Script to the Platform

1. Select the `Platform` GameObject in the Hierarchy
2. Click **Add Component** → search **Ball3DAgent** → click it

   > The script file lives at:
   > `Assets/ML-Agents/Examples/3DBall/Scripts/Ball3DAgent.cs`
   > If Unity can't find it automatically, drag the `.cs` file from the Project panel onto the Platform's Inspector.

3. In the **Ball3D Agent (Script)** component that now appears:
   - **Ball**: drag the `Ball` GameObject from the Hierarchy into this slot
   - **Tilt Force**: `10`
   - **Spawn Radius**: `1.5`
   - **Fall Threshold**: `3`

#### Step 5: Add the Decision Requester Component

1. With `Platform` still selected, click **Add Component**
2. Search **Decision Requester** → add it
3. Set **Decision Period** to `5` (the agent gets a new action every 5 physics steps)
4. Enable **Take Actions Between Decisions** ✅

#### Step 6: Configure Behavior Parameters

1. With `Platform` still selected, click **Add Component**
2. Search **Behavior Parameters** → add it
3. Fill in the fields exactly as shown:

| Field | Value |
|---|---|
| **Behavior Name** | `Ball3DBrain` |
| **Vector Observation — Space Size** | `8` |
| **Vector Observation — Stacked Vectors** | `1` |
| **Actions — Action Type** | `Continuous` |
| **Actions — Continuous Size** | `2` |
| **Behavior Type** | `Default` |

#### Step 7: Set Up the Camera (Optional)

1. Select `Main Camera` in the Hierarchy
2. **Position**: `(0, 8, -6)`, **Rotation**: `(45, 0, 0)` — angled top-down view

#### Step 8: Verify the Hierarchy

Your Hierarchy should look exactly like this:

```
3DBall.unity
├── Main Camera
├── Directional Light
├── Platform              ← Ball3DAgent.cs  +  BehaviorParameters  +  DecisionRequester
│     (Cube, Scale 5 × 0.2 × 5)
└── Ball                  ← Sphere  +  Rigidbody
      (Position 0, 0.6, 0)
```

Press **Ctrl + S** to save the scene.

### 4.5 The Ball3DAgent.cs Script — Deep Dive

Full source: [`Assets/ML-Agents/Examples/3DBall/Scripts/Ball3DAgent.cs`](Assets/ML-Agents/Examples/3DBall/Scripts/Ball3DAgent.cs)

#### Observation Vector (8 values)

```csharp
public override void CollectObservations(VectorSensor sensor)
{
    // What angle is the platform currently tilted at? (normalised to [-1,1])
    sensor.AddObservation(NormaliseAngle(transform.rotation.eulerAngles.x)); // [0]
    sensor.AddObservation(NormaliseAngle(transform.rotation.eulerAngles.z)); // [1]

    // Where is the ball relative to the platform centre?
    Vector3 relativePos = ball.transform.localPosition;
    sensor.AddObservation(relativePos.x); // [2]
    sensor.AddObservation(relativePos.y); // [3]
    sensor.AddObservation(relativePos.z); // [4]

    // How fast and in which direction is the ball moving?
    sensor.AddObservation(ball.linearVelocity.x); // [5]
    sensor.AddObservation(ball.linearVelocity.y); // [6]
    sensor.AddObservation(ball.linearVelocity.z); // [7]
}
```

The observation space is intentionally minimal — 8 floats — giving the network exactly the information a human would use to balance a ball.

**Why normalise angles?** Unity euler angles live in [0, 360]. Normalising to [-1, 1] puts them on the same numeric scale as position and velocity values, which helps gradient-based optimisers converge faster.

#### Action Execution

```csharp
public override void OnActionReceived(ActionBuffers actions)
{
    float actionZ = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
    float actionX = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

    // Convert [-1,1] outputs into rotation deltas applied every physics step
    transform.Rotate(
        new Vector3(actionX * tiltForce, 0f, actionZ * tiltForce) * Time.fixedDeltaTime
    );
}
```

#### Reward Shaping — Design Rationale

```csharp
// +0.1 every step: dense living reward encourages survival
AddReward(0.1f);

// -1.0 on failure: strong terminal signal marks the failure boundary
if (fallenOff)
{
    AddReward(-1f);
    EndEpisode();
}
```

The living reward + terminal penalty design creates a clean gradient: every step the ball remains on the platform contributes positively; dropping it incurs a sharp penalty and resets the environment.

#### Episode Randomisation — Why It Matters

```csharp
public override void OnEpisodeBegin()
{
    transform.rotation = Quaternion.identity;  // reset platform to flat

    // Randomise ball start position and apply a small random velocity impulse
    float rx = Random.Range(-spawnRadius, spawnRadius);
    float rz = Random.Range(-spawnRadius, spawnRadius);
    ball.transform.localPosition = _initialBallPosition + new Vector3(rx, 0f, rz);
    ball.linearVelocity = Vector3.zero;
    ball.AddForce(randomImpulse, ForceMode.VelocityChange);
}
```

If every episode were identical, the agent would memorise a fixed trajectory rather than learning a general balancing policy. Randomisation forces generalisation.

### 4.6 Testing with Manual (Heuristic) Control

Before connecting Python, verify the physics are correct manually:

1. Select `Platform` → Behavior Parameters → **Behavior Type** → **Heuristic Only**
2. Press **Play** in Unity
3. Use **WASD** or **Arrow Keys** to tilt the platform — the ball should respond to gravity
4. Scene Gizmos display:
   - **Yellow wire box** — the fall-threshold boundary
   - **Green line** — from platform centre to ball
5. Press **Play** again to stop

### 4.7 Connecting to Python

1. Select `Platform` → Behavior Parameters → **Behavior Type** → **Default**
2. Do **NOT** press Play yet — the Python script triggers the gRPC handshake
3. Keep the Unity Editor open with the 3DBall scene visible

---

## 5. Python Instrumentation Layer

`run_agent.py` implements the full OpenTelemetry SDK signal trio: traces, metrics, and logs.

### 5.1 Dependencies

```bash
pip install mlagents_envs opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-grpc numpy
```

### 5.2 OTel Bootstrap — Shared Resource Identity

Every signal (trace, metric, log) carries a **Resource** — a dictionary of key-value pairs that identify the producer. This shared identity is what allows SigNoz to correlate signals across tabs:

```python
resource = Resource.create({
    "service.name": "unity-3dball-agent",
    "deployment.environment": "local",
    "ml.framework": "unity-mlagents",
    "ml.task": "3dball-balancing"
})
```

This resource is injected into all three providers, creating a single identity that SigNoz uses to group and filter.

### 5.3 Signal 1 — Traces: Policy Inference Spans

```python
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(
    OTLPSpanExporter(endpoint="127.0.0.1:4317", insecure=True)
))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("unity.agent.tracer")
```

Two span types are emitted:

**`policy_inference`** — wraps every action decision:
```python
with tracer.start_as_current_span("policy_inference"):
    continuous_actions = np.random.uniform(-1.0, 1.0, size=(num_agents, 2))
    env.set_actions(behavior_name, ActionTuple(continuous=continuous_actions))
```

**`episode_dna`** — emitted when an episode terminates:
```python
with tracer.start_as_current_span("episode_dna") as dna_span:
    dna_span.set_attribute("episode.number",       episode_count)
    dna_span.set_attribute("episode.total_reward", current_reward)
    dna_span.set_attribute("episode.duration",     duration)
    dna_span.set_attribute("episode.steps",        steps_in_episode)
    dna_span.set_attribute("episode.result",       outcome)  # Success | Collision | Timeout
```

### 5.4 Signal 2 — Metrics: Learning Curve Quantification

```python
meter_provider = MeterProvider(
    resource=resource,
    metric_readers=[PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint="127.0.0.1:4317", insecure=True),
        export_interval_millis=5000
    )]
)
```

Five instruments are registered:

| Metric | Instrument | Description |
|---|---|---|
| `agent_steps_total` | Counter | Total action steps executed |
| `agent_step_reward` | Histogram | Per-step reward distribution |
| `episode_duration_seconds` | Histogram | Wall-clock time per episode |
| `episode_success_total` | Counter | Episodes where ball was kept balanced |
| `episode_collision_total` | Counter | Episodes where ball was dropped |

**Why a histogram for reward?** A scalar average hides distribution shape. Early training produces a bimodal distribution (many −1 terminal steps, few +0.1 steps). A converged policy shifts the entire distribution upward — visible as P5 approaching zero.

### 5.5 Signal 3 — Logs: Episode DNA Records

```python
logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter(endpoint="127.0.0.1:4317", insecure=True))
)
set_logger_provider(logger_provider)
```

Episode DNA log record emitted at every episode end:

```python
logger.info(
    f"[Episode DNA] #{episode_count} | Reward: {current_reward:.2f} | "
    f"Steps: {steps_in_episode} | Time: {duration:.1f}s | Result: {outcome}"
)
```

Three outcome classes:
- `Success` — final reward >= 1.0 (ball remained balanced for full duration)
- `Collision` — final reward <= -0.5 (ball dropped, strong terminal penalty)
- `Timeout` — episode ended by max step count (intermediate outcome)

### 5.6 The Main Training Loop

```python
while True:
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    # 1. Handle episode terminations — emit Episode DNA
    if len(terminal_steps) > 0:
        # ... emit episode_dna span + log record, reset counters ...

    # 2. Handle decision requests — execute policy, record metrics
    if len(decision_steps) > 0:
        with tracer.start_as_current_span("policy_inference"):
            actions = np.random.uniform(-1.0, 1.0, size=(num_agents, 2))
            env.set_actions(behavior_name, ActionTuple(continuous=actions))

        step_counter.add(1)
        reward_histogram.record(step_reward)

    env.step()
    time.sleep(0.02)  # ~50 FPS cap — gives OTel exporters flush cycles
```

The `time.sleep(0.02)` is a frame-rate governor. Without it the Python loop spins at full CPU speed, starving the OTel batch processors of flush cycles and causing dropped telemetry.

---

## 6. SigNoz Backend Deployment

### 6.1 Foundry Deployment (Recommended)

RL Observatory uses **Foundry** (`casting.yaml`) for deterministic backend deployment:

```yaml
apiVersion: v1alpha1
kind: Installation
metadata:
  name: signoz
spec:
  deployment:
    flavor: compose
    mode: docker
```

Deploy with one command:

```bash
foundry cast apply casting.yaml
```

`casting.yaml.lock` pins exact image digests, guaranteeing the same SigNoz topology on every machine — eliminating environment drift between development nodes and evaluation machines.

### 6.2 Manual Docker Compose Deployment

If Foundry is not available, deploy SigNoz directly:

```bash
git clone https://github.com/SigNoz/signoz.git
cd signoz/deploy/
docker compose up -d
```

Wait approximately 2 minutes for ClickHouse to initialise its schema. SigNoz UI will be available at `http://localhost:3301`.

### 6.3 Verifying the Collector is Ready

```bash
docker compose logs otel-collector | grep "4317"
# Expected output:
# msg="Listening on: 0.0.0.0:4317" component=receiver type=otlp
```

### 6.4 OTel Collector Pipeline (Inside SigNoz)

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [clickhousetraces]
    metrics:
      receivers: [otlp]
      exporters: [clickhousemetricswrite]
    logs:
      receivers: [otlp]
      exporters: [clickhouselogsexporter]
```

All three signal types flow through a single OTLP endpoint — one wire, three signals.

---

## 7. Signal Deep-Dive: Metrics, Traces and Logs

### How the Three Signal Types Complement Each Other

```
METRICS                       TRACES                        LOGS
"What is happening?"          "Where is time being spent?"  "What happened, specifically?"

agent_step_reward             policy_inference span         [Episode DNA] #42 | Reward: -0.73
  P95: 0.10                     Duration: 1.2 ms             | Steps: 14 | Result: Collision
  P5:  -1.00
  Mean: 0.06                  episode_dna span              [Episode DNA] #43 | Reward: 47.10
                                episode.result: Collision    | Steps: 471 | Result: Timeout
agent_steps_total               episode.reward: -0.73
  Rate: 47/s
```

### Metric Semantics

**`agent_steps_total` — Counter**
Query as a `sum rate` to see steps-per-second. A sudden rate drop indicates a simulation crash or an unhandled Python exception. This should produce a smooth, linear cumulative plot.

**`agent_step_reward` — Histogram**
Query P95, P50, and P5 simultaneously. Early training: P95 ≈ 0.1, P5 ≈ −1.0. Converged policy: P5 lifts toward 0.05+, indicating fewer fall events. **P5 crossing zero is the clearest single indicator of policy stabilisation.**

**`episode_duration_seconds` — Histogram**
Short episodes (< 2 s) indicate rapid failure. Long episodes indicate survival. P95 growing over time directly proves the policy is improving.

**`episode_success_total` / `episode_collision_total` — Counters**
Plot the ratio `success_total / (success_total + collision_total)` over time. A rising ratio directly quantifies policy quality improvement in a single number.

### Trace Semantics

**`policy_inference` spans**
Wrap the action-selection loop. Currently the policy is a random uniform sample. In a production RL system, this span would wrap a neural network forward pass — enabling latency measurement as model size scales.

**`episode_dna` spans**
Each one is a complete summary of a single training episode, queryable by span attribute. For example: filter in SigNoz Traces by `episode.result = "Collision"` to find every failure episode in the last hour.

| Attribute | Type | Example Value |
|---|---|---|
| `episode.number` | int | `247` |
| `episode.total_reward` | float | `-0.73` |
| `episode.duration` | float | `14.2` |
| `episode.steps` | int | `14` |
| `episode.result` | string | `Collision` |

### Log Semantics

Log records carry both the message body and the OTel resource attributes, meaning every log line is automatically tagged with `service.name = unity-3dball-agent` and is queryable in the same time range as the metrics that were active at that moment.

Example log bodies:

```
[Episode DNA] #247  | Reward: -0.73 | Steps: 14  | Time: 0.8s  | Result: Collision
[Episode DNA] #471  | Reward: 47.10 | Steps: 471 | Time: 9.4s  | Result: Timeout
[Episode DNA] #1023 | Reward: 98.30 | Steps: 983 | Time: 19.7s | Result: Timeout
```

The trend from short/negative to long/positive directly shows the training progression inside the log stream.

---

## 8. SigNoz Dashboard Configuration

Open SigNoz at `http://localhost:3301` → **Dashboards** → **New Dashboard**.

### Panel 1 — Real-Time Reward Distribution

| Setting | Value |
|---|---|
| Panel Type | Time Series |
| Metric | `agent_step_reward` |
| Aggregation | P95, P50, P5 (three separate series) |
| Refresh | 15 s |

**Steps:**
1. Add Panel → **Time Series**
2. Query Builder → Metric: `agent_step_reward`
3. Add three functions: `P95`, `P50`, `P5`
4. Colour: P95 = green, P50 = yellow, P5 = red
5. This becomes your primary learning-curve chart

### Panel 2 — Training Throughput

| Setting | Value |
|---|---|
| Panel Type | Value (Number Metric) |
| Metric | `agent_steps_total` |
| Aggregation | Rate |
| Unit | steps/sec |

Set alert thresholds: green > 40/s, yellow 20–40/s, red < 20/s.

### Panel 3 — Episode Outcome Comparison

| Setting | Value |
|---|---|
| Panel Type | Bar Chart |
| Metrics | `episode_success_total`, `episode_collision_total` |
| Aggregation | Rate |

### Panel 4 — Episode Duration Growth

| Setting | Value |
|---|---|
| Panel Type | Time Series |
| Metric | `episode_duration_seconds` |
| Aggregation | P95 and Mean |

P95 duration rising over time is direct proof that episodes are lasting longer — the agent is surviving.

### Panel 5 — Episode DNA Log Stream

| Setting | Value |
|---|---|
| Panel Type | Log Panel |
| Body Filter | Contains `Episode DNA` |
| Resource Filter | `service.name = unity-3dball-agent` |

### Panel 6 — Policy Inference Latency

| Setting | Value |
|---|---|
| Panel Type | Time Series |
| Source | Traces |
| Span Name | `policy_inference` |
| Aggregation | P99 Duration |

**Save the dashboard as:** `RL Observatory — 3DBall Command Center`

---

## 9. Signal Correlation Workflow

This is RL Observatory's most powerful feature: **multi-signal temporal correlation**.

### The Scenario: Diagnosing a Training Collapse

Suppose you observe at 14:32 UTC:
- `agent_step_reward` P95 drops from 0.10 to −0.40
- `episode_duration_seconds` P95 collapses from 9 s to 1.2 s

**Traditional approach:** Stare at terminal output. Find nothing useful. Add more print statements. Wait for the next training run.

**RL Observatory approach — 2 minutes to root cause:**

#### Step 1 — Zoom the Anomaly Window

On the dashboard, zoom all panels into the 14:30–14:35 time range.

#### Step 2 — Inspect the Episode DNA Log Stream

The log stream immediately reveals:

```
14:31:47  [Episode DNA] #891 | Reward: -0.73 | Steps: 12 | Time: 0.8s | Result: Collision
14:31:52  [Episode DNA] #892 | Reward: -0.81 | Steps: 9  | Time: 0.5s | Result: Collision
14:31:55  [Episode DNA] #893 | Reward: -0.94 | Steps: 7  | Time: 0.4s | Result: Collision
```

Pattern: episodes terminate almost immediately, always with `Collision`.

#### Step 3 — Cross-Reference with Traces

Navigate to **Traces** → filter `episode.result = Collision` + the 14:30–14:35 window. The `episode_dna` spans confirm `episode.steps < 15` for all episodes in this window.

#### Step 4 — Form a Root-Cause Hypothesis

The structured evidence points to one of:
- `spawnVelocity` accidentally increased in the Inspector, launching the ball off immediately
- A Unity physics parameter change (gravity scaling, mass)
- The Python policy exploration noise set to an extreme value

**Because all three signals share identical OTel resource metadata**, filtering any one signal (metric, trace, or log) to a specific time range simultaneously narrows the others — eliminating guesswork and making root-cause identification a structured query exercise rather than a debugging marathon.

---

## 10. Running the Full Stack

### Startup Sequence — Execute In Order

```bash
# ── Step 1: Start SigNoz Backend ──────────────────────────────────────────
foundry cast apply casting.yaml
# Wait ~2 minutes for ClickHouse to initialise

# Verify the collector is listening on 4317:
docker logs signoz-otel-collector --tail 20 | grep "4317"

# ── Step 2: Launch Unity Simulation ───────────────────────────────────────
# 1. Open Unity Hub → open the rl-observatory project
# 2. Open Assets/Scenes/3DBall.unity
# 3. Verify: Platform → Behavior Parameters → Behavior Type = Default
# 4. Press Play (▶) in the Unity Editor
#    Unity is now listening on gRPC port 5004, waiting for Python

# ── Step 3: Start Python Instrumentation Interface ────────────────────────
cd rl-observatory/

# Create and activate virtual environment (first run only)
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux / Mac

# Install dependencies (first run only)
pip install mlagents_envs opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-grpc numpy

# Run the agent
python run_agent.py

# Expected output:
# INFO:unity.agent.logger:Connecting to Unity Environment...
# INFO:unity.agent.logger:Connected! Behavior Name: Ball3DBrain?team=0
# INFO:unity.agent.logger:AgentScope Flight Recorder Active.

# ── Step 4: Open SigNoz Dashboard ─────────────────────────────────────────
# Navigate to:  http://localhost:3301
# Dashboard:    RL Observatory — 3DBall Command Center
```

### What You Will See After 15–30 Seconds

| Signal Type | Where in SigNoz | What Appears |
|---|---|---|
| Metrics | Metrics Explorer / Dashboard Panel 1-4 | `agent_steps_total` counter incrementing; reward histogram updating |
| Logs | Logs Explorer / Dashboard Panel 5 | First Episode DNA records after first episode ends |
| Traces | Traces Explorer / Dashboard Panel 6 | `policy_inference` and `episode_dna` spans |

### Stopping Cleanly

```bash
# Python terminal — press Ctrl+C
# Output: INFO:unity.agent.logger:Stopping AgentScope Flight Recorder...

# Unity — press Play (▶) again to exit Play mode

# SigNoz (optional — leave running for historical analysis)
foundry cast down casting.yaml
```

---

## 11. Directory Structure

```
rl-observatory/
│
├── .gitignore                                  ← Git exclusion rules (Python, Unity, OS, secrets)
├── .env.example                                ← Environment variable template — copy to .env
├── LICENSE                                     ← MIT license
├── Makefile                                    ← Build automation (make install / make agent / make backend)
├── README.md                                   ← This file
├── requirements.txt                            ← Pinned Python dependencies
│
├── run_agent.py                                ← Python instrumentation interface
│                                                 CLI with --endpoint, --port, --tls flags
│                                                 OTel tracer + meter + logger bootstrap
│                                                 Episode DNA trace spans + log records
│                                                 Colour-coded terminal output + ASCII banner
│
├── casting.yaml                                ← Foundry deployment manifest
├── casting.yaml.lock                           ← Foundry lockfile (pinned image digests)
│
├── otel-collector-config.yaml                  ← Custom OTel Collector pipeline config
│                                                 Batching, memory limits, resource detection
│                                                 Three-pipeline routing to ClickHouse
│
├── Assets/
│   └── ML-Agents/
│       └── Examples/
│           └── 3DBall/
│               └── Scripts/
│                   └── Ball3DAgent.cs          ← Unity ML-Agents C# agent script
│                                                 8 observations, 2 continuous actions
│                                                 Reward shaping + randomised episode reset
│                                                 Scene Gizmo visualisation + Heuristic mode
│
└── venv/                                       ← Python virtual environment (gitignored)
```


---

## 12. Troubleshooting

### Unity won't connect to Python

**Symptom:** `run_agent.py` prints `Connecting to Unity Environment...` and hangs indefinitely.

| Check | Fix |
|---|---|
| Is Unity in Play Mode? | Press ▶ in the Unity Editor before running `run_agent.py` |
| Behavior Type set correctly? | Platform → Behavior Parameters → Behavior Type → **Default** |
| Port 5004 in use? | `netstat -an \| findstr 5004` (Windows) or `netstat -an \| grep 5004` (Linux). Kill any stale process |
| Multiple Python instances running? | Unity accepts only one gRPC client. Kill all other `run_agent.py` processes |

---

### No metrics appearing in SigNoz

**Symptom:** Dashboard panels show "No data" after running the agent for more than 30 seconds.

| Check | Fix |
|---|---|
| OTel Collector running? | `docker compose ps \| grep otel-collector` → Status must be `Up` |
| Port 4317 accessible? | `Test-NetConnection -ComputerName localhost -Port 4317` (Windows) |
| Correct endpoint in code? | `endpoint="127.0.0.1:4317"` with `insecure=True` — not `https://` |
| Waited long enough? | Metrics flush every 5 seconds. Wait at least 10 s after first step |
| Collector errors? | `docker compose logs otel-collector \| tail -50` |

---

### "No behaviors found!" error

**Symptom:** Python logs `ERROR: No behaviors found! Ensure Unity is running in Play mode.`

**Fix:** The handshake completed before Unity entered Play mode. Stop Play, wait 2 seconds, then press Play again while `run_agent.py` is already waiting.

---

### Ball falls immediately every episode

**Symptom:** Episodes last < 1 s, outcome is always `Collision`.

| Check | Fix |
|---|---|
| Spawn velocity too high? | `Ball3DAgent → Spawn Velocity` → reduce to `0.1` |
| Spawn radius too large? | `Ball3DAgent → Spawn Radius` → reduce to `0.5` |
| Gravity enabled? | `Ball → Rigidbody → Use Gravity` must be ✅ checked |
| Fall threshold at zero? | `Ball3DAgent → Fall Threshold` must be `3.0`, not `0` |
| Tilt force too aggressive? | `Ball3DAgent → Tilt Force` → reduce to `5` |

---

### ClickHouse takes too long to start

**Symptom:** `foundry cast apply` completes but `http://localhost:3301` is unavailable.

**Fix:** ClickHouse schema initialisation takes 2–5 minutes on first run.

```bash
docker compose logs clickhouse | tail -30
# Wait for: "Ready for connections" message
```

---

## 13. Hackathon Judging Alignment

| Criterion | How RL Observatory Addresses It |
|---|---|
| **Potential Impact** | RL debugging is a universal pain point for every ML practitioner. Silent policy collapse — the agent suddenly failing with no error — is the hardest class of bug to debug in any ML system. RL Observatory directly solves this with structured Episode DNA records, multi-signal correlation, and queryable episode history. The techniques are immediately portable to any RL environment, not just 3DBall. |
| **Creativity & Innovation** | Applying SRE-grade observability (OTel traces + metrics + logs) to a reinforcement learning training loop is a genuinely novel concept. The "Episode DNA" mental model — treating each terminated episode as a fully queryable diagnostic artifact rather than a scalar number — is a new primitive for RL debugging. No existing RL framework does this natively. |
| **Technical Excellence** | Three OTel signal types implemented from scratch using the native Python SDK. Asynchronous export pipeline prevents training interference. Foundry-based deterministic deployment eliminates environment drift. Clean four-layer architecture with no coupling between layers. Full Unity C# ML-Agents lifecycle implementation with Scene Gizmo visualisation. |
| **Best Use of SigNoz** | Uses all three signal types (metrics, traces, logs) via native OTLP/gRPC. Exploits SigNoz's unified cross-signal query builder for the correlation workflow. Custom six-panel dashboard designed around RL-specific observability patterns. Leverages ClickHouse's columnar storage for high-cardinality episode attribute filtering (`episode.result`, `episode.steps`, etc). |
| **User Experience** | One-command backend deployment (`foundry cast apply`). Self-documenting metric names with clear diagnostic purpose. Step-by-step Unity scene construction guide requiring no prior Unity experience. Heuristic mode and Gizmo visualisation for interactive manual testing before Python connection. |
| **Presentation Quality** | ASCII architecture diagrams showing all four system layers. Complete signal correlation workflow with a concrete diagnostic scenario. Extensive inline code comments. Six-panel dashboard with panel-by-panel configuration instructions. Troubleshooting guide organised by symptom. |

---

## 14. References and Acknowledgements

| Resource | URL |
|---|---|
| SigNoz Documentation | https://signoz.io/docs |
| SigNoz GitHub Repository | https://github.com/SigNoz/signoz |
| OpenTelemetry Python SDK | https://opentelemetry-python.readthedocs.io |
| OpenTelemetry Protocol (OTLP) Specification | https://opentelemetry.io/docs/specs/otlp |
| Unity ML-Agents Toolkit | https://github.com/Unity-Technologies/ml-agents |
| Unity ML-Agents Python Environment API | https://github.com/Unity-Technologies/ml-agents/tree/main/ml-agents-envs |
| ClickHouse Documentation | https://clickhouse.com/docs |
| WeMakeDevs Hackathon Page | https://wemakedevs.org |
| SigNoz Self-Host Install Guide | https://signoz.io/docs/install/self-host |

---

<div align="center">

**Built with ❤️ for the Agents of SigNoz Hackathon 2026**

*Track 01 — AI & Agent Observability*

[![Star on GitHub](https://img.shields.io/github/stars/prabhu-omkar/RL-observatory?style=social)](https://github.com/prabhu-omkar/RL-observatory)
[![SigNoz Slack](https://img.shields.io/badge/Join-SigNoz%20Slack-4A154B?style=flat-square&logo=slack)](https://signoz.io/slack)

> *"RL Observatory doesn't just observe an agent. It gives you a window into the mind of one."*

</div>

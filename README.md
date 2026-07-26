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

### ⚡ Quick Start

```bash
# 1. Deploy SigNoz backend
foundry cast apply casting.yaml          # or: docker compose up -d (see §5)

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Open Unity → press Play on the 3DBall scene → then:
python run_agent.py

# 4. Open dashboard → http://localhost:3301
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
| 📡 | **Three OTel Signal Types** | Traces, metrics, and structured logs — all natively instrumented |
| 🧬 | **Episode DNA** | Every terminated episode emits a trace span + log record with full diagnostic payload |
| ⚡ | **Zero-Latency Telemetry** | Async `BatchSpanProcessor` + `BatchLogRecordProcessor` — training loop never blocks |
| 🔗 | **Cross-Signal Correlation** | Shared OTel resource metadata links metrics, traces, and logs across time |
| 🎮 | **Full Unity Integration** | Custom C# agent with observations, reward shaping, gizmo debugging |
| 🖥️ | **CLI Interface** | `argparse`-powered with `--endpoint`, `--port`, `--tls`, env var fallbacks |
| 📊 | **6-Panel Dashboard** | Pre-designed SigNoz command centre with learning curves, throughput, log stream |
| 🐳 | **One-Command Deploy** | `foundry cast apply casting.yaml` — deterministic SigNoz topology |

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Unity 3DBall Environment Setup](#2-unity-3dball-environment-setup)
3. [Python Instrumentation Layer](#3-python-instrumentation-layer)
4. [Telemetry Signals Reference](#4-telemetry-signals-reference)
5. [SigNoz Backend Deployment](#5-signoz-backend-deployment)
6. [SigNoz Dashboard Configuration](#6-signoz-dashboard-configuration)
7. [Signal Correlation Workflow](#7-signal-correlation-workflow)
8. [Running the Full Stack](#8-running-the-full-stack)
9. [Directory Structure](#9-directory-structure)
10. [Troubleshooting](#10-troubleshooting)
11. [Hackathon Judging Alignment](#11-hackathon-judging-alignment)
12. [References](#12-references)

---

## 1. System Architecture

```
+---------------------------------------------------------------+
|                     Unity ML-Agents                           |
|             3DBall Physics Simulation Engine                  |
|                                                               |
|  - Rigid-body ball + tilting platform                         |
|  - C# AgentScopeBalancer script (8 obs, 2 actions, rewards)  |
|  - gRPC communicator listening on port 5004                   |
+---------------------------+-----------------------------------+
                            |
                   gRPC  (port 5004)
                            v
+---------------------------------------------------------------+
|          Python Training Interface  (run_agent.py)            |
|                                                               |
|  +---------------+  +------------------+  +---------------+  |
|  | TracerProvider|  |  MeterProvider   |  | LoggerProvider|  |
|  | BatchSpan     |  | PeriodicExport   |  | BatchLog      |  |
|  | Processor     |  | (5 s interval)   |  | Processor     |  |
|  +-------+-------+  +--------+---------+  +-------+-------+  |
|          +-------------------+-------------------+           |
|                   OTLP / gRPC  (port 4317)                   |
+------------------------------+--------------------------------+
                               v
+---------------------------------------------------------------+
|                    SigNoz Backend                             |
|  OTel Collector → ClickHouse (traces, metrics, logs)          |
|  SigNoz Query Engine + Dashboard UI → localhost:3301          |
+---------------------------------------------------------------+
```

| Layer | Role |
|---|---|
| **Unity ML-Agents** | Physics simulation, observations, rewards, gRPC server on :5004 |
| **Python (run_agent.py)** | gRPC client, OTel SDK bootstrap, Episode DNA emission |
| **OTLP / gRPC** | Async transport for all three signal types on :4317 |
| **SigNoz + ClickHouse** | Ingest, index, query, and visualise all signals in one UI |

---

## 2. Unity 3DBall Environment Setup

### Prerequisites

| Tool | Version |
|---|---|
| Unity Hub + Editor | 2022.3 LTS+ |
| ML-Agents Package | `com.unity.ml-agents` 3.0+ |

### Scene Construction (Step by Step)

**1. Create the project** → Unity Hub → New Project → 3D template → name it `RL-Observatory`

**2. Install ML-Agents** → Window → Package Manager → + → Add by name → `com.unity.ml-agents`

**3. Create the Platform**
- Hierarchy → 3D Object → Cube → rename to `Platform`
- Transform: Position `(0, 0, 0)`, Scale `(5, 0.2, 5)`

**4. Create the Ball**
- Hierarchy → 3D Object → Sphere → rename to `Ball`
- Transform: Position `(0, 0.6, 0)`, Scale `(0.5, 0.5, 0.5)`
- Add Component → **Rigidbody** (Mass: 1, Use Gravity: ✅)

**5. Attach the agent script**
- Select `Platform` → Add Component → `AgentScopeBalancer`
- Drag `Ball` into the **ball** field in the Inspector

**6. Add Decision Requester**
- Select `Platform` → Add Component → **Decision Requester**
- Decision Period: `5`, Take Actions Between Decisions: ✅

**7. Configure Behavior Parameters**
- Select `Platform` → Add Component → **Behavior Parameters**

| Field | Value |
|---|---|
| Behavior Name | `Ball3DBrain` |
| Observation Space Size | `8` |
| Continuous Actions | `2` |
| Behavior Type | `Default` |

**8. Camera** → Main Camera → Position `(0, 8, -6)`, Rotation `(45, 0, 0)`

### Final Hierarchy

```
3DBall.unity
├── Main Camera
├── Directional Light
├── Platform       ← AgentScopeBalancer + BehaviorParameters + DecisionRequester
└── Ball           ← Sphere + Rigidbody
```

### Testing Manually

Set Behavior Type → **Heuristic Only** → Press Play → use **WASD** to tilt the platform. The ball should respond to gravity. Scene Gizmos show a yellow boundary box and a green line to the ball.

---

## 3. Python Instrumentation Layer

### OTel Bootstrap — The Key Insight

All three providers share one `Resource`. This is what enables cross-signal correlation in SigNoz:

```python
resource = Resource.create({
    "service.name": "unity-3dball-agent",
    "service.version": "1.0.0",
    "deployment.environment": "local",
})
# Injected into TracerProvider, MeterProvider, and LoggerProvider
```

### Episode DNA — Two Signals Per Episode

Every terminated episode emits both a **trace span** and a **log record**:

```python
# Trace span with structured attributes
with tracer.start_as_current_span("episode_dna") as span:
    span.set_attribute("episode.number", episode_count)
    span.set_attribute("episode.total_reward", current_reward)
    span.set_attribute("episode.steps", steps_in_episode)
    span.set_attribute("episode.result", outcome)  # Success | Collision | Timeout

# Structured log record
logger.info(
    f"[Episode DNA] #{episode_count} | Reward: {current_reward:.2f} | "
    f"Steps: {steps_in_episode} | Time: {duration:.1f}s | Result: {outcome}"
)
```

### CLI Usage

```bash
python run_agent.py                            # defaults (localhost:4317)
python run_agent.py --endpoint 10.0.0.5:4317   # remote SigNoz
python run_agent.py --port 5005 --tls          # custom Unity port + TLS
```

All flags have env var fallbacks (see `.env.example`). Run `python run_agent.py --help` for the full list.

---

## 4. Telemetry Signals Reference

### Metrics (7 instruments)

| Metric | Type | What It Tells You |
|---|---|---|
| `agent_steps_total` | Counter | Training throughput (query as rate → steps/sec) |
| `agent_step_reward` | Histogram | **P5 crossing zero = policy stabilisation** |
| `episode_duration_seconds` | Histogram | P95 growing = agent surviving longer |
| `episode_total_reward` | Histogram | Cumulative reward distribution per episode |
| `episode_success_total` | Counter | Ball balanced to max steps |
| `episode_collision_total` | Counter | Ball dropped off platform |
| `episode_timeout_total` | Counter | Ambiguous intermediate outcome |

### Traces (2 span types)

| Span | Emitted When | Key Attributes |
|---|---|---|
| `policy_inference` | Every action decision | `agent.count` |
| `episode_dna` | Episode terminates | `episode.number`, `episode.total_reward`, `episode.steps`, `episode.result`, `episode.avg_reward_per_step` |

### Logs

Every log record carries OTel resource attributes (`service.name`, etc.) and is searchable in SigNoz by body text + timestamp.

```
[Episode DNA] #247  | Reward: -0.73 | Steps: 14  | Time: 0.8s  | Result: Collision
[Episode DNA] #1023 | Reward: 98.30 | Steps: 983 | Time: 19.7s | Result: Timeout
```

---

## 5. SigNoz Backend Deployment

### Option A — Foundry (Recommended)

```bash
foundry cast apply casting.yaml
# casting.yaml.lock pins exact image digests for reproducibility
```

### Option B — Docker Compose

```bash
git clone https://github.com/SigNoz/signoz.git && cd signoz/deploy/
docker compose up -d
# Wait ~2 min for ClickHouse → UI at http://localhost:3301
```

### Verify Collector

```bash
docker compose logs otel-collector | grep "4317"
# Expected: msg="Listening on: 0.0.0.0:4317"
```

---

## 6. SigNoz Dashboard Configuration

Open `http://localhost:3301` → **Dashboards** → **New Dashboard**. Create six panels:

| # | Panel Type | Metric / Source | Aggregation | Purpose |
|---|---|---|---|---|
| 1 | Time Series | `agent_step_reward` | P95, P50, P5 | **Learning curve** — P5 rising = convergence |
| 2 | Value | `agent_steps_total` | Rate | Training throughput (steps/sec) |
| 3 | Bar Chart | `episode_success_total` + `episode_collision_total` | Rate | Success vs. failure ratio |
| 4 | Time Series | `episode_duration_seconds` | P95, Mean | Episode survival duration |
| 5 | Log Panel | Body contains `Episode DNA` | — | Structured episode records |
| 6 | Time Series | Traces → `policy_inference` | P99 Duration | Inference loop latency |

Save as: **RL Observatory — 3DBall Command Center**

---

## 7. Signal Correlation Workflow

This is the project's most powerful feature: **multi-signal temporal correlation**.

### Scenario — Diagnosing a Training Collapse

You notice at 14:32 UTC:
- `agent_step_reward` P5 drops from 0.04 to −1.0
- `episode_duration_seconds` P95 collapses from 9 s to 1.2 s

### Step 1 — Zoom the Dashboard

All six panels zoom to the 14:30–14:35 window simultaneously.

### Step 2 — Read the Episode DNA Log Stream

```
14:31:47  [Episode DNA] #891 | Reward: -0.73 | Steps: 12 | Result: Collision
14:31:52  [Episode DNA] #892 | Reward: -0.81 | Steps: 9  | Result: Collision
14:31:55  [Episode DNA] #893 | Reward: -0.94 | Steps: 7  | Result: Collision
```

Every episode dying in < 15 steps. Always `Collision`.

### Step 3 — Filter Traces

Traces → filter `episode.result = Collision` + 14:30–14:35 → confirms `episode.steps < 15` for all spans.

### Step 4 — Root Cause

Evidence points to: `spawnVelocity` accidentally increased in Unity Inspector, launching the ball off before the policy can react.

**Without RL Observatory**: hours of print-statement debugging.
**With RL Observatory**: 2 minutes of SigNoz query navigation.

---

## 8. Running the Full Stack

```bash
# 1. Start SigNoz
foundry cast apply casting.yaml      # wait ~2 min

# 2. Launch Unity
# Open 3DBall scene → Behavior Type = Default → Press Play

# 3. Start Python agent
pip install -r requirements.txt
python run_agent.py

# 4. Open SigNoz → http://localhost:3301

# Stopping:
# Python: Ctrl+C (session summary printed, telemetry flushed)
# Unity:  Press Play again
# SigNoz: foundry cast down casting.yaml
```

### What Appears in SigNoz (within 15–30 seconds)

| Signal | Where | What |
|---|---|---|
| Metrics | Panels 1–4 | Counters incrementing, reward histogram updating |
| Logs | Panel 5 | Episode DNA records after first episode ends |
| Traces | Panel 6 | `policy_inference` and `episode_dna` spans |

---

## 9. Directory Structure

```
rl-observatory/
├── .gitignore                        ← Python, Unity, IDE, secrets
├── .env.example                      ← All configurable parameters documented
├── LICENSE                           ← MIT
├── Makefile                          ← make install / make agent / make backend
├── README.md                         ← This file
├── blog.md                           ← Hackathon blog post
├── requirements.txt                  ← Pinned Python dependencies
│
├── run_agent.py                      ← Python OTel instrumentation interface
│                                       CLI, ASCII banner, colour-coded output
│                                       TelemetryKit with graceful shutdown
│
├── casting.yaml                      ← Foundry deployment manifest
├── casting.yaml.lock                 ← Pinned image digests
├── otel-collector-config.yaml        ← Collector pipeline (batching, memory limits)
│
└── Assets/ML-Agents/Examples/3DBall/Scripts/
    └── Ball3DAgent.cs                ← Unity C# agent (8 obs, 2 actions, rewards, gizmos)
```

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `run_agent.py` hangs on "Connecting..." | Unity must be in **Play Mode** with Behavior Type = **Default** |
| "No behaviors found!" | Stop Play, wait 2 s, press Play again while Python is already running |
| No metrics in SigNoz after 30 s | Check `docker compose ps` for OTel Collector; verify port 4317; wait for 5 s flush |
| Ball falls instantly every episode | Reduce Spawn Velocity to 0.1, Spawn Radius to 0.5 in Inspector |
| SigNoz UI unavailable | ClickHouse init takes 2–5 min first run — check `docker compose logs clickhouse` |

---

## 11. Hackathon Judging Alignment

| Criterion | How RL Observatory Addresses It |
|---|---|
| **Potential Impact** | RL debugging is a universal pain point. Silent policy collapse is the hardest class of ML bug. Episode DNA + histogram reward curves + multi-signal correlation are immediately applicable to any RL environment. |
| **Creativity & Innovation** | Applying SRE-grade observability to an RL training loop is genuinely novel. "Episode DNA" — treating each terminated episode as a queryable diagnostic artifact — is a new primitive for RL debugging. |
| **Technical Excellence** | Three OTel signal types from scratch. Async export pipeline. Foundry-based deployment. Clean four-layer architecture. Full C# ML-Agents implementation. CLI with env var fallbacks. |
| **Best Use of SigNoz** | All three signals via native OTLP/gRPC. Cross-signal correlation workflow. Six-panel dashboard. ClickHouse for high-cardinality episode attribute filtering. |
| **User Experience** | One-command deploy. ASCII banner + colour-coded terminal. Step-by-step Unity guide. Heuristic mode for manual testing. `.env.example` for configuration. |
| **Presentation Quality** | Architecture diagram. Signal correlation diagnostic scenario. Troubleshooting table. Blog post with real debugging story. |

---

## 12. References

| Resource | URL |
|---|---|
| SigNoz Docs | https://signoz.io/docs |
| SigNoz GitHub | https://github.com/SigNoz/signoz |
| OpenTelemetry Python SDK | https://opentelemetry-python.readthedocs.io |
| OTLP Specification | https://opentelemetry.io/docs/specs/otlp |
| Unity ML-Agents | https://github.com/Unity-Technologies/ml-agents |
| ClickHouse Docs | https://clickhouse.com/docs |

---

<div align="center">

**Built with ❤️ for the Agents of SigNoz Hackathon 2026**

*Track 01 — AI & Agent Observability*

> *"RL Observatory doesn't just observe an agent. It gives you a window into the mind of one."*

</div>

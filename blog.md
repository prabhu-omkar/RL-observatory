# I Debugged a Reinforcement Learning Agent with SigNoz — Here's What `print()` Never Told Me

**TL;DR:** I instrumented a Unity ML-Agents training loop with OpenTelemetry and shipped traces, metrics, and logs into SigNoz. What I got back was something `print(f"reward={reward}")` could never give me — a time-correlated, queryable diagnostic record of every episode my agent ever lived and died through. This post walks through exactly how I did it, what I learned, and the one SigNoz feature that changed how I think about AI debugging.

---

## The Problem That Started This

I was training a reinforcement learning agent to balance a ball on a tilting platform inside Unity (the classic [ML-Agents 3DBall task](https://github.com/Unity-Technologies/ml-agents)). The setup is simple: a neural network controls how the platform tilts, and the ball either stays balanced or rolls off the edge.

Training was going fine — reward was climbing, episodes were getting longer — and then it wasn't. Around episode 800, the reward graph cratered. The agent went from balancing the ball for 10+ seconds to dropping it in under a second. Every single episode.

My debugging toolkit at this point? This:

```python
print(f"Episode {ep}: reward={reward:.3f}")
```

Scrolling through hundreds of terminal lines, I could see the numbers were bad. But I couldn't answer the questions that mattered:

- **When exactly** did the collapse start?
- **Was it sudden or gradual?** (A distribution shift, or a cliff?)
- **What was the agent doing differently** in the failing episodes vs. the succeeding ones?
- Did the episode **duration** collapse at the same time as the reward, or before it?

Print statements can't answer any of these. They show you a value. They don't show you a distribution, a correlation, or a timeline.

That's when I thought: what if I treat this training loop like a production microservice? What if I instrument it with OpenTelemetry and ship the signals into an actual observability backend?

---

## The Setup: OTel SDK → OTLP/gRPC → SigNoz

The architecture is four layers:

1. **Unity** runs the physics simulation (C# `AgentScopeBalancer` script)
2. **Python** (`run_agent.py`) connects via gRPC, steps the environment, and wraps every action in OTel telemetry
3. **OTLP/gRPC** transports traces, metrics, and logs to port 4317
4. **SigNoz** (self-hosted via Docker) ingests, indexes, and visualises everything

I self-hosted SigNoz using their Docker Compose setup:

```bash
git clone https://github.com/SigNoz/signoz.git
cd signoz/deploy/
docker compose up -d
```

Two minutes later, the OTel Collector was listening on `localhost:4317` and the SigNoz UI was live at `localhost:3301`. No account, no cloud, no API keys. That frictionlessness matters when you're iterating fast.

---

## The Instrumentation: Three Signals, One Resource

Here's the part where I learned something the docs don't emphasise enough: **the power of shared resource identity**.

Every OTel signal — traces, metrics, logs — carries a `Resource` object. If all three providers share the same resource, SigNoz can correlate them across time. This is the entire trick:

```python
resource = Resource.create({
    "service.name": "unity-3dball-agent",
    "service.version": "1.0.0",
    "deployment.environment": "local",
})
```

I injected this into the `TracerProvider`, `MeterProvider`, and `LoggerProvider`. From that point on, every signal was automatically tagged with the same service identity. This is what makes the correlation workflow possible later.

### Metrics I Registered

```python
step_counter     = meter.create_counter("agent_steps_total")
reward_histogram = meter.create_histogram("agent_step_reward")
episode_duration = meter.create_histogram("episode_duration_seconds")
episode_reward   = meter.create_histogram("episode_total_reward")
success_counter  = meter.create_counter("episode_success_total")
collision_counter = meter.create_counter("episode_collision_total")
```

The key insight: **use a histogram for reward, not just a counter**. A counter gives you a total. A histogram gives you a *distribution* — P5, P50, P95. When your agent is learning, the P5 of reward gradually lifts from −1.0 toward zero. When it collapses, the P5 drops back down. This is invisible with a counter or a moving average.

### The "Episode DNA" Concept

Every time an episode ends — ball drops, or max steps reached — I emit two things:

**A trace span** with structured attributes:

```python
with tracer.start_as_current_span("episode_dna") as span:
    span.set_attribute("episode.number", episode_count)
    span.set_attribute("episode.total_reward", current_reward)
    span.set_attribute("episode.steps", steps_in_episode)
    span.set_attribute("episode.result", outcome)  # "Success" | "Collision" | "Timeout"
```

**A structured log record:**

```python
logger.info(
    f"[Episode DNA] #{episode_count} | Reward: {current_reward:.2f} | "
    f"Steps: {steps_in_episode} | Time: {duration:.1f}s | Result: {outcome}"
)
```

I call this "Episode DNA" because it's the full genetic fingerprint of that episode — everything you need to diagnose what happened, queryable in SigNoz by any attribute.

---

## What SigNoz Showed Me (That print() Never Could)

### The Learning Curve Is a Histogram, Not a Line

In SigNoz's Metrics Explorer, I queried `agent_step_reward` with P95, P50, and P5 aggregations. What I saw was not a single line going up — it was three lines spreading apart and then converging.

Early training: P95 ≈ 0.1 (the living reward), P5 ≈ −1.0 (the terminal penalty). The agent is either surviving a step or dying. Bimodal.

After 400 episodes: P5 starts lifting. Fewer −1.0 events. The distribution is tightening around +0.1. **The agent is learning to not die.**

This is something a `print(reward)` statement literally cannot show you. You'd need to collect thousands of values, bucket them into a histogram, and plot it yourself. OTel histograms + SigNoz do this automatically.

### The Collapse Was Instant, Not Gradual

When the training collapsed at episode ~800, the P5 dropped from 0.04 back to −1.0 in under 30 seconds. The P95 stayed at 0.1 (because even dying agents get one step of living reward). The spread between P95 and P5 *widened* — the inverse of convergence.

This told me the collapse wasn't gradual degradation. It was a hard transition. Something changed at that moment.

### Cross-Signal Correlation Found the Cause

This is the feature that made this whole project worth building.

I zoomed the SigNoz dashboard to the 30-second window where P5 collapsed. Then I switched to the Logs tab and filtered for `body CONTAINS "Episode DNA"`. Because the logs carry the same `service.name` resource attribute as the metrics, they automatically aligned to the same time range.

What I saw:

```
14:31:47  [Episode DNA] #891 | Reward: -0.73 | Steps: 12 | Result: Collision
14:31:52  [Episode DNA] #892 | Reward: -0.81 | Steps: 9  | Result: Collision
14:31:55  [Episode DNA] #893 | Reward: -0.94 | Steps: 7  | Result: Collision
```

Every episode was dying in under 15 steps with `Collision`. I then went to the Traces tab, filtered for `episode_dna` spans where `episode.result = Collision`, and confirmed: `episode.steps < 15` for every single one in the window.

The root cause? I had accidentally bumped the `spawnVelocity` parameter in my Unity Inspector while adjusting the camera. The ball was spawning with enough velocity to fly off the platform before the agent could react. A one-click undo in Unity fixed it.

**Time to diagnosis: about 2 minutes.** Without SigNoz, I would have spent an hour adding more print statements, re-running training, and staring at terminal output trying to correlate timestamps manually.

---

## What I Got Wrong Along the Way

**The C# script took multiple attempts.** My first version used euler angles for observations (gimbal lock risk), `localPosition` instead of world-space subtraction (breaks if the ball isn't parented), and multiplied rotation by `Time.fixedDeltaTime` which over-dampened the tilt so much the agent could barely move the platform. The working version uses raw quaternion components, world-space position subtraction, and direct rotation — much simpler, much more stable.

**I initially forgot `time.sleep(0.02)` in the Python loop.** Without it, the loop spins at full CPU speed and the OTel `BatchSpanProcessor` never gets a chance to flush. Spans were being created but silently dropped because the export buffer was always full. Adding the sleep fixed it instantly — but it took me 20 minutes to figure out why SigNoz was showing traces for the first 5 seconds and then nothing.

**The `PeriodicExportingMetricReader` default interval matters.** I initially left it at the default (60 seconds). For an RL training loop that runs 50 steps/second, that means your first metric data point arrives a full minute after training starts. I dropped it to 5 seconds (`export_interval_millis=5000`) and the dashboard came alive.

---

## The Takeaway

The mental model shift is this: **a reinforcement learning training loop is a distributed system**. It has a producer (the simulator), a consumer (the policy), a communication protocol (gRPC), and a time-series of events (episodes). The tools SRE teams use to debug production services — traces, metrics, structured logs — apply directly.

SigNoz was the right backend for this because it's OpenTelemetry-native. I didn't need an adapter, a proprietary SDK, or a custom exporter. The Python OTel SDK's `OTLPSpanExporter`, `OTLPMetricExporter`, and `OTLPLogExporter` all point at `localhost:4317` and it just works. Three signals, one endpoint, one query interface.

If you train RL agents and you've ever stared at a terminal full of reward numbers wondering *why* your agent suddenly got worse — try this. Instrument the loop, ship to SigNoz, and look at the histogram. The answer is usually in the P5.

---

*The full project is open-source: [github.com/prabhu-omkar/RL-observatory](https://github.com/prabhu-omkar/RL-observatory)*

*Built for the [Agents of SigNoz Hackathon 2026](https://wemakedevs.org) — Track 01: AI & Agent Observability.*
        
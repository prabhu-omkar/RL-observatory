#!/usr/bin/env python3
"""
RL Observatory — Observability Interface for Unity ML-Agents
─────────────────────────────────────────────────────────────────────────────
Bridges a Unity 3DBall simulation with SigNoz via native OpenTelemetry
instrumentation, emitting three signal types (traces, metrics, structured
logs) over OTLP/gRPC for real-time training diagnostics.

Usage:
    python run_agent.py                          # defaults
    python run_agent.py --endpoint 10.0.0.5:4317 # remote collector
    python run_agent.py --port 5005 --delay 0.01 # custom Unity port

Author:  RL Observatory Contributors
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

# ── OpenTelemetry: Traces ────────────────────────────────────────────────────
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# ── OpenTelemetry: Logs ──────────────────────────────────────────────────────
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & BANNER
# ═════════════════════════════════════════════════════════════════════════════

VERSION = "1.0.0"

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║   ██████╗ ██╗          ██████╗ ██████╗ ███████╗███████╗██████╗ ██╗   ██╗   ║
║   ██╔══██╗██║         ██╔═══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██║   ██║   ║
║   ██████╔╝██║         ██║   ██║██████╔╝███████╗█████╗  ██████╔╝██║   ██║   ║
║   ██╔══██╗██║         ██║   ██║██╔══██╗╚════██║██╔══╝  ██╔══██╗╚██╗ ██╔╝   ║
║   ██║  ██║███████╗    ╚██████╔╝██████╔╝███████║███████╗██║  ██║ ╚████╔╝    ║
║   ╚═╝  ╚═╝╚══════╝     ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝  ╚═══╝     ║
║                                                                            ║
║           Enterprise-Grade Observability for RL Agents   v{ver}           ║
║                                                                            ║
║   Signals ──► SigNoz via OTLP/gRPC                                        ║
║   ├── Traces   : policy_inference, episode_dna                             ║
║   ├── Metrics  : agent_steps_total, agent_step_reward, episode_duration    ║
║   └── Logs     : Episode DNA structured records                            ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".format(ver=VERSION)

# ANSI colour codes for terminal output
class _C:
    """Terminal colour helpers (ANSI escape sequences)."""
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Runtime configuration, populated from CLI args + environment vars."""
    # OTel
    otel_endpoint: str = "127.0.0.1:4317"
    otel_insecure: bool = True
    service_name: str = "unity-3dball-agent"
    environment: str = "local"
    metrics_interval_ms: int = 5000

    # Unity
    unity_port: int = 5004
    unity_worker_id: int = 0

    # Simulation
    step_delay: float = 0.02

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Merge CLI args with environment variable fallbacks."""
        return cls(
            otel_endpoint=args.endpoint or os.getenv("OTEL_EXPORTER_ENDPOINT", "127.0.0.1:4317"),
            otel_insecure=not args.tls,
            service_name=os.getenv("OTEL_SERVICE_NAME", "unity-3dball-agent"),
            environment=os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "local"),
            metrics_interval_ms=int(os.getenv("METRICS_EXPORT_INTERVAL_MS", "5000")),
            unity_port=args.port or int(os.getenv("UNITY_BASE_PORT", "5004")),
            unity_worker_id=args.worker or int(os.getenv("UNITY_WORKER_ID", "0")),
            step_delay=args.delay or float(os.getenv("STEP_DELAY_SECONDS", "0.02")),
        )


# ═════════════════════════════════════════════════════════════════════════════
#  TELEMETRY BOOTSTRAP
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TelemetryKit:
    """Bundles every OTel provider, exporter and instrument into one object.

    Calling `shutdown()` flushes all pending telemetry and closes exporters
    cleanly — critical to avoid losing the final batch of episode data.
    """
    tracer: trace.Tracer = field(default=None)
    tracer_provider: Optional[TracerProvider] = None
    meter_provider: Optional[MeterProvider] = None
    logger_provider: Optional[LoggerProvider] = None
    logger: Optional[logging.Logger] = None

    # Metric instruments
    step_counter: Optional[metrics.Counter] = None
    reward_histogram: Optional[metrics.Histogram] = None
    episode_duration: Optional[metrics.Histogram] = None
    episode_reward: Optional[metrics.Histogram] = None
    success_counter: Optional[metrics.Counter] = None
    collision_counter: Optional[metrics.Counter] = None
    timeout_counter: Optional[metrics.Counter] = None

    def shutdown(self) -> None:
        """Flush all pending telemetry and close exporters."""
        if self.tracer_provider:
            self.tracer_provider.force_flush()
            self.tracer_provider.shutdown()
        if self.meter_provider:
            self.meter_provider.shutdown()
        if self.logger_provider:
            self.logger_provider.force_flush()
            self.logger_provider.shutdown()


def bootstrap_telemetry(cfg: Config) -> TelemetryKit:
    """Initialise the full OTel signal trio and return a TelemetryKit."""

    kit = TelemetryKit()

    resource = Resource.create({
        "service.name": cfg.service_name,
        "service.version": VERSION,
        "deployment.environment": cfg.environment,
        "ml.framework": "unity-mlagents",
        "ml.task": "3dball-balancing",
    })

    # ── Traces ────────────────────────────────────────────────────────────
    kit.tracer_provider = TracerProvider(resource=resource)
    kit.tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=cfg.otel_endpoint, insecure=cfg.otel_insecure)
        )
    )
    trace.set_tracer_provider(kit.tracer_provider)
    kit.tracer = trace.get_tracer("rl.observatory.tracer", VERSION)

    # ── Metrics ───────────────────────────────────────────────────────────
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=cfg.otel_endpoint, insecure=cfg.otel_insecure),
        export_interval_millis=cfg.metrics_interval_ms,
    )
    kit.meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(kit.meter_provider)
    meter = metrics.get_meter("rl.observatory.meter", VERSION)

    # Register instruments
    kit.step_counter = meter.create_counter(
        "agent_steps_total",
        description="Monotonic count of environment steps executed by the agent",
        unit="steps",
    )
    kit.reward_histogram = meter.create_histogram(
        "agent_step_reward",
        description="Distribution of scalar rewards received at each step",
        unit="reward",
    )
    kit.episode_duration = meter.create_histogram(
        "episode_duration_seconds",
        description="Wall-clock duration of each completed episode",
        unit="s",
    )
    kit.episode_reward = meter.create_histogram(
        "episode_total_reward",
        description="Cumulative reward earned per episode",
        unit="reward",
    )
    kit.success_counter = meter.create_counter(
        "episode_success_total",
        description="Episodes terminated by max-step timeout (ball kept balanced)",
        unit="episodes",
    )
    kit.collision_counter = meter.create_counter(
        "episode_collision_total",
        description="Episodes terminated by ball falling off the platform",
        unit="episodes",
    )
    kit.timeout_counter = meter.create_counter(
        "episode_timeout_total",
        description="Episodes ended with ambiguous intermediate reward",
        unit="episodes",
    )

    # ── Logs ──────────────────────────────────────────────────────────────
    kit.logger_provider = LoggerProvider(resource=resource)
    kit.logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(endpoint=cfg.otel_endpoint, insecure=cfg.otel_insecure)
        )
    )
    set_logger_provider(kit.logger_provider)

    otel_handler = LoggingHandler(level=logging.NOTSET, logger_provider=kit.logger_provider)
    kit.logger = logging.getLogger("rl.observatory")
    kit.logger.addHandler(otel_handler)
    kit.logger.setLevel(logging.INFO)

    # Also log to stderr with a clean format for local visibility.
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(
        f"{_C.DIM}%(asctime)s{_C.RESET} │ %(message)s",
        datefmt="%H:%M:%S",
    ))
    kit.logger.addHandler(console)

    return kit


# ═════════════════════════════════════════════════════════════════════════════
#  EPISODE OUTCOME CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════

def classify_outcome(final_reward: float) -> str:
    """Determine the semantic outcome of an episode from its terminal reward.

    Returns one of:
        'Success'   — final reward >= 1.0  (ball survived to max steps)
        'Collision' — final reward <= -0.5 (ball dropped off platform)
        'Timeout'   — intermediate value   (ambiguous / partial episode)
    """
    if final_reward >= 1.0:
        return "Success"
    if final_reward <= -0.5:
        return "Collision"
    return "Timeout"


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN OBSERVABILITY LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_observability_loop(cfg: Config, kit: TelemetryKit) -> None:
    """Core training loop: step the Unity env, emit telemetry on every cycle."""

    log = kit.logger

    log.info(f"{_C.CYAN}Connecting to Unity on port {cfg.unity_port}...{_C.RESET}")
    env = UnityEnvironment(
        file_name=None,
        base_port=cfg.unity_port,
        worker_id=cfg.unity_worker_id,
    )
    env.reset()

    behavior_names = list(env.behavior_specs.keys())
    if not behavior_names:
        log.error(f"{_C.RED}No behaviors found! Is Unity in Play mode?{_C.RESET}")
        env.close()
        return

    behavior_name = behavior_names[0]
    spec = env.behavior_specs[behavior_name]

    log.info(f"{_C.GREEN}✓ Connected{_C.RESET} — Behavior: {_C.BOLD}{behavior_name}{_C.RESET}")
    log.info(f"{_C.GREEN}✓ Flight Recorder active{_C.RESET} — streaming to {_C.BOLD}{cfg.otel_endpoint}{_C.RESET}")
    log.info(f"{_C.DIM}  Obs size: {spec.observation_specs[0].shape}  |  "
             f"Action size: {spec.action_spec.continuous_size}  |  "
             f"Export interval: {cfg.metrics_interval_ms} ms{_C.RESET}")
    log.info("")

    # ── Episode state ─────────────────────────────────────────────────────
    episode_count      = 0
    total_steps        = 0
    current_reward     = 0.0
    steps_in_episode   = 0
    episode_start_time = time.time()
    session_start_time = time.time()

    try:
        while True:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # ── 1. Handle episode termination ─────────────────────────────
            if len(terminal_steps) > 0:
                episode_count += 1
                duration = time.time() - episode_start_time

                final_reward = float(np.sum(terminal_steps.reward))
                current_reward += final_reward

                outcome = classify_outcome(final_reward)

                # Record metrics
                kit.episode_duration.record(duration)
                kit.episode_reward.record(current_reward)

                if outcome == "Success":
                    kit.success_counter.add(1)
                elif outcome == "Collision":
                    kit.collision_counter.add(1)
                else:
                    kit.timeout_counter.add(1)

                # Emit Episode DNA trace span
                with kit.tracer.start_as_current_span("episode_dna") as span:
                    span.set_attribute("episode.number", episode_count)
                    span.set_attribute("episode.total_reward", round(current_reward, 4))
                    span.set_attribute("episode.duration", round(duration, 4))
                    span.set_attribute("episode.steps", steps_in_episode)
                    span.set_attribute("episode.result", outcome)
                    span.set_attribute("episode.avg_reward_per_step",
                                       round(current_reward / max(steps_in_episode, 1), 6))

                # Colour-coded Episode DNA log
                colour = (
                    _C.GREEN if outcome == "Success" else
                    _C.RED   if outcome == "Collision" else
                    _C.YELLOW
                )
                log.info(
                    f"{_C.BOLD}[Episode DNA]{_C.RESET}  "
                    f"#{episode_count:<5}  "
                    f"Reward: {colour}{current_reward:>8.2f}{_C.RESET}  │  "
                    f"Steps: {steps_in_episode:>5}  │  "
                    f"Time: {duration:>6.1f}s  │  "
                    f"Result: {colour}{outcome}{_C.RESET}"
                )

                # Reset for next episode
                current_reward = 0.0
                steps_in_episode = 0
                episode_start_time = time.time()

            # ── 2. Handle decision requests ───────────────────────────────
            num_agents = len(decision_steps)
            if num_agents > 0:
                steps_in_episode += 1
                total_steps += 1

                with kit.tracer.start_as_current_span("policy_inference") as span:
                    continuous_actions = np.random.uniform(
                        -1.0, 1.0,
                        size=(num_agents, spec.action_spec.continuous_size),
                    )
                    action_tuple = ActionTuple(continuous=continuous_actions)
                    env.set_actions(behavior_name, action_tuple)
                    span.set_attribute("agent.count", num_agents)

                step_reward = float(np.sum(decision_steps.reward))
                current_reward += step_reward
                kit.step_counter.add(1)
                kit.reward_histogram.record(step_reward)

            env.step()
            time.sleep(cfg.step_delay)

    except KeyboardInterrupt:
        session_duration = time.time() - session_start_time
        log.info("")
        log.info(f"{_C.YELLOW}■ Flight Recorder stopped{_C.RESET}")
        log.info(f"  Episodes: {episode_count}  │  Total steps: {total_steps}  │  "
                 f"Session: {session_duration:.1f}s")
        log.info(f"  Flushing telemetry to {cfg.otel_endpoint}...")

    finally:
        env.close()
        kit.shutdown()
        log.info(f"{_C.GREEN}✓ Shutdown complete.{_C.RESET}")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="run_agent",
        description="RL Observatory — OTel instrumentation bridge for Unity ML-Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_agent.py                            # defaults (localhost:4317)
  python run_agent.py --endpoint 10.0.0.5:4317   # remote SigNoz collector
  python run_agent.py --port 5005 --worker 1     # second Unity instance
  python run_agent.py --tls                      # secure gRPC channel
        """,
    )
    p.add_argument("--endpoint", type=str, default=None,
                   help="OTel Collector gRPC endpoint (default: 127.0.0.1:4317)")
    p.add_argument("--port", type=int, default=None,
                   help="Unity gRPC base port (default: 5004)")
    p.add_argument("--worker", type=int, default=None,
                   help="Unity worker ID — unique per concurrent agent (default: 0)")
    p.add_argument("--delay", type=float, default=None,
                   help="Inter-step sleep in seconds (default: 0.02)")
    p.add_argument("--tls", action="store_true",
                   help="Use TLS for the OTLP gRPC channel")
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return p


def main() -> None:
    """Parse arguments, print banner, bootstrap telemetry, run loop."""
    args = build_parser().parse_args()
    cfg = Config.from_args(args)

    # Enable ANSI colours on Windows
    if sys.platform == "win32":
        os.system("")  # triggers ANSI escape processing in cmd/powershell

    print(f"{_C.CYAN}{BANNER}{_C.RESET}")

    print(f"  {_C.DIM}Configuration{_C.RESET}")
    print(f"  ├── OTel Endpoint   : {_C.BOLD}{cfg.otel_endpoint}{_C.RESET}")
    print(f"  ├── Service Name    : {cfg.service_name}")
    print(f"  ├── Unity Port      : {cfg.unity_port}")
    print(f"  ├── Step Delay      : {cfg.step_delay}s")
    print(f"  └── TLS             : {'enabled' if not cfg.otel_insecure else 'disabled'}")
    print()

    kit = bootstrap_telemetry(cfg)
    run_observability_loop(cfg, kit)


if __name__ == "__main__":
    main()
             
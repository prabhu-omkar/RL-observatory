import time
import logging
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# OpenTelemetry Imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource

# OpenTelemetry Log Imports
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

# 1. OpenTelemetry Configuration
resource = Resource.create({"service.name": "unity-3dball-agent"})

# Traces Setup
span_exporter = OTLPSpanExporter(endpoint="127.0.0.1:4317", insecure=True)
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("unity.agent.tracer")

# Metrics Setup
metric_exporter = OTLPMetricExporter(endpoint="127.0.0.1:4317", insecure=True)
reader = PeriodicExportingMetricReader(metric_exporter)
meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter("unity.agent.meter")

# Logs Setup
logger_provider = LoggerProvider(resource=resource)
set_logger_provider(logger_provider)
log_exporter = OTLPLogExporter(endpoint="127.0.0.1:4317", insecure=True)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
otel_handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
logging.getLogger().addHandler(otel_handler)
logger = logging.getLogger("unity.agent.logger")
logger.setLevel(logging.INFO)

# Metrics definitions
step_counter = meter.create_counter("agent_steps_total", description="Total steps taken by agent")
reward_histogram = meter.create_histogram("agent_step_reward", description="Rewards per step")
episode_duration = meter.create_histogram("episode_duration_seconds", description="Time taken per episode")
success_counter = meter.create_counter("episode_success_total", description="Count of goal reached")
collision_counter = meter.create_counter("episode_collision_total", description="Count of obstacle collisions")

def run_observability_loop():
    logger.info("Connecting to Unity Environment...")
    env = UnityEnvironment(file_name=None, base_port=5004, worker_id=0)
    env.reset()

    behavior_names = list(env.behavior_specs.keys())
    if not behavior_names:
        logger.error("No behaviors found! Ensure Unity is running in Play mode.")
        return
    
    behavior_name = behavior_names[0]
    spec = env.behavior_specs[behavior_name]
    logger.info(f"Connected! Behavior Name: {behavior_name}")
    logger.info("AgentScope Flight Recorder Active. Press PLAY in Unity...")

    episode_count = 0
    current_reward = 0.0
    steps_in_episode = 0
    episode_start_time = time.time()

    try:
        while True:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # 1. Record completed episode metrics & emit Episode DNA trace/log
            if len(terminal_steps) > 0:
                episode_count += 1
                duration = time.time() - episode_start_time
                episode_duration.record(duration)
                
                final_reward = float(np.sum(terminal_steps.reward))
                current_reward += final_reward
                
                # Evaluate episode outcome based on final step reward
                outcome = "Success" if final_reward >= 1.0 else ("Collision" if final_reward <= -0.5 else "Timeout")
                
                if outcome == "Success":
                    success_counter.add(1)
                elif outcome == "Collision":
                    collision_counter.add(1)

                # Export Episode DNA trace span
                with tracer.start_as_current_span("episode_dna") as dna_span:
                    dna_span.set_attribute("episode.number", episode_count)
                    dna_span.set_attribute("episode.total_reward", current_reward)
                    dna_span.set_attribute("episode.duration", duration)
                    dna_span.set_attribute("episode.steps", steps_in_episode)
                    dna_span.set_attribute("episode.result", outcome)

                # Emit Episode DNA Log directly to SigNoz
                logger.info(f"[Episode DNA] #{episode_count} | Reward: {current_reward:.2f} | Steps: {steps_in_episode} | Time: {duration:.1f}s | Result: {outcome}")

                # Reset state for next episode
                current_reward = 0.0
                steps_in_episode = 0
                episode_start_time = time.time()

            # 2. Step agents requiring decision
            num_agents = len(decision_steps)
            if num_agents > 0:
                steps_in_episode += 1

                with tracer.start_as_current_span("policy_inference"):
                    continuous_actions = np.random.uniform(-1.0, 1.0, size=(num_agents, spec.action_spec.continuous_size))
                    action_tuple = ActionTuple(continuous=continuous_actions)
                    env.set_actions(behavior_name, action_tuple)

                step_reward = float(np.sum(decision_steps.reward))
                current_reward += step_reward
                step_counter.add(1)
                reward_histogram.record(step_reward)

            env.step()
            time.sleep(0.02)

    except KeyboardInterrupt:
        logger.info("Stopping AgentScope Flight Recorder...")
    finally:
        env.close()

if __name__ == "__main__":
    run_observability_loop()

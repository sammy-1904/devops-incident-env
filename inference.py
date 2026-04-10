"""
inference.py — Baseline inference script for the DevOps Incident environment.

MANDATORY for submission. Follows the exact stdout format required by the hackathon:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables (set in .env or system):
    HF_TOKEN          Your Hugging Face token (used as API key)
    API_BASE_URL      LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME        Model to use (default: Qwen/Qwen2.5-7B-Instruct)
    LOCAL_IMAGE_NAME  Docker image name (optional — only needed when using from_docker_image())

Usage:
    # Build and start the environment:
    docker build -t devops-incident-env .
    docker run -d -p 8000:8000 --name devops-incident-env devops-incident-env

    # Run inference:
    python inference.py
"""

import subprocess
import sys

# Auto-install required packages if missing (validator runs in a fresh environment)
def _ensure_packages():
    required = [
        ("openai",       "openai>=1.0.0"),
        ("dotenv",       "python-dotenv"),
        ("openenv_core", "openenv-core"),
    ]
    for import_name, install_name in required:
        try:
            __import__(import_name)
        except ImportError:
            print(f"[SETUP] Installing {install_name}...", flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", install_name, "-q"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"[SETUP] pip install {install_name} failed:\n{result.stderr}", flush=True)
                raise RuntimeError(f"Could not install required package: {install_name}")
            print(f"[SETUP] Installed {install_name}", flush=True)

_ensure_packages()

import asyncio
import json
import os
import time
import textwrap
from pathlib import Path
from typing import List, Optional

# Windows fix: stdout defaults to cp1252 which can't encode Unicode dashboard chars (✓✗)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load .env file — token stays out of code
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from openai import OpenAI
from client import IncidentEnv, IncidentAction

# ---------------------------------------------------------------------------
# Configuration — all from environment variables
# ---------------------------------------------------------------------------

API_KEY          = os.environ.get("HF_TOKEN", "")
API_BASE_URL     = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")   # no default — set when using from_docker_image()
# ENV_BASE_URL: connect directly to a running server (HF Space or local uvicorn).
# Overrides LOCAL_IMAGE_NAME when provided.
# Example: ENV_BASE_URL=https://sammy-1904-devops-incident-env.hf.space
ENV_BASE_URL     = os.environ.get("ENV_BASE_URL", "")

TASK_NAME  = "devops-incident"
BENCHMARK  = "devops-incident-env"
MAX_STEPS  = 22          # worst-case max (Scenario 5)
MAX_REWARD = 70.0        # theoretical max per episode (used for score normalisation)

# Scenarios to evaluate (at least 2 required by checklist)
SCENARIOS_TO_RUN = [1, 2, 3, 4, 5]

# ---------------------------------------------------------------------------
# Required stdout helpers  (exact format from sample_inference.py)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Tool definitions — LLM uses function-calling to take actions
# ---------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {
        "name": "query_logs",
        "description": "Retrieve recent log entries for a service. Look for errors, stack traces, OOM kills.",
        "parameters": {"type": "object", "properties": {
            "service":   {"type": "string", "description": "Service name to query"},
            "reasoning": {"type": "string", "description": "Why are you checking this service?"},
        }, "required": ["service", "reasoning"]},
    }},
    {"type": "function", "function": {
        "name": "check_metrics",
        "description": "Check CPU, memory, error rate, and latency for a service.",
        "parameters": {"type": "object", "properties": {
            "service":   {"type": "string"},
            "reasoning": {"type": "string"},
        }, "required": ["service", "reasoning"]},
    }},
    {"type": "function", "function": {
        "name": "restart_service",
        "description": "Restart a service. Use when it has crashed or been OOMKilled.",
        "parameters": {"type": "object", "properties": {
            "service":   {"type": "string"},
            "reasoning": {"type": "string"},
        }, "required": ["service", "reasoning"]},
    }},
    {"type": "function", "function": {
        "name": "rollback_deployment",
        "description": "Roll back the last deployment. Use when a recent deploy caused errors.",
        "parameters": {"type": "object", "properties": {
            "service":   {"type": "string"},
            "reasoning": {"type": "string"},
        }, "required": ["service", "reasoning"]},
    }},
    {"type": "function", "function": {
        "name": "rollback_config",
        "description": "Roll back the last configuration change for a service.",
        "parameters": {"type": "object", "properties": {
            "service":   {"type": "string"},
            "reasoning": {"type": "string"},
        }, "required": ["service", "reasoning"]},
    }},
    {"type": "function", "function": {
        "name": "enable_circuit_breaker",
        "description": "Enable circuit breaker on a service to prevent cascade failures.",
        "parameters": {"type": "object", "properties": {
            "service":   {"type": "string"},
            "reasoning": {"type": "string"},
        }, "required": ["service", "reasoning"]},
    }},
    {"type": "function", "function": {
        "name": "enable_rate_limiting",
        "description": "Enable rate limiting to throttle excessive traffic (e.g. DDoS).",
        "parameters": {"type": "object", "properties": {
            "service":   {"type": "string"},
            "reasoning": {"type": "string"},
        }, "required": ["service", "reasoning"]},
    }},
    {"type": "function", "function": {
        "name": "diagnose",
        "description": "State your root cause hypothesis once you are confident.",
        "parameters": {"type": "object", "properties": {
            "root_cause": {"type": "string", "description": "Clear description of the root cause"},
            "reasoning":  {"type": "string"},
        }, "required": ["root_cause", "reasoning"]},
    }},
    {"type": "function", "function": {
        "name": "write_postmortem",
        "description": "Submit a post-mortem and close the incident. Only after incident is resolved.",
        "parameters": {"type": "object", "properties": {
            "report": {"type": "string"},
        }, "required": ["report"]},
    }},
    {"type": "function", "function": {
        "name": "escalate",
        "description": "Escalate to a human on-call. Only if you cannot resolve it yourself.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"},
        }, "required": ["reason"]},
    }},
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior Site Reliability Engineer (SRE) responding to a production incident.
    Your job is to:
      1. Investigate using query_logs and check_metrics
      2. Identify the root cause with diagnose()
      3. Apply the correct fix (restart_service, rollback_deployment, etc.)
      4. Write a post-mortem once resolved

    Rules:
    - Always use the 'reasoning' field to explain your thinking.
    - Start by querying logs for the service mentioned in the alert.
    - Do NOT restart random services — investigate first.
    - Call diagnose() once you know the root cause, then fix it.
    - Do NOT escalate unless all options are exhausted.

    Fix selection guide:
    - Service is "down" with OOM / memory errors → restart_service
    - Service is "down" after a recent deployment → rollback_deployment
    - Service is "down" after a config change → rollback_config
    - DB connection pool errors → restart_service on the DB, then restart dependent services
    - Cascading failures (multiple services down) → restart_service each one
    - Traffic surge / DDoS → enable_rate_limiting on api-gateway
    - Circuit breaker needed → enable_circuit_breaker on the failing service

    After [PARTIAL SUCCESS]:
    - The incident is NOT fully resolved. Check the dashboard for services still "down" or "degraded".
    - Apply the SAME type of fix (e.g. restart_service) to OTHER affected services still showing problems.
    - Do NOT switch to a different fix type unless logs specifically indicate it.
    - Do NOT escalate after a partial success — you're on the right track, keep going.

    Cascading failure pattern (multiple services down from one upstream cause):
    - Identify the upstream root cause — usually the service with OOM/memory errors in its logs.
    - restart_service on the upstream root cause FIRST.
    - Then enable_circuit_breaker on downstream services still showing errors.
    - Do NOT try to fix downstream services directly — fix the root upstream first, then isolate.

    Multi-incident pattern (two independent alerts, different failure types):
    - Treat each alert independently — they may have completely different root causes.
    - Use query_logs on EACH failing service to understand what type of failure each one is.
    - Apply the correct fix type for EACH independently (e.g. rate limiting for traffic surge, rollback_config for config drift).
    - Do NOT assume the same fix type applies to both.

    After incident is RESOLVED (incident_resolved=True):
    - ALWAYS call write_postmortem() immediately with a brief summary.
    - This earns bonus reward and formally closes the incident.
    - Do NOT call any other action after the incident is resolved.
""").strip()

# ---------------------------------------------------------------------------
# Single-episode runner (async — matches sample_inference.py pattern)
# ---------------------------------------------------------------------------

async def run_episode(oai_client: OpenAI, env: IncidentEnv, scenario_id: int) -> dict:
    """Run one full incident episode. Returns summary dict."""

    log_start(task=f"{TASK_NAME}-scenario{scenario_id}", env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None
    obs = None

    try:
        # Reset — pass scenario_id so the server picks the right scenario
        result = await env.reset(scenario_id=scenario_id)
        obs = result.observation

        alert = obs.active_alerts[0] if obs.active_alerts else "No alert"
        print(f"\n[INFO] Scenario {scenario_id}: {alert[:80]}", flush=True)
        print(obs.service_dashboard, flush=True)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"INCIDENT ALERT:\n{alert}\n\n"
                f"{obs.service_dashboard}\n\n"
                f"Steps remaining: {obs.steps_remaining}\n\n"
                f"Investigate and resolve the incident."
            )},
        ]

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # LLM call
            try:
                response = oai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                error_msg = str(e)
                log_step(step=step, action="llm_error", reward=0.0, done=True, error=error_msg)
                break

            msg = response.choices[0].message

            # No tool call — nudge the model
            if not msg.tool_calls:
                content = msg.content or "(no response)"
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Use one of the available tools to take an action."})
                log_step(step=step, action="no_tool_call", reward=0.0, done=False, error=None)
                continue

            # Parse tool call
            tool_call = msg.tool_calls[0]
            tool_name = tool_call.function.name
            args      = json.loads(tool_call.function.arguments)

            service   = args.get("service", args.get("root_cause", ""))
            reasoning = args.get("reasoning", args.get("reason", args.get("report", "")))
            params    = {k: v for k, v in args.items() if k not in ("service", "reasoning", "root_cause", "reason", "report")}
            action_str = f"{tool_name}(service={service!r})"

            # Step the environment
            try:
                result = await env.step(IncidentAction(
                    action_type=tool_name,
                    service=service,
                    reasoning=reasoning,
                    parameters=params,
                ))
            except Exception as e:
                error_msg = str(e)
                log_step(step=step, action=action_str, reward=0.0, done=True, error=error_msg)
                break

            obs        = result.observation
            reward     = float(obs.step_reward or 0.0)
            done       = result.done
            steps_taken = step

            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            # Update conversation
            messages.append({"role": "assistant", "content": None, "tool_calls": msg.tool_calls})
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": obs.action_result})

            if not done:
                messages.append({"role": "user", "content": (
                    f"{obs.service_dashboard}\n\n"
                    f"Steps remaining: {obs.steps_remaining} | Resolved: {obs.incident_resolved}\n"
                    + (f"Alert: {obs.active_alerts[0]}" if obs.active_alerts else "All alerts cleared.")
                )})

        # Normalise score to (0, 1) exclusive — validator rejects 0.0 and 1.0
        total_reward = sum(rewards)
        score   = min(max(total_reward / MAX_REWARD, 0.001), 0.999)
        success = obs.incident_resolved if obs is not None else False

    except Exception as e:
        error_msg = str(e)
        print(f"[DEBUG] Episode error: {e}", flush=True)
        score = min(max(sum(rewards) / MAX_REWARD, 0.001), 0.999)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "scenario_id":  scenario_id,
        "resolved":     success,
        "steps_used":   steps_taken,
        "total_reward": sum(rewards),
        "score":        score,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not API_KEY:
        raise ValueError(
            "HF_TOKEN not set. Add it to your .env file:\n"
            "  HF_TOKEN=hf_your_token_here"
        )

    print(f"DevOps Incident Management — Inference Script", flush=True)
    print(f"Model:      {MODEL_NAME}", flush=True)
    print(f"API URL:    {API_BASE_URL}", flush=True)
    print(f"Scenarios:  {SCENARIOS_TO_RUN}\n", flush=True)

    oai_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    # Connection priority:
    #   1. ENV_BASE_URL — connect directly to a running server (HF Space or local uvicorn)
    #   2. LOCAL_IMAGE_NAME — pull & start Docker image locally
    #   3. Fallback — assume server already running on localhost:8000
    if ENV_BASE_URL:
        env = IncidentEnv(base_url=ENV_BASE_URL)
        print(f"[INFO] Connecting to remote server: {ENV_BASE_URL}", flush=True)
    else:
        try:
            env = await IncidentEnv.from_docker_image(LOCAL_IMAGE_NAME)
            print(f"[INFO] Using Docker image: {LOCAL_IMAGE_NAME}", flush=True)
        except Exception:
            env = IncidentEnv(base_url="http://localhost:8000")
            print(f"[INFO] Connecting to running server at http://localhost:8000", flush=True)

    results = []
    async with env:
        for i, scenario_id in enumerate(SCENARIOS_TO_RUN):
            summary = await run_episode(oai_client, env, scenario_id)
            results.append(summary)
            if i < len(SCENARIOS_TO_RUN) - 1:
                time.sleep(3)  # avoid HF rate limits between scenarios

    # Final summary
    print("\n" + "="*70, flush=True)
    print("INFERENCE RUN COMPLETE", flush=True)
    resolved = sum(1 for r in results if r["resolved"])
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"Resolved:  {resolved}/{len(results)}", flush=True)
    print(f"Avg score: {avg_score:.3f}", flush=True)
    for r in results:
        status = "RESOLVED" if r["resolved"] else "UNRESOLVED"
        print(f"  Scenario {r['scenario_id']}: {status} | steps={r['steps_used']} | score={r['score']:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

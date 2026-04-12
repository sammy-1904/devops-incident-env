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
    ENV_BASE_URL      Connect directly to a running server (HF Space or local uvicorn)

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
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")
# ENV_BASE_URL: connect directly to a running server (HF Space or local uvicorn).
# Overrides LOCAL_IMAGE_NAME when provided.
# Example: ENV_BASE_URL=https://sammy-1904-devops-incident-env.hf.space
ENV_BASE_URL     = os.environ.get("ENV_BASE_URL", "")

TASK_NAME  = "devops-incident"
BENCHMARK  = "devops-incident-env"
MAX_STEPS  = 22          # worst-case max (Scenario 5)

# Scenarios to evaluate (all 6 — exceeds the minimum-2-tasks requirement)
SCENARIOS_TO_RUN = [1, 2, 3, 4, 5, 6]

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
        "description": (
            "State your root cause hypothesis once you are confident. "
            "In multi-incident scenarios, call diagnose() once per distinct root cause."
        ),
        "parameters": {"type": "object", "properties": {
            "root_cause": {"type": "string", "description": "Clear description of the root cause"},
            "reasoning":  {"type": "string"},
        }, "required": ["root_cause", "reasoning"]},
    }},
    {"type": "function", "function": {
        "name": "write_postmortem",
        "description": (
            "Submit a post-mortem and close the incident. "
            "For false alarms with no fix needed, submit after diagnosing the real cause. "
            "For real incidents, only submit after the incident is fully resolved."
        ),
        "parameters": {"type": "object", "properties": {
            "report": {"type": "string"},
        }, "required": ["report"]},
    }},
    {"type": "function", "function": {
        "name": "escalate",
        "description": "Escalate to a human on-call. Only if you truly cannot resolve it yourself.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"},
        }, "required": ["reason"]},
    }},
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior Site Reliability Engineer (SRE) responding to a production incident.

    CRITICAL SCORING RULES — follow these exactly to maximise your score:
    1. Call diagnose() BEFORE applying any fix. diagnose() earns +20 points per root cause.
       Applying a fix without diagnosing first permanently skips those +20 points.
    2. Call write_postmortem() after the incident is resolved. It earns +5 to +15 points.
       Include root-cause keywords in the report (e.g. "oom", "connection pool", "deployment").
    3. For false alarms: diagnose() then write_postmortem() immediately — do NOT apply any fix.

    INVESTIGATION WORKFLOW (follow in order):
    1. query_logs / check_metrics on the alerting service — gather evidence
    2. diagnose(root_cause=...) — state the root cause (call once per distinct root cause)
    3. Apply the correct fix action
    4. write_postmortem() — close the incident

    RULES:
    - Always provide a detailed 'reasoning' field (>30 chars) for every action — earns bonus points.
    - Start by querying logs for the service mentioned in the alert.
    - Do NOT restart random services. Investigate first, diagnose, then fix.
    - Do NOT escalate unless all options are exhausted.

    DISTINGUISHING ROOT CAUSES FROM DOWNSTREAM VICTIMS:
    - Root cause: the service whose OWN logs show the original failure
      (OOMKilled, deployment crash, config change, connection leak, traffic spike).
    - Downstream victim: a service that failed ONLY because it cannot reach the root cause.
      Its logs show "connection refused", "upstream unavailable", "503 from <other service>".
    - Restarting downstream victims does nothing — fix the root cause first.
    - When many services are down, find the one with the ORIGINAL failure in its own logs.

    FIX SELECTION GUIDE:
    - Service OOMKilled / memory limit in logs     → restart_service on THAT service
    - Services down after a recent deployment      → rollback_deployment on that service
    - Services down after a config change          → rollback_config on that service
    - DB connection pool exhausted in DB logs +
      connection leak in one service's logs        → restart_service:postgres-db,
                                                     then restart_service on the leaking service
    - Traffic surge / DDoS on api-gateway          → enable_rate_limiting:api-gateway
    - Cascading from one upstream OOM              → restart_service on the OOM'd service,
                                                     then enable_circuit_breaker on services
                                                     that were depending on it

    AFTER [PARTIAL SUCCESS]:
    - At least one more fix is needed.
    - Re-read ALL logs you collected. Find the NEXT root cause service — the one whose own logs
      show an original failure, not just "cannot reach upstream".
    - The second fix MAY be a DIFFERENT action type than the first
      (e.g. restart_service for one cause, enable_circuit_breaker for another).
    - Do NOT restart a service just because it is "down" in the dashboard — check its logs.
      If it is down ONLY because its upstream is unavailable, it is a victim, not a root cause.

    MULTI-INCIDENT PATTERN (two independent alerts):
    - Treat each alert as a separate investigation with a separate root cause.
    - Call diagnose() for each root cause separately (you earn +20 per correct diagnosis).
    - Apply a separate fix for each. Do NOT assume the same fix type applies to both.

    FALSE ALARM PATTERN:
    - Check api-gateway logs for "load test", "scheduled", "planned", "ci-pipeline".
    - If caused by expected activity: call diagnose() explaining the real cause.
    - Do NOT apply any fix to healthy services.
    - Immediately call write_postmortem() after diagnosing.

    WORKED EXAMPLE — single OOM crash (follow this pattern exactly):

      Alert: payment-service health checks failing.

      Step 1 — query_logs(service="payment-service", reasoning="Start with the service named in the alert")
        Result: FATAL Killed by signal 9 (OOMKilled). Memory limit exceeded: 256Mi

      Step 2 — diagnose(root_cause="payment-service OOMKilled: memory limit 256Mi exceeded",
                        reasoning="Logs confirm OOMKilled signal. This is the root cause.")
        Result: [DIAGNOSIS ACCEPTED] +20 reward.

      Step 3 — restart_service(service="payment-service",
                               reasoning="OOMKilled process needs restart to recover")
        Result: [SUCCESS] Incident RESOLVED. +30 reward.

      Step 4 — write_postmortem(report="payment-service was OOMKilled after exceeding its 256Mi memory limit.
                                Restarted service to restore. Root cause: memory limit too low or leak present.",
                                reasoning="Closing resolved incident")
        Result: +7 reward.

    Key takeaways from the example:
    - Always query_logs FIRST, then diagnose, then fix, then postmortem.
    - diagnose() uses keywords from the logs (e.g. "oom", "memory limit", "oomkilled").
    - write_postmortem() is called AFTER the fix, not before.
    - For cascading failures: fix root cause first, then remaining affected services.
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
    scenario_max_reward = 80.0   # safe fallback; overwritten from reset metadata

    try:
        # Reset — pass scenario_id so the server picks the right scenario
        result = await env.reset(scenario_id=scenario_id)
        obs = result.observation

        # Read per-scenario max reward from metadata (set by environment.reset())
        scenario_max_reward = float(obs.metadata.get("scenario_max_reward", scenario_max_reward))

        alert = obs.active_alerts[0] if obs.active_alerts else "No alert"
        print(f"\n[INFO] Scenario {scenario_id}: {alert[:80]}", flush=True)
        print(obs.service_dashboard, flush=True)

        # Build initial user message, including deployment events if any
        deploy_section = ""
        if obs.recent_deployments:
            deploy_section = "\n\nRECENT DEPLOYMENTS:\n" + "\n".join(
                f"  {d}" for d in obs.recent_deployments
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"INCIDENT ALERT:\n{alert}\n\n"
                f"{obs.service_dashboard}"
                f"{deploy_section}\n\n"
                f"Steps remaining: {obs.steps_remaining}\n\n"
                f"Investigate and resolve the incident."
            )},
        ]

        # Allow one extra turn for write_postmortem when a fix action resolves the incident
        _allow_postmortem = False
        _terminal_actions = {"write_postmortem", "escalate"}
        tool_name = ""
        _consecutive_no_tool_calls = 0   # hard-stop after 3 in a row

        for step in range(1, MAX_STEPS + 1):
            if result.done and not _allow_postmortem:
                break
            _allow_postmortem = False  # consume the extra turn

            # LLM call — run in a thread so the sync OpenAI client
            # doesn't block the asyncio event loop (which would drop the WebSocket).
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: oai_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="required",
                    ),
                )
            except Exception as e:
                # Fallback: some hosted models don't support tool_choice="required" —
                # retry once with "auto" before giving up.
                try:
                    loop2 = asyncio.get_running_loop()
                    response = await loop2.run_in_executor(
                        None,
                        lambda: oai_client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            tools=TOOLS,
                            tool_choice="auto",
                        ),
                    )
                except Exception as e2:
                    error_msg = str(e2)
                    log_step(step=step, action="llm_error", reward=0.0, done=True, error=error_msg)
                    break

            msg = response.choices[0].message

            # No tool call — model ignored required; nudge once and continue
            if not msg.tool_calls:
                _consecutive_no_tool_calls += 1
                content = msg.content or "(no response)"
                messages.append({"role": "assistant", "content": content})

                # Hard stop: 3 consecutive non-actions → force escalate to end the episode
                # cleanly rather than burning all remaining steps on no-ops.
                if _consecutive_no_tool_calls >= 3:
                    log_step(step=step, action="no_tool_call", reward=0.0, done=False, error=None)
                    try:
                        result = await env.step(IncidentAction(
                            action_type="escalate",
                            reasoning="Agent unable to determine next action. Escalating to human on-call.",
                        ))
                        obs = result.observation
                        reward = float(obs.step_reward or 0.0)
                        rewards.append(reward)
                        log_step(step=step + 1, action="escalate(forced)", reward=reward, done=True, error=None)
                    except Exception:
                        pass
                    break

                if obs and obs.incident_resolved:
                    nudge = (
                        "You MUST call write_postmortem() now. "
                        "Do not write text — call the tool directly."
                    )
                elif not rewards:
                    nudge = (
                        "You MUST call query_logs(service=...) immediately. "
                        "Do not write text — call the tool directly."
                    )
                else:
                    nudge = (
                        "You MUST call one of the available tools now. "
                        "Do not write text — call a tool directly."
                    )
                messages.append({"role": "user", "content": nudge})
                log_step(step=step, action="no_tool_call", reward=0.0, done=False, error=None)
                continue

            # Real tool call — reset the no_tool_call streak counter
            _consecutive_no_tool_calls = 0

            # Parse tool call
            tool_call = msg.tool_calls[0]
            tool_name = tool_call.function.name
            args      = json.loads(tool_call.function.arguments)

            # Routing: diagnose tool uses 'root_cause' param; others use 'service'
            service   = args.get("service", "")
            hypothesis = args.get("root_cause", "")
            reasoning = args.get("reasoning", args.get("reason", args.get("report", "")))
            params    = {
                k: v for k, v in args.items()
                if k not in ("service", "reasoning", "root_cause", "reason", "report")
            }

            if tool_name == "diagnose":
                action_str = f"diagnose(root_cause={hypothesis!r})"
            else:
                action_str = f"{tool_name}(service={service!r})"

            # Step the environment
            try:
                result = await env.step(IncidentAction(
                    action_type=tool_name,
                    service=service,
                    hypothesis=hypothesis,
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

            if done and obs.incident_resolved and tool_name not in _terminal_actions:
                # Fix action resolved the incident — give one extra turn for write_postmortem
                _allow_postmortem = True
                messages.append({"role": "user", "content": (
                    "✓ INCIDENT RESOLVED! Call write_postmortem() now to close the incident "
                    "and earn a quality bonus. Include the root cause keywords in your report "
                    "(e.g. 'oom', 'connection pool', 'bad deployment', 'config change', 'ddos')."
                )})
            elif not done:
                messages.append({"role": "user", "content": (
                    f"{obs.service_dashboard}\n\n"
                    f"Steps remaining: {obs.steps_remaining} | Resolved: {obs.incident_resolved}\n"
                    + (f"Alert: {obs.active_alerts[0]}" if obs.active_alerts else "All alerts cleared.")
                )})

        # Normalise score to (0, 1) exclusive — validator rejects exactly 0.0 and 1.0
        total_reward = sum(rewards)
        score   = min(max(total_reward / scenario_max_reward, 0.001), 0.999)
        success = obs.incident_resolved if obs is not None else False

    except Exception as e:
        error_msg = str(e)
        print(f"[DEBUG] Episode error: {e}", flush=True)
        total_reward = sum(rewards)
        score = min(max(total_reward / scenario_max_reward, 0.001), 0.999)

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
                await asyncio.sleep(3)  # avoid HF rate limits between scenarios

    # Final summary
    print("\n" + "="*70, flush=True)
    print("INFERENCE RUN COMPLETE", flush=True)
    resolved = sum(1 for r in results if r["resolved"])
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"Resolved:  {resolved}/{len(results)}", flush=True)
    print(f"Avg score: {avg_score:.3f}", flush=True)
    for r in results:
        status = "RESOLVED" if r["resolved"] else "UNRESOLVED"
        print(
            f"  Scenario {r['scenario_id']}: {status} | steps={r['steps_used']} | score={r['score']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())

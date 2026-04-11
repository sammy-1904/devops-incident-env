"""
gradio_ui.py — Custom Gradio debug interface for the DevOps Incident environment.

Mounted at /web by app.py (overrides the default OpenEnv template UI).

Purpose:
  - Manually step through any of the 6 scenarios to verify reward signals fire correctly.
  - Inspect service dashboards and log output at each step — the primary way Ben recommended
    for catching reward hacking ("look at the trajectories and give it a smell test").
  - Confirm that wrong actions produce negative rewards and correct ones earn positives.
  - Verify the new anti-reward-hacking gate: diagnose() rejected if no investigation done.
"""

import threading
from typing import List, Tuple

import gradio as gr

from .environment import IncidentEnvironment
from .models import IncidentAction

# ---------------------------------------------------------------------------
# Shared state for the UI session (separate from training WebSocket sessions)
# ---------------------------------------------------------------------------

_env = IncidentEnvironment()
_lock = threading.Lock()
_step_history: List[dict] = []


def _fmt_history() -> str:
    if not _step_history:
        return "(no actions taken yet)"
    lines = []
    for i, entry in enumerate(_step_history, 1):
        flag = f"  {entry['note']}" if entry["note"] else ""
        lines.append(
            f"Step {i:>2}: {entry['action']:<42}  reward={entry['reward']:>+6.1f}{flag}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Callback: reset
# ---------------------------------------------------------------------------

def reset_env(scenario_id: int) -> Tuple[str, str, str, str, str]:
    global _step_history
    with _lock:
        _step_history = []
        obs = _env.reset(scenario_id=int(scenario_id))

    alert = obs.active_alerts[0] if obs.active_alerts else "No alert"
    deploy_section = ""
    if obs.recent_deployments:
        deploy_section = "\n\nRECENT DEPLOYMENTS:\n" + "\n".join(
            f"  {d}" for d in obs.recent_deployments
        )

    return (
        obs.service_dashboard,
        alert + deploy_section,
        obs.action_result,
        f"Steps remaining: {obs.steps_remaining}  |  Total reward: 0.0  |  Resolved: False",
        _fmt_history(),
    )


# ---------------------------------------------------------------------------
# Callback: take one action
# ---------------------------------------------------------------------------

def take_action(
    action_type: str,
    service_or_text: str,
    reasoning: str,
) -> Tuple[str, str, str, str, str]:
    """
    Route service_or_text to the correct IncidentAction field:
      - diagnose          → hypothesis
      - write_postmortem  → reasoning (report)
      - escalate          → reasoning
      - everything else   → service
    """
    service = ""
    hypothesis = ""

    if action_type == "diagnose":
        hypothesis = service_or_text
    elif action_type in ("write_postmortem", "escalate"):
        reasoning = service_or_text or reasoning
    else:
        service = service_or_text

    with _lock:
        try:
            obs = _env.step(IncidentAction(
                action_type=action_type,
                service=service,
                hypothesis=hypothesis,
                reasoning=reasoning,
            ))
            state = _env.state

            note = ""
            if obs.incident_resolved:
                note = "✓ RESOLVED"
            elif obs.done:
                note = "✗ EPISODE ENDED"

            target = service or hypothesis or "—"
            _step_history.append({
                "action": f"{action_type}({target!r})",
                "reward": obs.step_reward,
                "note":   note,
            })

            alert = obs.active_alerts[0] if obs.active_alerts else "✓ All alerts cleared"
            status = (
                f"Steps remaining: {obs.steps_remaining}  |  "
                f"Total reward: {state.total_reward:.1f}  |  "
                f"Resolved: {state.resolved}"
            )
            return (
                obs.service_dashboard,
                alert,
                obs.action_result,
                status,
                _fmt_history(),
            )
        except Exception as exc:
            return ("", "", f"[ERROR] {exc}", "", _fmt_history())


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

_SCENARIO_INFO = {
    1: "Single OOM Crash                (D: 0.1)",
    2: "Bad Deployment                  (D: 0.3)",
    3: "DB Connection Pool Exhaustion   (D: 0.5)",
    4: "Cascading Microservice Failure  (D: 0.75)",
    5: "Multi-Incident Surge            (D: 1.0)",
    6: "False Alarm: Planned Load Test  (D: 0.2)",
}

_ACTION_CHOICES = [
    "query_logs",
    "check_metrics",
    "restart_service",
    "rollback_deployment",
    "rollback_config",
    "enable_circuit_breaker",
    "enable_rate_limiting",
    "diagnose",
    "write_postmortem",
    "escalate",
]

_SERVICE_HINT = (
    "Service name (e.g. user-service)  "
    "— or root-cause text for diagnose  "
    "— or report text for write_postmortem"
)


def create_gradio_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app for /web."""

    scenario_md = "\n".join(f"- **{k}** — {v}" for k, v in _SCENARIO_INFO.items())

    with gr.Blocks(title="DevOps Incident Env", theme=gr.themes.Monochrome()) as demo:

        gr.Markdown(
            "# DevOps Incident Management Environment\n"
            "**Debug UI** — step through scenarios manually to verify reward signals "
            "and inspect agent trajectories.\n\n"
            f"**Scenarios:**\n{scenario_md}"
        )

        # ── Episode controls ──────────────────────────────────────────────
        with gr.Row():
            scenario_slider = gr.Slider(
                minimum=1, maximum=6, step=1, value=1,
                label="Scenario ID",
                scale=3,
            )
            reset_btn = gr.Button("⟳  Reset / New Episode", variant="primary", scale=1)

        # ── Dashboard + result ────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                dashboard_box = gr.Textbox(
                    label="Service Dashboard",
                    lines=10, max_lines=12,
                    interactive=False,
                )
                alert_box = gr.Textbox(
                    label="Active Alert & Recent Deployments",
                    lines=4, max_lines=6,
                    interactive=False,
                )
                status_box = gr.Textbox(
                    label="Episode Status",
                    lines=1,
                    interactive=False,
                )

            with gr.Column(scale=1):
                result_box = gr.Textbox(
                    label="Last Action Result",
                    lines=10, max_lines=15,
                    interactive=False,
                )
                history_box = gr.Textbox(
                    label="Step History  (action → reward)",
                    lines=8, max_lines=10,
                    interactive=False,
                )

        # ── Action controls ───────────────────────────────────────────────
        gr.Markdown("### Take an Action")
        with gr.Row():
            action_dd = gr.Dropdown(
                choices=_ACTION_CHOICES,
                value="query_logs",
                label="Action Type",
                scale=1,
            )
            service_input = gr.Textbox(
                label=_SERVICE_HINT,
                placeholder="e.g.  user-service",
                scale=2,
            )
            reasoning_input = gr.Textbox(
                label="Reasoning  (>30 chars earns +1 bonus)",
                placeholder="Why are you taking this action?",
                scale=2,
            )
        step_btn = gr.Button("▶  Take Action", variant="secondary")

        # ── Reward reference ──────────────────────────────────────────────
        with gr.Accordion("Reward reference", open=False):
            gr.Markdown(
                "| Event | Reward |\n"
                "|---|---|\n"
                "| First look at root-cause service | +5 |\n"
                "| Correct diagnosis (per root cause) | +20 |\n"
                "| Partial fix (multi-fix scenarios) | +10 |\n"
                "| Final fix — incident resolved | +30 |\n"
                "| False alarm correctly identified | +20 bonus |\n"
                "| Post-mortem after resolution | +5 base + up to +10 quality |\n"
                "| Efficiency bonus (steps saved) | up to +10 |\n"
                "| Reasoning field > 30 chars | +1 per step |\n"
                "| Wrong fix applied | −10 |\n"
                "| Repeated query ≥ 3× same service | −2 |\n"
                "| Incorrect escalation | −15 |\n"
                "| Max steps exceeded | −20 |\n"
                "| Premature post-mortem | −10 |\n"
                "| Diagnose without investigating first | 0 (rejected) |"
            )

        # ── Wire up callbacks ─────────────────────────────────────────────
        outputs = [dashboard_box, alert_box, result_box, status_box, history_box]

        reset_btn.click(fn=reset_env, inputs=[scenario_slider], outputs=outputs)
        step_btn.click(
            fn=take_action,
            inputs=[action_dd, service_input, reasoning_input],
            outputs=outputs,
        )

    return demo

"""
gradio_ui.py — Custom Gradio interface for the DevOps Incident Management environment.

Mounted at /web by app.py, replacing the default OpenEnv web interface.
Lets judges and developers interactively drive incidents: pick a scenario,
take actions step-by-step, and watch rewards accumulate in real time.
"""

import gradio as gr
from .environment import IncidentEnvironment
from .models import IncidentAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENARIO_CHOICES = [
    "1 — Single OOM Crash            (D: 0.1)",
    "2 — Bad Deployment               (D: 0.3)",
    "3 — DB Connection Pool           (D: 0.5)",
    "4 — Cascading Microservice Failure (D: 0.75)",
    "5 — Multi-Incident: DDoS + Config Drift (D: 1.0)",
]

SCENARIO_IDS = {label: i + 1 for i, label in enumerate(SCENARIO_CHOICES)}

ACTION_TYPES = [
    "query_logs",
    "check_metrics",
    "restart_service",
    "rollback_deployment",
    "rollback_config",
    "enable_circuit_breaker",
    "enable_rate_limiting",
    "diagnose",
    "escalate",
    "write_postmortem",
]

ACTION_DESCRIPTIONS = {
    "query_logs":             "Retrieve log entries for a service. Look for errors, OOM kills, stack traces.",
    "check_metrics":          "Check CPU, memory, error rate and p99 latency for a service.",
    "restart_service":        "Restart a crashed/OOMKilled service.",
    "rollback_deployment":    "Roll back the last deployment (use when a recent deploy caused errors).",
    "rollback_config":        "Roll back the last config change for a service.",
    "enable_circuit_breaker": "Isolate a service to prevent cascade failures.",
    "enable_rate_limiting":   "Throttle traffic to a service (e.g. DDoS mitigation).",
    "diagnose":               "State your root cause hypothesis once confident.",
    "escalate":               "Hand off to a human on-call (only if all options exhausted).",
    "write_postmortem":       "Submit incident report and close the ticket (only after resolution).",
}

CSS = """
.monospace textarea, .monospace input { font-family: 'Courier New', monospace !important; font-size: 12px !important; }
.resolved { background-color: #d4edda !important; }
.active   { background-color: #fff3cd !important; }
.failed   { background-color: #f8d7da !important; }
footer { display: none !important; }
"""

# ---------------------------------------------------------------------------
# Event handlers — all pure Python, no HTTP round-trip
# ---------------------------------------------------------------------------

def start_incident(scenario_label, env_state, rewards_state, log_state):
    """Reset (or create) environment for the chosen scenario."""
    scenario_id = SCENARIO_IDS[scenario_label]

    if env_state is None:
        env_state = IncidentEnvironment()

    obs = env_state.reset(scenario_id=scenario_id)
    rewards_state = []
    log_state = []

    log_state.append(f"[RESET] Scenario {scenario_id} started.")

    hint_text = obs.hint or ""

    return (
        env_state,
        rewards_state,
        log_state,
        obs.service_dashboard,
        "\n".join(obs.active_alerts),
        obs.action_result,
        hint_text,
        0.0,           # step reward
        0.0,           # total reward
        float(obs.steps_remaining),
        "ACTIVE",
        _build_log_text(log_state),
    )


def take_action(action_type, service, reasoning, env_state, rewards_state, log_state):
    """Execute one agent action against the environment."""
    if env_state is None or env_state._scenario is None:
        return (
            env_state, rewards_state, log_state,
            "No active incident. Start one first.", "", "", "",
            0.0, 0.0, 0.0, "Not started", "",
        )

    action = IncidentAction(
        action_type=action_type,
        service=service.strip(),
        reasoning=reasoning.strip(),
        parameters={},
        metadata={},
    )

    obs = env_state.step(action)
    rewards_state = rewards_state + [obs.step_reward]
    total = sum(rewards_state)

    log_entry = (
        f"[STEP {len(rewards_state)}] {action_type}(service={service!r}) "
        f"-> reward={obs.step_reward:+.1f}  total={total:+.1f}"
    )
    log_state = log_state + [log_entry]

    if obs.incident_resolved:
        status = "RESOLVED"
    elif obs.done:
        status = "FAILED / TIMEOUT"
    else:
        status = "ACTIVE"

    alerts_text = (
        "\n".join(obs.active_alerts) if obs.active_alerts else "All alerts cleared."
    )

    return (
        env_state,
        rewards_state,
        log_state,
        obs.service_dashboard,
        alerts_text,
        obs.action_result,
        obs.hint or "",
        obs.step_reward,
        total,
        float(obs.steps_remaining),
        status,
        _build_log_text(log_state),
    )


def update_action_hint(action_type):
    return ACTION_DESCRIPTIONS.get(action_type, "")


def _build_log_text(log_state):
    return "\n".join(log_state) if log_state else ""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def create_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="DevOps Incident Manager") as demo:

        # ── Persistent state (server-side, per session) ──────────────────
        env_state     = gr.State(None)
        rewards_state = gr.State([])
        log_state     = gr.State([])

        # ── Header ───────────────────────────────────────────────────────
        gr.Markdown("""
        # DevOps Incident Management — Interactive Demo
        **An RL environment for training AI agents as senior SREs.**
        Pick a scenario, click *Start Incident*, then use the action panel to investigate and resolve.
        Rewards accumulate in real time exactly as they do during RL training.
        """)

        # ── Scenario selector ─────────────────────────────────────────────
        with gr.Row():
            scenario_dd = gr.Dropdown(
                choices=SCENARIO_CHOICES,
                value=SCENARIO_CHOICES[0],
                label="Scenario",
                scale=4,
            )
            start_btn = gr.Button("Start Incident", variant="primary", scale=1)

        # ── Alerts + hint ─────────────────────────────────────────────────
        alerts_box = gr.Textbox(label="Active Alerts", lines=2, interactive=False)
        hint_box   = gr.Textbox(label="Hint (Scenario 1 only)", lines=1, interactive=False)

        # ── Service dashboard ─────────────────────────────────────────────
        dashboard_box = gr.Textbox(
            label="Service Dashboard",
            lines=10,
            interactive=False,
            elem_classes=["monospace"],
        )

        # ── Action panel ──────────────────────────────────────────────────
        gr.Markdown("### Take an Action")
        with gr.Row():
            action_dd = gr.Dropdown(
                choices=ACTION_TYPES,
                value="query_logs",
                label="Action Type",
                scale=1,
            )
            service_box = gr.Textbox(
                label="Service / Root Cause / Report",
                placeholder="e.g. user-service",
                scale=2,
            )
        action_hint_box = gr.Textbox(
            label="Action description",
            value=ACTION_DESCRIPTIONS["query_logs"],
            interactive=False,
            lines=1,
        )
        reasoning_box = gr.Textbox(
            label="Reasoning  (>30 chars earns +1 bonus reward)",
            placeholder="Why are you taking this action?",
            lines=2,
        )
        step_btn = gr.Button("Take Action", variant="secondary")

        # ── Result ────────────────────────────────────────────────────────
        action_result_box = gr.Textbox(
            label="Action Result",
            lines=7,
            interactive=False,
            elem_classes=["monospace"],
        )

        # ── Metrics row ───────────────────────────────────────────────────
        with gr.Row():
            step_reward_num    = gr.Number(label="Step Reward",     value=0.0,  interactive=False)
            total_reward_num   = gr.Number(label="Total Reward",    value=0.0,  interactive=False)
            steps_remaining_num = gr.Number(label="Steps Remaining", value=0.0, interactive=False)
            status_box         = gr.Textbox(label="Episode Status", value="Not started", interactive=False)

        # ── Step log ──────────────────────────────────────────────────────
        log_box = gr.Textbox(
            label="Step Log",
            lines=6,
            interactive=False,
            elem_classes=["monospace"],
        )

        # ── Shared output list ────────────────────────────────────────────
        outputs = [
            env_state, rewards_state, log_state,
            dashboard_box, alerts_box, action_result_box, hint_box,
            step_reward_num, total_reward_num, steps_remaining_num,
            status_box, log_box,
        ]

        # ── Event wiring ──────────────────────────────────────────────────
        start_btn.click(
            fn=start_incident,
            inputs=[scenario_dd, env_state, rewards_state, log_state],
            outputs=outputs,
        )

        step_btn.click(
            fn=take_action,
            inputs=[action_dd, service_box, reasoning_box, env_state, rewards_state, log_state],
            outputs=outputs,
        )

        action_dd.change(
            fn=update_action_hint,
            inputs=[action_dd],
            outputs=[action_hint_box],
        )

    return demo

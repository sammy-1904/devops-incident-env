"""
environment.py — Core episode logic for the DevOps Incident Management environment.

Confirmed API (from inspecting openenv-core):
  - Environment is ABC, generic: Environment[ActT, ObsT, StateT]
  - reset() signature: reset(self, seed=None, episode_id=None, **kwargs) -> ObsT
  - step() signature:  step(self, action: ActT, timeout_s=None, **kwargs) -> ObsT
  - state: property returning StateT
"""

import uuid
from typing import List, Optional

from openenv.core import Environment

from .models import IncidentAction, IncidentObservation, IncidentState

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.scenarios import ALL_SCENARIOS, SCENARIO_BY_ID, Scenario, ServiceState


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

R_CORRECT_FIX        =  30.0
R_CORRECT_DIAGNOSIS  =  20.0
R_FIRST_LOOK_CULPRIT =   5.0
R_PARTIAL_FIX        =  10.0
R_GOOD_POSTMORTEM    =   5.0

P_WRONG_FIX          = -10.0
P_REDUNDANT_ACTION   =  -2.0
P_WRONG_ESCALATE     = -15.0
P_EXCEED_STEPS       = -20.0
P_UNRESOLVED_END     = -10.0

R_ESCALATE_CORRECT   =   5.0


class IncidentEnvironment(Environment[IncidentAction, IncidentObservation, IncidentState]):
    """
    Stateful DevOps incident simulation.

    SUPPORTS_CONCURRENT_SESSIONS = True because each reset() creates fully
    independent state — no shared mutable variables between sessions.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._scenario: Optional[Scenario] = None
        self._state: IncidentState = IncidentState()
        self._services: dict = {}
        self._action_log: List[str] = []
        self._fixes_applied: List[str] = []
        self._scenario_index: int = -1

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_id: Optional[int] = None,
        **kwargs,
    ) -> IncidentObservation:
        """
        Start a new incident episode.

        Args:
            seed: Unused (reserved by OpenEnv spec).
            episode_id: Caller-supplied episode ID; auto-generated if None.
            scenario_id: 1–5 to pick a specific scenario; cycles if None.
        """
        # Pick scenario
        if scenario_id is not None and scenario_id in SCENARIO_BY_ID:
            self._scenario = SCENARIO_BY_ID[scenario_id]
        else:
            self._scenario_index = (self._scenario_index + 1) % len(ALL_SCENARIOS)
            self._scenario = ALL_SCENARIOS[self._scenario_index]

        sc = self._scenario

        # Deep-copy service states for this episode
        self._services = {
            name: ServiceState(
                name=svc.name,
                status=svc.status,
                cpu_pct=svc.cpu_pct,
                memory_pct=svc.memory_pct,
                error_rate_pct=svc.error_rate_pct,
                p99_latency_ms=svc.p99_latency_ms,
                is_root_cause=svc.is_root_cause,
                is_red_herring=svc.is_red_herring,
            )
            for name, svc in sc.services.items()
        }

        self._action_log = []
        self._fixes_applied = []

        self._state = IncidentState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            scenario_id=sc.id,
            scenario_name=sc.name,
            difficulty=sc.difficulty,
            total_reward=0.0,
            resolved=False,
            services_investigated=0,
            correct_diagnosis_given=False,
        )

        dashboard = self._build_dashboard()
        return IncidentObservation(
            done=False,
            reward=0.0,
            metadata={},
            action_result=(
                f"=== INCIDENT OPENED ===\n"
                f"Scenario: {sc.name} (Difficulty: {sc.difficulty:.1f})\n"
                f"Episode ID: {self._state.episode_id}\n\n"
                f"AVAILABLE ACTIONS:\n"
                f"  query_logs <service>          — retrieve log entries\n"
                f"  check_metrics <service>       — see CPU/memory/latency/error_rate\n"
                f"  restart_service <service>     — restart a service\n"
                f"  rollback_deployment <service> — roll back last deployment\n"
                f"  rollback_config <service>     — roll back last config change\n"
                f"  enable_circuit_breaker <svc>  — isolate a service\n"
                f"  enable_rate_limiting <svc>    — throttle traffic\n"
                f"  diagnose <root_cause_text>    — state your hypothesis\n"
                f"  escalate <reason>             — hand off to human\n"
                f"  write_postmortem <text>       — close incident\n\n"
                f"SERVICES: {', '.join(sc.services.keys())}"
            ),
            service_dashboard=dashboard,
            active_alerts=[sc.alert_message],
            steps_remaining=sc.max_steps,
            incident_resolved=False,
            step_reward=0.0,
            hint=sc.hint,
        )

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------

    def step(
        self,
        action: IncidentAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> IncidentObservation:
        """Execute one agent action and return the resulting observation."""
        sc = self._scenario
        self._state.step_count += 1
        steps_left = sc.max_steps - self._state.step_count
        step_reward = 0.0
        result_text = ""
        done = False

        action_key = f"{action.action_type}:{action.service}"

        # Format reward: small bonus for non-trivial reasoning (RL training signal)
        if len(action.reasoning.strip()) > 30:
            step_reward += 1.0

        if action.action_type == "query_logs":
            result_text, bonus = self._handle_query_logs(action.service)
            step_reward += bonus

        elif action.action_type == "check_metrics":
            result_text, bonus = self._handle_check_metrics(action.service)
            step_reward += bonus

        elif action.action_type in (
            "restart_service", "rollback_deployment", "rollback_config",
            "enable_circuit_breaker", "enable_rate_limiting",
        ):
            result_text, reward, done = self._handle_remediation(action.service, action.action_type)
            step_reward += reward

        elif action.action_type == "diagnose":
            result_text, reward = self._handle_diagnose(action.service or action.reasoning)
            step_reward += reward

        elif action.action_type == "escalate":
            result_text, reward, done = self._handle_escalate(action.reasoning)
            step_reward += reward

        elif action.action_type == "write_postmortem":
            result_text, reward, done = self._handle_postmortem(action.reasoning)
            step_reward += reward

        else:
            result_text = f"Unknown action_type: '{action.action_type}'. No effect."
            step_reward = -1.0

        # Redundancy penalty
        if action.action_type in ("query_logs", "check_metrics"):
            self._action_log.append(action_key)
            count = self._action_log.count(action_key)
            if count >= 3:
                step_reward += P_REDUNDANT_ACTION
                result_text += f"\n[WARN] '{action_key}' called {count} times. Redundancy penalty."

        # Max-steps exceeded
        if steps_left <= 0 and not done:
            done = True
            step_reward += P_EXCEED_STEPS
            result_text += f"\n\n[TIMEOUT] Max steps ({sc.max_steps}) reached. Incident auto-escalated."

        self._state.total_reward += step_reward

        if self._state.resolved:
            done = True

        # Efficiency bonus on resolution
        if done and self._state.resolved and steps_left > 0:
            efficiency_bonus = round(10.0 * (steps_left / sc.max_steps), 2)
            self._state.total_reward += efficiency_bonus
            result_text += f"\n\n[EFFICIENCY BONUS] +{efficiency_bonus:.1f} ({steps_left} steps remaining)."

        dashboard = self._build_dashboard()
        alerts = [] if self._state.resolved else [sc.alert_message]

        return IncidentObservation(
            done=done,
            reward=step_reward,
            metadata={"total_reward": self._state.total_reward},
            action_result=result_text,
            service_dashboard=dashboard,
            active_alerts=alerts,
            steps_remaining=max(0, steps_left),
            incident_resolved=self._state.resolved,
            step_reward=step_reward,
            hint=sc.hint if self._state.step_count <= 2 else None,
        )

    # -----------------------------------------------------------------------
    # state property
    # -----------------------------------------------------------------------

    @property
    def state(self) -> IncidentState:
        return self._state

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _build_dashboard(self) -> str:
        lines = ["SERVICE DASHBOARD:"]
        lines.append(f"  {'Service':<25} {'Status':<10} {'CPU%':<8} {'Mem%':<8} {'ErrRate%':<10} {'p99ms'}")
        lines.append("  " + "-" * 72)
        for name, svc in self._services.items():
            icon = {"healthy": "✓", "degraded": "!", "down": "✗"}.get(svc.status, "?")
            lines.append(
                f"  {icon} {name:<23} {svc.status:<10} {svc.cpu_pct:<8.1f} "
                f"{svc.memory_pct:<8.1f} {svc.error_rate_pct:<10.1f} {svc.p99_latency_ms:.0f}"
            )
        return "\n".join(lines)

    def _handle_query_logs(self, service: str):
        sc = self._scenario
        if service not in self._services:
            return (f"Error: '{service}' not found. Available: {', '.join(self._services.keys())}", 0.0)

        bonus = 0.0
        action_key = f"query_logs:{service}"
        if self._services[service].is_root_cause and self._action_log.count(action_key) == 0:
            bonus = R_FIRST_LOOK_CULPRIT
            self._state.services_investigated += 1

        relevant_logs = [log for log in sc.logs if log.service == service]
        if not relevant_logs:
            log_text = f"  [No log entries found for {service} in the last 30 minutes]"
        else:
            log_text = "\n".join(
                f"  [{log.timestamp}] [{log.level}] {log.message}"
                for log in relevant_logs
            )

        result = f"=== LOGS: {service} (last 30 min) ===\n{log_text}"
        if bonus > 0:
            result += f"\n[+{bonus:.0f} reward: investigating root cause service]"
        return result, bonus

    def _handle_check_metrics(self, service: str):
        if service not in self._services:
            return (f"Error: '{service}' not found. Available: {', '.join(self._services.keys())}", 0.0)

        svc = self._services[service]
        bonus = 0.0
        action_key = f"check_metrics:{service}"
        if svc.is_root_cause and self._action_log.count(action_key) == 0:
            bonus = R_FIRST_LOOK_CULPRIT
            self._state.services_investigated += 1

        result = (
            f"=== METRICS: {service} ===\n"
            f"  Status:      {svc.status}\n"
            f"  CPU:         {svc.cpu_pct:.1f}%\n"
            f"  Memory:      {svc.memory_pct:.1f}%\n"
            f"  Error rate:  {svc.error_rate_pct:.1f}%\n"
            f"  p99 latency: {svc.p99_latency_ms:.0f}ms"
        )
        if bonus > 0:
            result += f"\n[+{bonus:.0f} reward: investigating root cause service]"
        return result, bonus

    def _handle_remediation(self, service: str, action_type: str):
        sc = self._scenario
        fix_key = f"{action_type}:{service}"

        if service not in self._services:
            return (f"Error: '{service}' not found. Available: {', '.join(self._services.keys())}", 0.0, False)

        if fix_key in sc.correct_fix_actions:
            self._fixes_applied.append(fix_key)
            self._services[service].status = "healthy"
            self._services[service].error_rate_pct = 0.0
            self._services[service].p99_latency_ms = max(50.0, self._services[service].p99_latency_ms * 0.05)

            remaining = [f for f in sc.correct_fix_actions if f not in self._fixes_applied]

            if not remaining:
                self._state.resolved = True
                return (
                    f"[SUCCESS] {action_type.replace('_', ' ').title()} on {service}.\n"
                    f"Incident FULLY RESOLVED. All services recovering.\n",
                    R_CORRECT_FIX,
                    True,
                )
            else:
                still_down = [n for n, s in self._services.items() if s.status in ("down", "degraded") and n != service]
                still_down_str = ", ".join(still_down) if still_down else "check dashboard"
                return (
                    f"[PARTIAL SUCCESS] {action_type.replace('_', ' ').title()} on {service}. +{R_PARTIAL_FIX:.0f} reward.\n"
                    f"Other issue(s) still active. Services still affected: {still_down_str}.\n"
                    f"Apply the same fix type ({action_type}) to other affected services.\n",
                    R_PARTIAL_FIX,
                    False,
                )
        else:
            return (
                f"[FAILED] {action_type.replace('_', ' ').title()} on {service}. No improvement observed.\n"
                f"Incident still active. Is this really the root cause?\n",
                P_WRONG_FIX,
                False,
            )

    def _handle_diagnose(self, hypothesis: str):
        sc = self._scenario
        matched = any(kw in hypothesis.lower() for kw in sc.correct_diagnoses)

        if matched and not self._state.correct_diagnosis_given:
            self._state.correct_diagnosis_given = True
            return (
                f"[DIAGNOSIS ACCEPTED] '{hypothesis}'\nMatches actual root cause. +{R_CORRECT_DIAGNOSIS:.0f} reward.\n"
                f"Now apply the correct remediation action.",
                R_CORRECT_DIAGNOSIS,
            )
        elif matched:
            return "[DIAGNOSIS] Already recorded. Focus on remediation.", 0.0
        else:
            return (
                f"[DIAGNOSIS] '{hypothesis}' — does not match root cause. No reward. Keep investigating.",
                0.0,
            )

    def _handle_escalate(self, reason: str):
        sc = self._scenario
        if sc.id in (1, 2, 3, 4):
            return (
                f"[ESCALATION] Handed off: '{reason}'\nThis incident WAS solvable. Penalty: {P_WRONG_ESCALATE:.0f}.",
                P_WRONG_ESCALATE,
                True,
            )
        remaining = [f for f in sc.correct_fix_actions if f not in self._fixes_applied]
        if remaining and not self._state.resolved:
            return (
                f"[ESCALATION] '{reason}'\nPartial escalation. Small reward for acknowledging limits.",
                R_ESCALATE_CORRECT,
                True,
            )
        return (f"[ESCALATION] '{reason}'\nEpisode ended.", P_UNRESOLVED_END, True)

    def _handle_postmortem(self, report: str):
        if self._state.resolved:
            return (
                f"[POST-MORTEM SUBMITTED]\n{report}\n\nIncident closed. +{R_GOOD_POSTMORTEM:.0f} bonus.",
                R_GOOD_POSTMORTEM,
                True,
            )
        return (
            f"[POST-MORTEM — PREMATURE]\n{report}\n\nIncident not yet resolved. Penalty: {P_UNRESOLVED_END:.0f}.",
            P_UNRESOLVED_END,
            True,
        )

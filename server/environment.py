"""
environment.py — Core episode logic for the DevOps Incident Management environment.

Confirmed API (from inspecting openenv-core):
  - Environment is ABC, generic: Environment[ActT, ObsT, StateT]
  - reset() signature: reset(self, seed=None, episode_id=None, **kwargs) -> ObsT
  - step() signature:  step(self, action: ActT, timeout_s=None, **kwargs) -> ObsT
  - state: property returning StateT
"""

import random as _random
import uuid
from typing import List, Optional, Set

from openenv.core import Environment

from .models import IncidentAction, IncidentObservation, IncidentState

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.scenarios import generate_scenario, ALL_SCENARIO_IDS, Scenario, ServiceState


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

R_CORRECT_FIX        =  30.0
R_CORRECT_DIAGNOSIS  =  20.0
R_FIRST_LOOK_CULPRIT =   5.0
R_PARTIAL_FIX        =  10.0
R_GOOD_POSTMORTEM    =   5.0
R_NO_ACTION_NEEDED   =  20.0   # bonus for Scenario 6: correct false-alarm identification

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
        self._wrong_fixes_applied: List[str] = []   # tracks wrong fixes for Scenario 6 bonus
        self._scenario_index: int = -1
        self._diagnosis_groups_awarded: Set[int] = set()   # indices into correct_diagnosis_groups
        self._queried_services: Set[str] = set()           # ANY service queried (gate for diagnose)
        self._first_looked_root_causes: Set[str] = set()   # dedup first-look +5 bonus per service

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
            scenario_id: 1–6 to pick a specific scenario; cycles if None.
        """
        # Pick scenario — generate a fresh variant using the provided seed (or a random one)
        effective_seed = seed if seed is not None else _random.randint(0, 2**31 - 1)
        if scenario_id is not None and scenario_id in ALL_SCENARIO_IDS:
            self._scenario = generate_scenario(scenario_id, seed=effective_seed)
        else:
            self._scenario_index = (self._scenario_index + 1) % len(ALL_SCENARIO_IDS)
            self._scenario = generate_scenario(ALL_SCENARIO_IDS[self._scenario_index],
                                               seed=effective_seed)

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
        self._wrong_fixes_applied = []
        self._diagnosis_groups_awarded = set()
        self._queried_services = set()
        self._first_looked_root_causes = set()

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

        # Format recent deployment events for the observation
        if sc.deployment_events:
            deploy_lines = "\n".join(
                f"  {e.timestamp} | {e.service}: {e.old_version} → {e.new_version} by {e.deployed_by}"
                for e in sc.deployment_events
            )
        else:
            deploy_lines = "  None in the last 30 minutes"

        recent_deployments = [
            f"{e.timestamp} | {e.service}: {e.old_version} → {e.new_version} by {e.deployed_by}"
            for e in sc.deployment_events
        ]

        return IncidentObservation(
            done=False,
            reward=None,   # no action taken on reset; step_reward=0.0 is the domain field
            metadata={"scenario_max_reward": sc.max_reward},
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
                f"SERVICES: {', '.join(sc.services.keys())}\n\n"
                f"RECENT DEPLOYMENTS:\n{deploy_lines}"
            ),
            service_dashboard=dashboard,
            active_alerts=[sc.alert_message],
            steps_remaining=sc.max_steps,
            incident_resolved=False,
            step_reward=0.0,
            hint=sc.hint,
            recent_deployments=recent_deployments,
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
        if self._state.resolved or (self._scenario and self._state.step_count >= self._scenario.max_steps):
            raise RuntimeError(
                "step() called after the episode has ended. Call reset() to start a new episode."
            )
        sc = self._scenario
        self._state.step_count += 1
        steps_left = sc.max_steps - self._state.step_count
        step_reward = 0.0
        result_text = ""
        done = False

        action_key = f"{action.action_type}:{action.service}"

        # Small bonus for non-trivial reasoning (RL training signal)
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
            # Read hypothesis from the dedicated field first, fall back to service/reasoning
            text = action.hypothesis or action.service or action.reasoning
            result_text, reward = self._handle_diagnose(text)
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

        # Redundancy penalty for repeated information queries
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

        # Efficiency bonus on resolution — added to step_reward so it surfaces in inference.py
        if done and self._state.resolved and steps_left > 0:
            efficiency_bonus = round(10.0 * (steps_left / sc.max_steps), 2)
            step_reward += efficiency_bonus
            self._state.total_reward += efficiency_bonus
            result_text += f"\n\n[EFFICIENCY BONUS] +{efficiency_bonus:.1f} ({steps_left} steps remaining)."

        dashboard = self._build_dashboard()
        alerts = [] if self._state.resolved else [sc.alert_message]

        return IncidentObservation(
            done=done,
            reward=step_reward,
            metadata={"total_reward": self._state.total_reward, "scenario_max_reward": sc.max_reward},
            action_result=result_text,
            service_dashboard=dashboard,
            active_alerts=alerts,
            steps_remaining=max(0, steps_left),
            incident_resolved=self._state.resolved,
            step_reward=step_reward,
            hint=sc.hint if self._state.step_count <= 3 else None,
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

        self._queried_services.add(service)

        bonus = 0.0
        if self._services[service].is_root_cause and service not in self._first_looked_root_causes:
            self._first_looked_root_causes.add(service)
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

        self._queried_services.add(service)

        svc = self._services[service]
        bonus = 0.0
        if svc.is_root_cause and service not in self._first_looked_root_causes:
            self._first_looked_root_causes.add(service)
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

        # Scenario 6 (false alarm): any fix attempt on a healthy system is penalised
        if not sc.correct_fix_actions:
            self._wrong_fixes_applied.append(fix_key)
            return (
                f"[WRONG ACTION] {action_type.replace('_', ' ').title()} on {service}.\n"
                f"All services are operating normally. This action was unnecessary.\n"
                f"Penalty: {P_WRONG_FIX:.0f}. Investigate more before acting.",
                P_WRONG_FIX,
                False,
            )

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
                # Show only the services that are part of remaining correct fixes — not all
                # degraded/down services (which may just be downstream victims).
                remaining_fixes = [f for f in sc.correct_fix_actions if f not in self._fixes_applied]
                remaining_svc_names = [f.split(":", 1)[1] for f in remaining_fixes]
                still_down_str = ", ".join(remaining_svc_names) if remaining_svc_names else "check dashboard"
                return (
                    f"[PARTIAL SUCCESS] {action_type.replace('_', ' ').title()} on {service}. +{R_PARTIAL_FIX:.0f} reward.\n"
                    f"Other root cause(s) still active. Next service(s) to address: {still_down_str}.\n"
                    f"Check logs/metrics on those specific services to confirm the root cause, then apply the correct fix.\n",
                    R_PARTIAL_FIX,
                    False,
                )
        else:
            self._wrong_fixes_applied.append(fix_key)
            return (
                f"[FAILED] {action_type.replace('_', ' ').title()} on {service}. No improvement observed.\n"
                f"Incident still active. Is this really the root cause?\n",
                P_WRONG_FIX,
                False,
            )

    def _handle_diagnose(self, hypothesis: str):
        sc = self._scenario
        hypothesis_lower = hypothesis.lower()

        # Anti-reward-hacking gate: require at least one service to have been investigated
        # before a diagnosis can earn reward.  This prevents a model from collecting +20 by
        # keyword-stuffing without doing any actual investigation.  We check _queried_services
        # (any service) rather than services_investigated (root-cause only) so that false-alarm
        # scenarios — which have no root-cause services — can still be diagnosed correctly after
        # the agent inspects the logs.
        if not self._queried_services:
            return (
                "[DIAGNOSIS REJECTED] You have not investigated any services yet.\n"
                "Use query_logs or check_metrics on the affected services first, then diagnose.\n"
                "No reward awarded.",
                0.0,
            )

        # Check each diagnosis group independently (supports multi-root-cause scenarios)
        for i, group in enumerate(sc.correct_diagnosis_groups):
            if any(kw in hypothesis_lower for kw in group):
                if i not in self._diagnosis_groups_awarded:
                    self._diagnosis_groups_awarded.add(i)
                    # correct_diagnosis_given is True when ALL groups are awarded
                    self._state.correct_diagnosis_given = (
                        len(self._diagnosis_groups_awarded) == len(sc.correct_diagnosis_groups)
                    )
                    groups_remaining = len(sc.correct_diagnosis_groups) - len(self._diagnosis_groups_awarded)
                    extra = (
                        f" ({groups_remaining} more root cause(s) to identify.)"
                        if groups_remaining > 0 else ""
                    )
                    return (
                        f"[DIAGNOSIS ACCEPTED] '{hypothesis}'\n"
                        f"Matches root cause group {i + 1}/{len(sc.correct_diagnosis_groups)}. "
                        f"+{R_CORRECT_DIAGNOSIS:.0f} reward.{extra}\n"
                        f"Now apply the correct remediation action.",
                        R_CORRECT_DIAGNOSIS,
                    )
                else:
                    return (
                        f"[DIAGNOSIS] Root cause group {i + 1} already recorded. Focus on remediation.",
                        0.0,
                    )

        return (
            f"[DIAGNOSIS] '{hypothesis}' — does not match any root cause. No reward. Keep investigating.",
            0.0,
        )

    def _handle_escalate(self, reason: str):
        sc = self._scenario
        # Use the scenario's escalation_correct field instead of hardcoded IDs
        if not sc.escalation_correct:
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
        sc = self._scenario
        report_lower = report.lower()

        # Scenario 6 (false alarm): reward if no wrong fixes were applied
        if not sc.correct_fix_actions:
            if not self._wrong_fixes_applied and self._state.correct_diagnosis_given:
                # Perfect false-alarm resolution: diagnosed correctly, no unnecessary actions
                all_kws = [kw for group in sc.correct_diagnosis_groups for kw in group]
                matched = sum(1 for kw in all_kws if kw in report_lower)
                quality_bonus = min(matched * 1.5, 10.0)
                total = R_GOOD_POSTMORTEM + R_NO_ACTION_NEEDED + quality_bonus
                self._state.resolved = True
                return (
                    f"[POST-MORTEM SUBMITTED]\nCorrectly identified as a false alarm with no unnecessary actions.\n"
                    f"Quality bonus: +{quality_bonus:.1f} | No-action bonus: +{R_NO_ACTION_NEEDED:.0f}\n"
                    f"{report}\n\nIncident closed. +{total:.1f} total reward.",
                    total,
                    True,
                )
            elif self._wrong_fixes_applied:
                return (
                    f"[POST-MORTEM — PENALISED]\nUnnecessary fix actions were applied to a healthy system.\n"
                    f"{report}\n\nIncident closed with penalty: {P_UNRESOLVED_END:.0f}.",
                    P_UNRESOLVED_END,
                    True,
                )
            else:
                # Submitted postmortem without diagnosing first
                return (
                    f"[POST-MORTEM — INCOMPLETE]\nDiagnose the root cause before submitting a postmortem.\n"
                    f"Penalty: {P_UNRESOLVED_END:.0f}.",
                    P_UNRESOLVED_END,
                    True,
                )

        if self._state.resolved:
            # Score quality by matching postmortem text against diagnosis keywords
            all_kws = [kw for group in sc.correct_diagnosis_groups for kw in group]
            matched = sum(1 for kw in all_kws if kw in report_lower)
            quality_bonus = min(matched * 1.5, 10.0)
            total = R_GOOD_POSTMORTEM + quality_bonus
            return (
                f"[POST-MORTEM SUBMITTED]\nQuality bonus: +{quality_bonus:.1f}\n"
                f"{report}\n\nIncident closed. +{total:.1f} reward.",
                total,
                True,
            )
        return (
            f"[POST-MORTEM — PREMATURE]\n{report}\n\nIncident not yet resolved. Penalty: {P_UNRESOLVED_END:.0f}.",
            P_UNRESOLVED_END,
            True,
        )

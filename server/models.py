"""
models.py — Pydantic models for the DevOps Incident environment.

Confirmed API (from inspecting the installed openenv-core package):
  - Action, Observation, State are all Pydantic BaseModel subclasses
  - Action base has:      metadata: Dict[str, Any],  extra="forbid"
  - Observation base has: done: bool, reward: float|None, metadata: Dict,  extra="forbid"
  - State base has:       episode_id: Optional[str], step_count: int,  extra="allow"
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core import Action, Observation, State


# ---------------------------------------------------------------------------
# 1. IncidentAction — what the LLM agent sends each turn
# ---------------------------------------------------------------------------
# Valid action_type values:
#   "query_logs"             — see log entries for a specific service
#   "check_metrics"          — see CPU/memory/latency/error_rate for a service
#   "restart_service"        — restart a named service
#   "rollback_deployment"    — roll back last deployment for a service
#   "rollback_config"        — roll back last config change for a service
#   "enable_circuit_breaker" — prevent cascade by isolating a service
#   "enable_rate_limiting"   — throttle traffic to a service (DDoS)
#   "diagnose"               — agent states its root cause hypothesis
#   "escalate"               — hand off to a human on-call
#   "write_postmortem"       — terminal action: submit incident report

class IncidentAction(Action):
    action_type: str = Field(description="One of the valid action type strings above")
    service: str = Field(default="", description="Target service name (if applicable)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Extra params e.g. replicas")
    reasoning: str = Field(default="", description="Agent's explanation for this action")
    # metadata: Dict[str, Any] — inherited from Action base


# ---------------------------------------------------------------------------
# 2. IncidentObservation — what the agent receives back each turn
# ---------------------------------------------------------------------------

class IncidentObservation(Observation):
    # done: bool          — inherited (True when episode ends)
    # reward: float|None  — inherited (step reward)
    # metadata: dict      — inherited (carries total_reward etc.)

    action_result: str = Field(default="", description="Human-readable result of the last action")
    service_dashboard: str = Field(default="", description="One-line status table of all services")
    active_alerts: List[str] = Field(default_factory=list, description="Currently firing alert strings")
    steps_remaining: int = Field(default=0, description="Steps left before episode auto-fails")
    incident_resolved: bool = Field(default=False, description="Whether the incident is fully resolved")
    step_reward: float = Field(default=0.0, description="Reward earned on this specific step")
    hint: Optional[str] = Field(default=None, description="Guidance hint (only on Scenario 1)")


# ---------------------------------------------------------------------------
# 3. IncidentState — episode metadata used by training frameworks
# ---------------------------------------------------------------------------

class IncidentState(State):
    # episode_id: Optional[str] — inherited
    # step_count: int           — inherited

    scenario_id: int = Field(default=1)
    scenario_name: str = Field(default="")
    difficulty: float = Field(default=0.1, description="D:0.0 to D:1.0")
    total_reward: float = Field(default=0.0, description="Cumulative reward across all steps")
    resolved: bool = Field(default=False)
    services_investigated: int = Field(default=0, description="Unique root-cause services the agent looked at")
    correct_diagnosis_given: bool = Field(default=False)

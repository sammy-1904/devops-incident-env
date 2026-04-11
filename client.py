"""
client.py — WebSocket client for the DevOps Incident environment.

Confirmed API (from inspecting openenv-core):
  - Base class: EnvClient (from openenv.core)
  - StepResult: from openenv.core.client_types
  - Must implement: _step_payload(), _parse_result(), _parse_state()

Usage:
    # Async (recommended for training):
    async with IncidentEnv(base_url="http://localhost:8000") as client:
        result = await client.reset()
        result = await client.step(IncidentAction(action_type="query_logs", service="user-service"))
        print(result.observation.action_result)

    # Sync (for inference script):
    with IncidentEnv(base_url="http://localhost:8000").sync() as client:
        result = client.reset()
        result = client.step(IncidentAction(action_type="query_logs", service="user-service"))
"""

from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from server.models import IncidentAction, IncidentObservation, IncidentState


class IncidentEnv(EnvClient[IncidentAction, IncidentObservation, IncidentState]):
    """
    WebSocket client for the DevOps Incident Management environment.
    The base class handles all connection management and message framing.
    We only implement serialization/deserialization.
    """

    # -----------------------------------------------------------------------
    # Serialization: IncidentAction → JSON dict  (sent over WebSocket)
    # -----------------------------------------------------------------------

    def _step_payload(self, action: IncidentAction) -> Dict[str, Any]:
        return {
            "action_type": action.action_type,
            "service":     action.service,
            "hypothesis":  action.hypothesis,
            "parameters":  action.parameters,
            "reasoning":   action.reasoning,
            "metadata":    action.metadata,
        }

    # -----------------------------------------------------------------------
    # Deserialization: JSON response → StepResult[IncidentObservation]
    # -----------------------------------------------------------------------

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[IncidentObservation]:
        # The server response is: {"observation": {...}, "reward": ..., "done": ...}
        # Our custom fields live inside "observation"
        obs_data = payload.get("observation", payload)

        obs = IncidentObservation(
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            metadata=obs_data.get("metadata", {}),
            action_result=obs_data.get("action_result", ""),
            service_dashboard=obs_data.get("service_dashboard", ""),
            active_alerts=obs_data.get("active_alerts", []),
            steps_remaining=obs_data.get("steps_remaining", 0),
            incident_resolved=obs_data.get("incident_resolved", False),
            step_reward=obs_data.get("step_reward", 0.0),
            hint=obs_data.get("hint"),
            recent_deployments=obs_data.get("recent_deployments", []),
        )
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    # -----------------------------------------------------------------------
    # Deserialization: JSON → IncidentState  (from GET /state)
    # -----------------------------------------------------------------------

    def _parse_state(self, payload: Dict[str, Any]) -> IncidentState:
        return IncidentState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            scenario_id=payload.get("scenario_id", 1),
            scenario_name=payload.get("scenario_name", ""),
            difficulty=payload.get("difficulty", 0.1),
            total_reward=payload.get("total_reward", 0.0),
            resolved=payload.get("resolved", False),
            services_investigated=payload.get("services_investigated", 0),
            correct_diagnosis_given=payload.get("correct_diagnosis_given", False),
        )

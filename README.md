---
title: DevOps Incident Management Environment
emoji: 🔧
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - rl-environment
  - devops
  - agent-environment
base_path: /web
---

# DevOps Incident Management Environment

A production-ready OpenEnv-compliant reinforcement learning environment for training AI agents in Site Reliability Engineering (SRE) tasks. The environment simulates real-world DevOps incidents, challenging frontier models to investigate telemetry, diagnose root causes, and apply targeted fixes — without causing further outages.

## Overview

An agent assumes the role of an on-call SRE. When an alert fires, the agent must:

1. **Investigate** — Query logs and check metrics for affected services.
2. **Diagnose** — Formulate a root cause hypothesis with `diagnose()`.
3. **Remediate** — Apply the correct fix (restart, rollback, rate-limit, circuit-break).
4. **Post-Incident** — Submit a post-mortem once resolved (or after a false alarm).

### 6 Progressive Scenarios

| # | Name | Difficulty | Key Skill |
|---|------|------------|-----------|
| 1 | Single OOM Crash | 0.1 | Identify and restart a crashed service |
| 2 | Bad Deployment | 0.3 | Correlate a deployment event with errors, rollback |
| 3 | DB Connection Pool Exhaustion | 0.5 | Diagnose shared-resource cascade, multi-fix |
| 4 | Cascading Microservice Failure | 0.75 | Find upstream OOM root cause, circuit-break downstream |
| 5 | Multi-Incident Surge | 1.0 | Resolve two simultaneous independent incidents (DDoS + config drift) |
| 6 | False Alarm: Planned Load Test | 0.2 | Identify expected traffic, no fix needed |

Each scenario contains **red herring services** with misleading symptoms to prevent lucky guessing.

---

## Reward Structure

| Action | Reward |
|--------|--------|
| First look at the root-cause service | +5 |
| Correct diagnosis (per root cause) | +20 |
| Correct fix — partial (multi-fix scenarios) | +10 |
| Correct fix — final (incident resolved) | +30 |
| False alarm correctly identified (no fix applied) | +20 bonus |
| Post-mortem submitted after resolution | +5 base + up to +10 quality bonus |
| Efficiency bonus (steps saved on resolution) | up to +10 |
| Reasoning field > 30 chars | +1 per step |
| Wrong fix applied | −10 |
| Repeated query on same service (≥3x) | −2 |
| Incorrect escalation | −15 |
| Max steps exceeded | −20 |
| Premature post-mortem | −10 |

**Score normalisation:** `score = clamp(total_reward / scenario_max_reward, 0.001, 0.999)`
Each scenario has its own `max_reward` ceiling to ensure score diversity across scenarios.

---

## How to Run

### 1. Build and Run the Environment (Docker)

```bash
docker build -t devops-incident-env .
docker run -d -p 8000:8000 --name devops-incident-env devops-incident-env
```

Test it's alive:
```bash
curl http://localhost:8000/health
# {"status": "healthy"}
```

### 2. Install the Client Package

```bash
pip install git+https://huggingface.co/spaces/sammy-1904/devops-incident-env
```

### 3. Run the Inference Baseline

Create a `.env` file:
```env
HF_TOKEN=hf_your_token_here
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

Run all 6 scenarios:
```bash
python inference.py
```

Expected output format:
```
[START] task=devops-incident-scenario1 env=devops-incident-env model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=query_logs(service='user-service') reward=5.00 done=false error=null
[STEP] step=2 action=check_metrics(service='user-service') reward=1.00 done=false error=null
[STEP] step=3 action=diagnose(root_cause='OOMKilled: memory limit exceeded') reward=21.00 done=false error=null
[STEP] step=4 action=restart_service(service='user-service') reward=30.00 done=true error=null
[END] success=true steps=4 score=0.701 rewards=5.00,1.00,21.00,30.00
```

### 4. Connect to the Live HF Space

```bash
ENV_BASE_URL=https://sammy-1904-devops-incident-env.hf.space python inference.py
```

- **Web UI**: https://sammy-1904-devops-incident-env.hf.space/web
- **Health**: https://sammy-1904-devops-incident-env.hf.space/health
- **API Docs**: https://sammy-1904-devops-incident-env.hf.space/docs

### 5. Validate with OpenEnv CLI

```bash
openenv validate
```

---

## Environment API

### Actions

```python
from devops_incident_env import IncidentEnv, IncidentAction

async with IncidentEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(scenario_id=1)          # start Scenario 1
    result = await env.step(IncidentAction(
        action_type="query_logs",
        service="user-service",
        reasoning="Check the service mentioned in the alert first",
    ))
    print(result.observation.action_result)          # log output
    print(result.observation.service_dashboard)      # live service table
    print(result.observation.active_alerts)          # firing alerts
    print(result.observation.steps_remaining)        # budget left
    print(result.observation.recent_deployments)     # deployment events
```

### Available `action_type` values

| action_type | Required fields | Effect |
|-------------|----------------|--------|
| `query_logs` | `service` | Returns last 30 min of logs |
| `check_metrics` | `service` | Returns CPU, memory, error rate, latency |
| `restart_service` | `service` | Restarts the named service |
| `rollback_deployment` | `service` | Reverts last deployment |
| `rollback_config` | `service` | Reverts last config change |
| `enable_circuit_breaker` | `service` | Isolates service from upstream |
| `enable_rate_limiting` | `service` | Throttles inbound traffic |
| `diagnose` | `hypothesis` | Scores root cause hypothesis |
| `write_postmortem` | `reasoning` | Closes the incident |
| `escalate` | `reasoning` | Hands off to human (penalised on solvable scenarios) |

---

## Architecture

```
devops-incident-env/
├── server/
│   ├── app.py           # FastAPI + WebSocket server (OpenEnv spec)
│   ├── environment.py   # Episode logic, reward shaping
│   ├── models.py        # IncidentAction, IncidentObservation, IncidentState
│   └── Dockerfile
├── data/
│   └── scenarios.py     # All 6 ground-truth scenario definitions
├── client.py            # WebSocket client (EnvClient implementation)
├── models.py            # Root-level re-export for pip install consumers
├── inference.py         # Hackathon-compliant baseline inference script
├── openenv.yaml         # HF Spaces deployment manifest
└── pyproject.toml
```

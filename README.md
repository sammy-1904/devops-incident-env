---
title: DevOps Incident Management Environment
emoji: 🔧
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - rl-environment
  - devops
  - agent-environment
base_path: /web
---

# DevOps Incident Management Environment

A production-ready OpenEnv-compliant reinforcement learning environment for training AI agents in Site Reliability Engineering (SRE) tasks. The environment simulates real-world devops incidents, challenging frontier models to investigate telemetry, diagnose root causes, and apply fixes without causing further outages.

## Overview

In this environment, an agent assumes the role of an SRE on-call. When an alert fires, the agent must efficiently:
1. **Investigate**: Query logs and check metrics for affected services.
2. **Diagnose**: Formulate a definitive root cause hypothesis.
3. **Remediate**: Select the correct restorative action (e.g., restart service, roll back deployment, enable circuit breaker) based on the gathered evidence.
4. **Post-Incident**: Submit a post-mortem once the issue is fully resolved.

### Scenarios

5 deterministic scenarios validate agent performance progressively:
- **Scenario 1 (Difficulty 0.1)**: Single OOM Crash (Restart Service)
- **Scenario 2 (Difficulty 0.3)**: Bad Deployment (Rollback Deployment)
- **Scenario 3 (Difficulty 0.5)**: DB Connection Pool Exhaustion (Scale/Restart upstream)
- **Scenario 4 (Difficulty 0.75)**: Cascading Microservice Failure (Root cause + Circuit breaker)
- **Scenario 5 (Difficulty 1.0)**: Multi-Incident Surge (DDoS + Config Drift simultaneously)

## How to Run

### 1. Build and Run the Environment (Docker)

```bash
docker build -t devops-incident-env .
docker run -d -p 8000:8000 --name devops-incident-env devops-incident-env
```

### 2. Run Inference Baseline

The inference script uses the OpenAI API client to process scenarios automatically using your chosen Hugging Face endpoint.

1. Generate a Hugging Face API Token.
2. In `.env`, add:
   ```env
   HF_TOKEN=hf_your_token_here
   ```
3. Run the evaluation script:
   ```bash
   python inference.py
   ```

### 3. Connect to the Live HF Space (no Docker needed)

The environment is deployed at:
- **Web UI**: https://sammy-1904-devops-incident-env.hf.space/web
- **Health**: https://sammy-1904-devops-incident-env.hf.space/health
- **API Docs**: https://sammy-1904-devops-incident-env.hf.space/docs

Run inference against the live Space:
```bash
ENV_BASE_URL=https://sammy-1904-devops-incident-env.hf.space python inference.py
```

Or pull and run the Docker image locally from the HF registry:
```bash
docker pull registry.hf.space/sammy-1904-devops-incident-env:latest
docker run -d -p 8000:8000 registry.hf.space/sammy-1904-devops-incident-env:latest
python inference.py
```

## Environment Design & Rewards

The reward landscape is shaped using intermediate feedback for better Reinforcement Learning mapping:
* **`+5`**   Investigating the correct hidden root-cause service.
* **`+20`**  Stating an accurate diagnosis.
* **`+30`**  Applying a full correct fix.
* **`-10`**  Applying the wrong fix.
* **`-15`**  Incorrectly escalating an issue.
* **Bonus**: Efficiency multiplier computed based on time-to-resolve vs max-allowed steps.

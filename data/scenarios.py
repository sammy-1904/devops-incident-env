"""
scenarios.py — Ground truth definitions for all 5 incident scenarios.

This file has ZERO OpenEnv dependencies. It is pure Python data.
The environment.py imports from here to set up each episode.

Each scenario contains:
  - The initial alert (what the agent sees first)
  - Service topology with pre-generated logs and metrics
  - Ground truth root cause(s) — NEVER shown to the agent
  - Correct fix actions — used by grader to assign reward
  - Red herring services — services with symptoms that are NOT the root cause
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Low-level data types
# ---------------------------------------------------------------------------

@dataclass
class ServiceState:
    name: str
    status: str                  # "healthy" | "degraded" | "down"
    cpu_pct: float
    memory_pct: float
    error_rate_pct: float        # 0–100
    p99_latency_ms: float
    is_root_cause: bool = False  # hidden from agent
    is_red_herring: bool = False


@dataclass
class LogEntry:
    timestamp: str               # e.g. "2024-01-15 14:32:01"
    service: str
    level: str                   # DEBUG | INFO | WARN | ERROR | FATAL
    message: str


@dataclass
class DeploymentEvent:
    timestamp: str
    service: str
    old_version: str
    new_version: str
    deployed_by: str


@dataclass
class Scenario:
    id: int
    name: str
    difficulty: float            # 0.0 – 1.0 as required by the spec
    alert_message: str           # what fires and wakes up the on-call SRE
    services: Dict[str, ServiceState]
    logs: List[LogEntry]         # all logs in the system (agent retrieves by service)
    deployment_events: List[DeploymentEvent]
    root_cause_services: List[str]   # HIDDEN — used for grading
    correct_fix_actions: List[str]   # HIDDEN — e.g. ["restart_service:user-service"]
    correct_diagnoses: List[str]     # HIDDEN — root cause text agent should say
    max_steps: int
    hint: Optional[str] = None   # only non-None on scenario 1 (easiest)


# ---------------------------------------------------------------------------
# Scenario 1 — Single OOM Crash  (D: 0.1)
# ---------------------------------------------------------------------------

SCENARIO_1 = Scenario(
    id=1,
    name="Single OOM Crash",
    difficulty=0.1,
    alert_message=(
        "[CRITICAL] user-service is not responding. "
        "Health check has failed 3 consecutive times. "
        "Alert fired at 2024-01-15 14:35:00 UTC. "
        "On-call engineer paged. Acknowledge and investigate."
    ),
    services={
        "api-gateway": ServiceState(
            name="api-gateway", status="degraded",
            cpu_pct=45.0, memory_pct=38.0,
            error_rate_pct=28.0, p99_latency_ms=1200.0,
        ),
        "user-service": ServiceState(
            name="user-service", status="down",
            cpu_pct=0.0, memory_pct=0.0,
            error_rate_pct=100.0, p99_latency_ms=0.0,
            is_root_cause=True,
        ),
        "postgres-db": ServiceState(
            name="postgres-db", status="healthy",
            cpu_pct=22.0, memory_pct=41.0,
            error_rate_pct=0.0, p99_latency_ms=12.0,
        ),
    },
    logs=[
        LogEntry("2024-01-15 14:30:11", "user-service", "WARN",  "Memory usage at 89% (limit: 512Mi)"),
        LogEntry("2024-01-15 14:31:44", "user-service", "WARN",  "Memory usage at 96% (limit: 512Mi)"),
        LogEntry("2024-01-15 14:32:01", "user-service", "FATAL", "Killed by signal 9 (OOMKilled). Memory limit exceeded: 512Mi"),
        LogEntry("2024-01-15 14:32:05", "api-gateway",  "ERROR", "upstream user-service connection refused (connection reset by peer)"),
        LogEntry("2024-01-15 14:32:07", "api-gateway",  "ERROR", "upstream user-service connection refused (connection reset by peer)"),
        LogEntry("2024-01-15 14:34:55", "api-gateway",  "ERROR", "health check FAILED for user-service: dial tcp: connection refused"),
        LogEntry("2024-01-15 14:35:00", "api-gateway",  "ERROR", "health check FAILED for user-service (3/3). Marking unhealthy."),
        LogEntry("2024-01-15 14:22:00", "postgres-db",  "INFO",  "Checkpoint completed: wrote 34 buffers, 0.1% of total"),
    ],
    deployment_events=[],
    root_cause_services=["user-service"],
    correct_fix_actions=["restart_service:user-service"],
    correct_diagnoses=["oom", "out of memory", "memory limit", "oomkilled"],
    max_steps=8,
    hint="Tip: Start by querying logs for the service mentioned in the alert, then check its metrics.",
)


# ---------------------------------------------------------------------------
# Scenario 2 — Bad Deployment  (D: 0.3)
# ---------------------------------------------------------------------------

SCENARIO_2 = Scenario(
    id=2,
    name="Bad Deployment",
    difficulty=0.3,
    alert_message=(
        "[HIGH] api-gateway p99 latency has exceeded 2000ms for the past 5 minutes. "
        "Error rate: 22%. Alert fired at 2024-01-15 16:10:00 UTC. "
        "A deployment occurred recently. Investigate and resolve."
    ),
    services={
        "api-gateway": ServiceState(
            name="api-gateway", status="degraded",
            cpu_pct=62.0, memory_pct=44.0,
            error_rate_pct=22.0, p99_latency_ms=2800.0,
        ),
        "order-service": ServiceState(
            name="order-service", status="degraded",
            cpu_pct=55.0, memory_pct=51.0,
            error_rate_pct=65.0, p99_latency_ms=3100.0,
            is_root_cause=True,
        ),
        "notification-service": ServiceState(
            name="notification-service", status="degraded",
            cpu_pct=88.0, memory_pct=42.0,   # high CPU — red herring (batch job)
            error_rate_pct=2.0, p99_latency_ms=95.0,
            is_red_herring=True,
        ),
        "auth-service": ServiceState(
            name="auth-service", status="healthy",
            cpu_pct=18.0, memory_pct=29.0,
            error_rate_pct=0.0, p99_latency_ms=45.0,
        ),
        "user-service": ServiceState(
            name="user-service", status="healthy",
            cpu_pct=31.0, memory_pct=38.0,
            error_rate_pct=0.0, p99_latency_ms=78.0,
        ),
    },
    logs=[
        LogEntry("2024-01-15 16:00:03", "order-service",       "INFO",  "Deployment started: 5.2.1 → 5.3.0 by jenkins-ci"),
        LogEntry("2024-01-15 16:00:41", "order-service",       "INFO",  "Deployment completed: running version 5.3.0"),
        LogEntry("2024-01-15 16:01:14", "order-service",       "ERROR", "NullPointerException at OrderProcessor.calculateDiscount(OrderProcessor.java:247)"),
        LogEntry("2024-01-15 16:01:15", "order-service",       "ERROR", "NullPointerException at OrderProcessor.calculateDiscount(OrderProcessor.java:247)"),
        LogEntry("2024-01-15 16:01:22", "order-service",       "ERROR", "Unhandled exception in request handler. Returning 500."),
        LogEntry("2024-01-15 16:02:10", "api-gateway",         "WARN",  "upstream order-service response time: 3100ms (threshold: 500ms)"),
        LogEntry("2024-01-15 16:05:00", "api-gateway",         "ERROR", "Circuit breaker OPEN for order-service (error rate 65%)"),
        LogEntry("2024-01-15 16:10:00", "api-gateway",         "ERROR", "p99 latency threshold breached: 2800ms. Firing alert."),
        LogEntry("2024-01-15 15:55:00", "notification-service", "INFO", "Scheduled batch job started: digest-email-sender"),
        LogEntry("2024-01-15 16:08:00", "notification-service", "INFO", "Batch job in progress: 45,000/120,000 emails processed"),
        LogEntry("2024-01-15 16:09:00", "auth-service",        "INFO",  "Processed 1240 authentication requests. All successful."),
    ],
    deployment_events=[
        DeploymentEvent("2024-01-15 16:00:03", "order-service", "5.2.1", "5.3.0", "jenkins-ci"),
    ],
    root_cause_services=["order-service"],
    correct_fix_actions=["rollback_deployment:order-service"],
    correct_diagnoses=["bad deployment", "nullpointerexception", "order-service deployment",
                       "version 5.3.0", "5.3.0", "rollback", "circuit breaker", "error rate",
                       "deployment issue", "high latency", "order service"],
    max_steps=12,
)


# ---------------------------------------------------------------------------
# Scenario 3 — DB Connection Pool Exhaustion  (D: 0.5)
# ---------------------------------------------------------------------------

SCENARIO_3 = Scenario(
    id=3,
    name="DB Connection Pool Exhaustion",
    difficulty=0.5,
    alert_message=(
        "[CRITICAL] Multiple services returning 503 Service Unavailable. "
        "Affected: auth-service (error_rate=98%), order-service (error_rate=95%), user-service (error_rate=91%). "
        "Alert fired at 2024-01-15 18:45:00 UTC. All three services are degraded simultaneously."
    ),
    services={
        "api-gateway": ServiceState(
            name="api-gateway", status="degraded",
            cpu_pct=71.0, memory_pct=48.0,
            error_rate_pct=88.0, p99_latency_ms=5200.0,
        ),
        "auth-service": ServiceState(
            name="auth-service", status="down",
            cpu_pct=12.0, memory_pct=35.0,
            error_rate_pct=98.0, p99_latency_ms=30000.0,
        ),
        "order-service": ServiceState(
            name="order-service", status="down",
            cpu_pct=9.0, memory_pct=31.0,
            error_rate_pct=95.0, p99_latency_ms=30000.0,
        ),
        "user-service": ServiceState(
            name="user-service", status="degraded",
            cpu_pct=18.0, memory_pct=44.0,
            error_rate_pct=91.0, p99_latency_ms=28000.0,
            is_root_cause=True,  # connection leak source
        ),
        "postgres-db": ServiceState(
            name="postgres-db", status="degraded",
            cpu_pct=85.0, memory_pct=78.0,
            error_rate_pct=0.0, p99_latency_ms=450.0,
            is_root_cause=True,
        ),
        "redis-cache": ServiceState(
            name="redis-cache", status="degraded",
            cpu_pct=41.0, memory_pct=82.0,   # elevated memory — red herring
            error_rate_pct=3.0, p99_latency_ms=8.0,
            is_red_herring=True,
        ),
    },
    logs=[
        LogEntry("2024-01-15 18:30:01", "user-service",  "DEBUG", "DB connection acquired from pool [pool_size=100, active=97]"),
        LogEntry("2024-01-15 18:31:12", "user-service",  "DEBUG", "DB connection acquired from pool [pool_size=100, active=99]"),
        LogEntry("2024-01-15 18:31:55", "user-service",  "DEBUG", "DB connection acquired from pool [pool_size=100, active=100]"),
        LogEntry("2024-01-15 18:32:00", "postgres-db",   "FATAL", "FATAL: remaining connection slots are reserved for non-replication superuser connections"),
        LogEntry("2024-01-15 18:32:01", "postgres-db",   "ERROR", "connection pool exhausted: max_connections=100, active=100"),
        LogEntry("2024-01-15 18:32:03", "auth-service",  "ERROR", "ConnectionPoolTimeoutError: QueuePool limit of size 20 overflow 10 reached, connection timed out"),
        LogEntry("2024-01-15 18:32:03", "order-service", "ERROR", "ConnectionPoolTimeoutError: QueuePool limit of size 20 overflow 10 reached, connection timed out"),
        LogEntry("2024-01-15 18:32:05", "auth-service",  "ERROR", "Returning 503: unable to acquire database connection"),
        LogEntry("2024-01-15 18:32:05", "order-service", "ERROR", "Returning 503: unable to acquire database connection"),
        LogEntry("2024-01-15 18:44:00", "redis-cache",   "WARN",  "Memory usage at 82% (maxmemory-policy: allkeys-lru)"),
        LogEntry("2024-01-15 18:44:30", "redis-cache",   "INFO",  "Evicted 1240 keys using allkeys-lru policy"),
        LogEntry("2024-01-15 18:45:00", "api-gateway",   "ERROR", "All backend services reporting errors. Firing critical alert."),
    ],
    deployment_events=[],
    root_cause_services=["postgres-db", "user-service"],
    correct_fix_actions=[
        "restart_service:postgres-db",
        "restart_service:user-service",
    ],
    correct_diagnoses=["connection pool", "db connection", "connection exhausted", "postgres",
                       "connection leak", "timeout", "cannot acquire", "database connection",
                       "pool exhausted", "db pool", "connection timeout"],
    max_steps=15,
)


# ---------------------------------------------------------------------------
# Scenario 4 — Cascading Microservice Failure  (D: 0.75)
# ---------------------------------------------------------------------------

SCENARIO_4 = Scenario(
    id=4,
    name="Cascading Microservice Failure",
    difficulty=0.75,
    alert_message=(
        "[CRITICAL] auth-service and order-service both reporting 100% error rate. "
        "Alert fired at 2024-01-15 20:15:00 UTC. "
        "Downstream impact: API gateway unavailable. Customer-facing features offline. "
        "Multiple alerts firing simultaneously. Investigate urgently."
    ),
    services={
        "api-gateway": ServiceState(
            name="api-gateway", status="down",
            cpu_pct=5.0, memory_pct=22.0,
            error_rate_pct=100.0, p99_latency_ms=0.0,
        ),
        "auth-service": ServiceState(
            name="auth-service", status="down",
            cpu_pct=8.0, memory_pct=28.0,
            error_rate_pct=100.0, p99_latency_ms=0.0,
        ),
        "order-service": ServiceState(
            name="order-service", status="down",
            cpu_pct=6.0, memory_pct=24.0,
            error_rate_pct=100.0, p99_latency_ms=0.0,
        ),
        "user-service": ServiceState(
            name="user-service", status="down",
            cpu_pct=0.0, memory_pct=0.0,
            error_rate_pct=100.0, p99_latency_ms=0.0,
            is_root_cause=True,
        ),
        "notification-service": ServiceState(
            name="notification-service", status="degraded",
            cpu_pct=35.0, memory_pct=48.0,
            error_rate_pct=40.0, p99_latency_ms=850.0,
        ),
        "redis-cache": ServiceState(
            name="redis-cache", status="degraded",
            cpu_pct=55.0, memory_pct=71.0,  # looks suspicious — red herring
            error_rate_pct=8.0, p99_latency_ms=22.0,
            is_red_herring=True,
        ),
        "postgres-db": ServiceState(
            name="postgres-db", status="healthy",
            cpu_pct=19.0, memory_pct=42.0,
            error_rate_pct=0.0, p99_latency_ms=14.0,
        ),
    },
    logs=[
        LogEntry("2024-01-15 20:10:01", "user-service",       "WARN",  "Memory usage at 91% (limit: 1Gi)"),
        LogEntry("2024-01-15 20:11:33", "user-service",       "FATAL", "Killed by signal 9 (OOMKilled). Memory limit exceeded: 1Gi"),
        LogEntry("2024-01-15 20:11:35", "auth-service",       "ERROR", "gRPC connection to user-service failed: connection refused"),
        LogEntry("2024-01-15 20:11:35", "order-service",      "ERROR", "gRPC connection to user-service failed: connection refused"),
        LogEntry("2024-01-15 20:11:36", "auth-service",       "ERROR", "Unable to validate user session: upstream unavailable. Returning 503."),
        LogEntry("2024-01-15 20:11:36", "order-service",      "ERROR", "Unable to fetch user profile: upstream unavailable. Returning 503."),
        LogEntry("2024-01-15 20:11:40", "api-gateway",        "ERROR", "Upstream auth-service: 503. Upstream order-service: 503."),
        LogEntry("2024-01-15 20:12:00", "notification-service","WARN", "Failed to send 340 notifications (user-service unreachable for preferences fetch)"),
        LogEntry("2024-01-15 20:13:00", "redis-cache",        "WARN",  "Eviction rate elevated: 2400 keys/min (normal: ~200 keys/min)"),
        LogEntry("2024-01-15 20:14:00", "redis-cache",        "WARN",  "Memory at 71%. LRU eviction active."),
        LogEntry("2024-01-15 20:15:00", "api-gateway",        "ERROR", "auth-service 100% error rate for 3min 20sec. Firing CRITICAL alert."),
    ],
    deployment_events=[],
    root_cause_services=["user-service"],
    correct_fix_actions=[
        "restart_service:user-service",
        "enable_circuit_breaker:order-service",
    ],
    correct_diagnoses=["oom", "out of memory", "user-service", "cascade", "cascading failure",
                       "upstream", "memory", "oomkill", "memory limit", "circuit",
                       "downstream", "service failure", "upstream failure"],
    max_steps=18,
)


# ---------------------------------------------------------------------------
# Scenario 5 — Multi-Incident Surge  (D: 1.0)
# ---------------------------------------------------------------------------

SCENARIO_5 = Scenario(
    id=5,
    name="Multi-Incident Surge",
    difficulty=1.0,
    alert_message=(
        "[CRITICAL] MULTIPLE SIMULTANEOUS INCIDENTS DETECTED. "
        "(1) api-gateway error rate: 95% — possible DDoS. "
        "(2) auth-service JWT validation failures: 100% — ALL user sessions being rejected. "
        "Alert fired at 2024-01-15 22:30:00 UTC. "
        "Customer impact: complete service outage. Both issues must be resolved independently."
    ),
    services={
        "api-gateway": ServiceState(
            name="api-gateway", status="degraded",
            cpu_pct=98.0, memory_pct=76.0,
            error_rate_pct=95.0, p99_latency_ms=8500.0,
            is_root_cause=True,  # DDoS target
        ),
        "auth-service": ServiceState(
            name="auth-service", status="degraded",
            cpu_pct=44.0, memory_pct=38.0,
            error_rate_pct=100.0, p99_latency_ms=120.0,
            is_root_cause=True,  # config drift
        ),
        "order-service": ServiceState(
            name="order-service", status="down",
            cpu_pct=3.0, memory_pct=20.0,
            error_rate_pct=100.0, p99_latency_ms=0.0,
        ),
        "user-service": ServiceState(
            name="user-service", status="down",
            cpu_pct=4.0, memory_pct=22.0,
            error_rate_pct=100.0, p99_latency_ms=0.0,
        ),
        "inventory-service": ServiceState(
            name="inventory-service", status="degraded",
            cpu_pct=38.0, memory_pct=41.0,
            error_rate_pct=35.0, p99_latency_ms=780.0,
            is_red_herring=True,
        ),
        "postgres-db": ServiceState(
            name="postgres-db", status="healthy",
            cpu_pct=28.0, memory_pct=55.0,
            error_rate_pct=0.0, p99_latency_ms=18.0,
        ),
        "redis-cache": ServiceState(
            name="redis-cache", status="healthy",
            cpu_pct=22.0, memory_pct=44.0,
            error_rate_pct=0.0, p99_latency_ms=3.0,
        ),
    },
    logs=[
        # DDoS evidence on api-gateway
        LogEntry("2024-01-15 22:28:00", "api-gateway",   "WARN",  "Unusually high request rate: 48,000 req/min (normal: 2,000 req/min)"),
        LogEntry("2024-01-15 22:28:30", "api-gateway",   "WARN",  "Single IP 203.0.113.42 responsible for 45,000 req/min"),
        LogEntry("2024-01-15 22:29:00", "api-gateway",   "ERROR", "Rate limit threshold breached. Dropping requests. error_rate rising."),
        LogEntry("2024-01-15 22:29:30", "api-gateway",   "ERROR", "CPU at 98%. Thread pool exhausted. Queuing degraded."),
        # Auth config drift evidence
        LogEntry("2024-01-15 22:25:01", "auth-service",  "INFO",  "Config reload triggered by configmap update: auth-config-v2"),
        LogEntry("2024-01-15 22:25:05", "auth-service",  "WARN",  "JWT_SECRET changed in environment. All active sessions may be invalidated."),
        LogEntry("2024-01-15 22:26:00", "auth-service",  "ERROR", "JWT validation failed: signature verification failed (token=eyJhbGci...)"),
        LogEntry("2024-01-15 22:26:01", "auth-service",  "ERROR", "JWT validation failed: signature verification failed (token=eyJhbGci...)"),
        LogEntry("2024-01-15 22:27:00", "auth-service",  "ERROR", "100% of JWT validation requests failing. JWT_SECRET mismatch suspected."),
        # Downstream cascades
        LogEntry("2024-01-15 22:29:00", "order-service", "ERROR", "Authentication required but auth-service returning 401 for all requests"),
        LogEntry("2024-01-15 22:29:00", "user-service",  "ERROR", "Authentication required but auth-service returning 401 for all requests"),
        # Red herring — inventory degraded due to DDoS flooding shared DB connections
        LogEntry("2024-01-15 22:29:30", "inventory-service","WARN","Database query latency elevated: 780ms (normal: 50ms). Possibly resource contention."),
        LogEntry("2024-01-15 22:30:00", "api-gateway",   "ERROR", "Error rate 95% sustained for 90 seconds. Firing CRITICAL multi-alert."),
    ],
    deployment_events=[
        DeploymentEvent("2024-01-15 22:25:00", "auth-service", "config-v1", "config-v2", "automated-configmap-sync"),
    ],
    root_cause_services=["api-gateway", "auth-service"],
    correct_fix_actions=[
        "enable_rate_limiting:api-gateway",
        "rollback_config:auth-service",
    ],
    correct_diagnoses=[
        "ddos", "rate limit", "ip", "203.0.113",
        "jwt", "jwt_secret", "config drift", "configmap", "auth config",
        "traffic spike", "rate limiting", "misconfiguration", "config",
        "secret", "authentication", "surge",
    ],
    max_steps=22,
)


# ---------------------------------------------------------------------------
# Registry — imported by environment.py
# ---------------------------------------------------------------------------

ALL_SCENARIOS: List[Scenario] = [
    SCENARIO_1,
    SCENARIO_2,
    SCENARIO_3,
    SCENARIO_4,
    SCENARIO_5,
]

SCENARIO_BY_ID: Dict[int, Scenario] = {s.id: s for s in ALL_SCENARIOS}

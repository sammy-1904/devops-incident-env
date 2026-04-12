"""
scenarios.py — Procedurally generated incident scenarios for all 6 types.

Each scenario type is implemented as a generator function _make_scenario_N(rng).
On every reset(), environment.py calls generate_scenario(id, seed) which passes
a seeded random.Random to the appropriate generator — producing a different but
logically consistent scenario instance each time.

Randomised axes per scenario:
  1 (OOM)           root-cause service, memory limit, warn thresholds
  2 (Bad Deploy)    deployed service, version numbers, exception class
  3 (DB Pool)       connection-leak service, pool size, fill percentages
  4 (Cascade)       OOM upstream service, memory limit
  5 (Multi-incident) DDoS IP, config version names, request rate magnitude
  6 (False Alarm)   activity type (load/stress/capacity test), team, rate

This file has ZERO OpenEnv dependencies. It is pure Python data.
"""

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Low-level data types (unchanged)
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
    logs: List[LogEntry]
    deployment_events: List[DeploymentEvent]
    root_cause_services: List[str]
    correct_fix_actions: List[str]
    correct_diagnoses: List[str]
    correct_diagnosis_groups: List[List[str]]
    max_steps: int
    max_reward: float = 70.0
    escalation_correct: bool = False
    hint: Optional[str] = None


# ---------------------------------------------------------------------------
# Service pools used across scenarios
# ---------------------------------------------------------------------------

_OOM_POOL = ["user-service", "order-service", "auth-service", "payment-service"]
_DEPLOY_POOL = ["order-service", "payment-service", "inventory-service", "catalog-service"]
_LEAK_POOL = ["user-service", "order-service", "auth-service"]
_CASCADE_UPSTREAM_POOL = ["user-service", "order-service", "auth-service"]
_ACTIVITY_TYPES = ["load test", "stress test", "capacity test"]
_LOAD_TEST_TEAMS = [
    "load-test-runner@ci-pipeline",
    "perf-team@k8s-job",
    "qa@github-actions",
]
_MEMORY_LIMITS = ["256Mi", "512Mi", "1Gi"]
_EXCEPTION_CLASSES = [
    "NullPointerException",
    "IndexOutOfBoundsException",
    "IllegalStateException",
    "ConcurrentModificationException",
]
_POOL_SIZES = [50, 100, 200]

# ---------------------------------------------------------------------------
# Helper: build a standard healthy / degraded ServiceState quickly
# ---------------------------------------------------------------------------

def _svc(name: str, status: str, cpu: float, mem: float,
         err: float, p99: float,
         root: bool = False, herring: bool = False) -> ServiceState:
    return ServiceState(
        name=name, status=status,
        cpu_pct=cpu, memory_pct=mem,
        error_rate_pct=err, p99_latency_ms=p99,
        is_root_cause=root, is_red_herring=herring,
    )


# ---------------------------------------------------------------------------
# Scenario 1 — Single OOM Crash  (D: 0.1)
# ---------------------------------------------------------------------------

def _make_scenario_1(rng: random.Random) -> Scenario:
    root_svc = rng.choice(_OOM_POOL)
    mem_limit = rng.choice(_MEMORY_LIMITS)
    warn_pct_1 = rng.randint(85, 91)
    warn_pct_2 = rng.randint(93, 98)

    services = {
        "api-gateway": _svc("api-gateway", "degraded",
                            cpu=rng.uniform(38, 55), mem=rng.uniform(30, 45),
                            err=rng.uniform(20, 35), p99=rng.uniform(900, 1500)),
        root_svc: _svc(root_svc, "down",
                       cpu=0.0, mem=0.0, err=100.0, p99=0.0, root=True),
        "postgres-db": _svc("postgres-db", "healthy",
                             cpu=rng.uniform(15, 30), mem=rng.uniform(35, 50),
                             err=0.0, p99=rng.uniform(8, 18)),
    }

    logs = [
        LogEntry("2024-01-15 14:30:11", root_svc, "WARN",
                 f"Memory usage at {warn_pct_1}% (limit: {mem_limit})"),
        LogEntry("2024-01-15 14:31:44", root_svc, "WARN",
                 f"Memory usage at {warn_pct_2}% (limit: {mem_limit})"),
        LogEntry("2024-01-15 14:32:01", root_svc, "FATAL",
                 f"Killed by signal 9 (OOMKilled). Memory limit exceeded: {mem_limit}"),
        LogEntry("2024-01-15 14:32:05", "api-gateway", "ERROR",
                 f"upstream {root_svc} connection refused (connection reset by peer)"),
        LogEntry("2024-01-15 14:32:07", "api-gateway", "ERROR",
                 f"upstream {root_svc} connection refused (connection reset by peer)"),
        LogEntry("2024-01-15 14:34:55", "api-gateway", "ERROR",
                 f"health check FAILED for {root_svc}: dial tcp: connection refused"),
        LogEntry("2024-01-15 14:35:00", "api-gateway", "ERROR",
                 f"health check FAILED for {root_svc} (3/3). Marking unhealthy."),
        LogEntry("2024-01-15 14:22:00", "postgres-db", "INFO",
                 "Checkpoint completed: wrote 34 buffers, 0.1% of total"),
    ]

    mem_lower = mem_limit.lower()
    return Scenario(
        id=1, name="Single OOM Crash", difficulty=0.1,
        alert_message=(
            f"[CRITICAL] {root_svc} is not responding. "
            "Health check has failed 3 consecutive times. "
            "Alert fired at 2024-01-15 14:35:00 UTC. "
            "On-call engineer paged. Acknowledge and investigate."
        ),
        services=services,
        logs=logs,
        deployment_events=[],
        root_cause_services=[root_svc],
        correct_fix_actions=[f"restart_service:{root_svc}"],
        correct_diagnoses=["oom", "out of memory", "memory limit", "oomkilled", mem_lower],
        correct_diagnosis_groups=[[
            "oom", "out of memory", "memory limit", "oomkilled", "signal 9", mem_lower,
        ]],
        max_steps=8,
        max_reward=82.0,
        escalation_correct=False,
        hint=f"Tip: Start by querying logs for the service mentioned in the alert, then check its metrics.",
    )


# ---------------------------------------------------------------------------
# Scenario 2 — Bad Deployment  (D: 0.3)
# ---------------------------------------------------------------------------

def _make_scenario_2(rng: random.Random) -> Scenario:
    root_svc = rng.choice(_DEPLOY_POOL)
    major = rng.randint(3, 9)
    old_minor = rng.randint(1, 8)
    new_minor = old_minor + 1
    old_ver = f"{major}.{old_minor}.{rng.randint(0, 5)}"
    new_ver = f"{major}.{new_minor}.0"
    exc_class = rng.choice(_EXCEPTION_CLASSES)
    java_line = rng.randint(100, 500)
    java_method = rng.choice([
        "calculateDiscount", "processOrder", "validatePayment",
        "resolveUser", "buildResponse", "fetchInventory",
    ])
    java_class = rng.choice([
        "OrderProcessor", "PaymentHandler", "UserResolver",
        "InventoryManager", "CatalogService", "PricingEngine",
    ])

    services = {
        "api-gateway": _svc("api-gateway", "degraded",
                            cpu=rng.uniform(55, 70), mem=rng.uniform(38, 50),
                            err=rng.uniform(18, 28), p99=rng.uniform(2200, 3200)),
        root_svc: _svc(root_svc, "degraded",
                       cpu=rng.uniform(45, 65), mem=rng.uniform(44, 58),
                       err=rng.uniform(55, 75), p99=rng.uniform(2800, 3500),
                       root=True),
        "notification-service": _svc("notification-service", "degraded",
                                      cpu=rng.uniform(80, 95), mem=rng.uniform(38, 46),
                                      err=rng.uniform(1, 4), p99=rng.uniform(80, 110),
                                      herring=True),
        "auth-service": _svc("auth-service", "healthy",
                             cpu=rng.uniform(12, 25), mem=rng.uniform(22, 34),
                             err=0.0, p99=rng.uniform(35, 55)),
        "user-service": _svc("user-service", "healthy",
                             cpu=rng.uniform(25, 38), mem=rng.uniform(32, 44),
                             err=0.0, p99=rng.uniform(65, 90)),
    }

    dep_ts = "2024-01-15 16:00:03"
    logs = [
        LogEntry(dep_ts, root_svc, "INFO",
                 f"Deployment started: {old_ver} → {new_ver} by jenkins-ci"),
        LogEntry("2024-01-15 16:00:41", root_svc, "INFO",
                 f"Deployment completed: running version {new_ver}"),
        LogEntry("2024-01-15 16:01:14", root_svc, "ERROR",
                 f"{exc_class} at {java_class}.{java_method}({java_class}.java:{java_line})"),
        LogEntry("2024-01-15 16:01:15", root_svc, "ERROR",
                 f"{exc_class} at {java_class}.{java_method}({java_class}.java:{java_line})"),
        LogEntry("2024-01-15 16:01:22", root_svc, "ERROR",
                 "Unhandled exception in request handler. Returning 500."),
        LogEntry("2024-01-15 16:02:10", "api-gateway", "WARN",
                 f"upstream {root_svc} response time: 3100ms (threshold: 500ms)"),
        LogEntry("2024-01-15 16:05:00", "api-gateway", "ERROR",
                 f"Circuit breaker OPEN for {root_svc} (error rate {int(services[root_svc].error_rate_pct)}%)"),
        LogEntry("2024-01-15 16:10:00", "api-gateway", "ERROR",
                 f"p99 latency threshold breached: {int(services['api-gateway'].p99_latency_ms)}ms. Firing alert."),
        LogEntry("2024-01-15 15:55:00", "notification-service", "INFO",
                 "Scheduled batch job started: digest-email-sender"),
        LogEntry("2024-01-15 16:08:00", "notification-service", "INFO",
                 "Batch job in progress: 45,000/120,000 emails processed"),
        LogEntry("2024-01-15 16:09:00", "auth-service", "INFO",
                 "Processed 1240 authentication requests. All successful."),
    ]

    exc_lower = exc_class.lower()
    return Scenario(
        id=2, name="Bad Deployment", difficulty=0.3,
        alert_message=(
            f"[HIGH] api-gateway p99 latency has exceeded 2000ms for the past 5 minutes. "
            f"Error rate: {int(services['api-gateway'].error_rate_pct)}%. "
            "Alert fired at 2024-01-15 16:10:00 UTC. "
            "A deployment occurred recently. Investigate and resolve."
        ),
        services=services,
        logs=logs,
        deployment_events=[
            DeploymentEvent(dep_ts, root_svc, old_ver, new_ver, "jenkins-ci"),
        ],
        root_cause_services=[root_svc],
        correct_fix_actions=[f"rollback_deployment:{root_svc}"],
        correct_diagnoses=[
            "bad deployment", exc_lower, f"{root_svc} deployment",
            new_ver, "rollback", "circuit breaker", "error rate",
            "deployment issue", "high latency", root_svc,
        ],
        correct_diagnosis_groups=[[
            "bad deployment", exc_lower, f"{root_svc} deployment",
            new_ver, "rollback", "circuit breaker", "error rate",
            "deployment issue", "high latency", root_svc,
        ]],
        max_steps=12,
        max_reward=85.0,
        escalation_correct=False,
    )


# ---------------------------------------------------------------------------
# Scenario 3 — DB Connection Pool Exhaustion  (D: 0.5)
# ---------------------------------------------------------------------------

def _make_scenario_3(rng: random.Random) -> Scenario:
    leak_svc = rng.choice(_LEAK_POOL)
    pool_size = rng.choice(_POOL_SIZES)
    fill_1 = pool_size - rng.randint(5, 10)
    fill_2 = pool_size - rng.randint(2, 4)
    fill_3 = pool_size

    # Other services that are blocked waiting for DB (not the leak source)
    blocked = [s for s in ["auth-service", "order-service", "user-service"] if s != leak_svc]
    svc_a, svc_b = blocked[0], blocked[1]

    services = {
        "api-gateway": _svc("api-gateway", "degraded",
                            cpu=rng.uniform(65, 78), mem=rng.uniform(42, 54),
                            err=rng.uniform(80, 93), p99=rng.uniform(4500, 6000)),
        svc_a: _svc(svc_a, "down",
                    cpu=rng.uniform(8, 15), mem=rng.uniform(28, 40),
                    err=rng.uniform(94, 99), p99=30000.0),
        svc_b: _svc(svc_b, "down",
                    cpu=rng.uniform(6, 14), mem=rng.uniform(25, 38),
                    err=rng.uniform(92, 97), p99=30000.0),
        leak_svc: _svc(leak_svc, "degraded",
                       cpu=rng.uniform(14, 22), mem=rng.uniform(38, 50),
                       err=rng.uniform(88, 94), p99=rng.uniform(25000, 30000),
                       root=True),
        "postgres-db": _svc("postgres-db", "degraded",
                            cpu=rng.uniform(78, 92), mem=rng.uniform(72, 84),
                            err=0.0, p99=rng.uniform(380, 520), root=True),
        "redis-cache": _svc("redis-cache", "degraded",
                            cpu=rng.uniform(35, 48), mem=rng.uniform(78, 88),
                            err=rng.uniform(2, 5), p99=rng.uniform(6, 12),
                            herring=True),
    }

    logs = [
        LogEntry("2024-01-15 18:30:01", leak_svc, "WARN",
                 f"DB pool usage rising: {fill_1}/{pool_size} active connections — connections not being released after requests"),
        LogEntry("2024-01-15 18:31:12", leak_svc, "WARN",
                 f"DB connection leak suspected: {fill_2}/{pool_size} active — request handlers in {leak_svc} are not closing connections"),
        LogEntry("2024-01-15 18:31:55", leak_svc, "ERROR",
                 f"DB connection pool exhausted by {leak_svc}: {fill_3}/{pool_size} connections held open (connection leak — restart {leak_svc} to clear)"),
        LogEntry("2024-01-15 18:32:00", "postgres-db", "FATAL",
                 "FATAL: remaining connection slots are reserved for non-replication superuser connections"),
        LogEntry("2024-01-15 18:32:01", "postgres-db", "ERROR",
                 f"connection pool exhausted: max_connections={pool_size}, active={pool_size}"),
        LogEntry("2024-01-15 18:32:03", svc_a, "ERROR",
                 "ConnectionPoolTimeoutError: QueuePool limit of size 20 overflow 10 reached, connection timed out"),
        LogEntry("2024-01-15 18:32:03", svc_b, "ERROR",
                 "ConnectionPoolTimeoutError: QueuePool limit of size 20 overflow 10 reached, connection timed out"),
        LogEntry("2024-01-15 18:32:05", svc_a, "ERROR",
                 "Returning 503: unable to acquire database connection"),
        LogEntry("2024-01-15 18:32:05", svc_b, "ERROR",
                 "Returning 503: unable to acquire database connection"),
        LogEntry("2024-01-15 18:44:00", "redis-cache", "WARN",
                 "Memory usage at 82% (maxmemory-policy: allkeys-lru)"),
        LogEntry("2024-01-15 18:44:30", "redis-cache", "INFO",
                 "Evicted 1240 keys using allkeys-lru policy"),
        LogEntry("2024-01-15 18:45:00", "api-gateway", "ERROR",
                 "All backend services reporting errors. Firing critical alert."),
    ]

    svc_a_err = int(services[svc_a].error_rate_pct)
    svc_b_err = int(services[svc_b].error_rate_pct)
    leak_err = int(services[leak_svc].error_rate_pct)
    return Scenario(
        id=3, name="DB Connection Pool Exhaustion", difficulty=0.5,
        alert_message=(
            "[CRITICAL] Multiple services returning 503 Service Unavailable. "
            f"Affected: {svc_a} (error_rate={svc_a_err}%), "
            f"{svc_b} (error_rate={svc_b_err}%), "
            f"{leak_svc} (error_rate={leak_err}%). "
            "Alert fired at 2024-01-15 18:45:00 UTC. "
            "All three services are degraded simultaneously."
        ),
        services=services,
        logs=logs,
        deployment_events=[],
        root_cause_services=["postgres-db", leak_svc],
        correct_fix_actions=[
            "restart_service:postgres-db",
            f"restart_service:{leak_svc}",
        ],
        correct_diagnoses=[
            "connection pool", "db connection", "connection exhausted", "postgres",
            "connection leak", "timeout", "cannot acquire", "database connection",
            "pool exhausted", "db pool", "connection timeout", leak_svc,
        ],
        correct_diagnosis_groups=[[
            "connection pool", "db connection", "connection exhausted", "postgres",
            "connection leak", "timeout", "cannot acquire", "database connection",
            "pool exhausted", "db pool", "connection timeout", leak_svc,
        ]],
        max_steps=15,
        max_reward=100.0,
        escalation_correct=False,
    )


# ---------------------------------------------------------------------------
# Scenario 4 — Cascading Microservice Failure  (D: 0.75)
# ---------------------------------------------------------------------------

def _make_scenario_4(rng: random.Random) -> Scenario:
    root_svc = rng.choice(_CASCADE_UPSTREAM_POOL)
    mem_limit = rng.choice(_MEMORY_LIMITS)
    warn_pct = rng.randint(88, 93)

    # Downstream services that cascade from root_svc going down
    downstream_pool = [s for s in ["auth-service", "order-service", "user-service"]
                       if s != root_svc]
    circuit_target = rng.choice(downstream_pool)
    other_down = [s for s in downstream_pool if s != circuit_target][0]

    services = {
        "api-gateway": _svc("api-gateway", "down",
                            cpu=rng.uniform(3, 8), mem=rng.uniform(18, 26),
                            err=100.0, p99=0.0),
        circuit_target: _svc(circuit_target, "down",
                             cpu=rng.uniform(5, 12), mem=rng.uniform(22, 32),
                             err=100.0, p99=0.0),
        other_down: _svc(other_down, "down",
                         cpu=rng.uniform(4, 10), mem=rng.uniform(20, 30),
                         err=100.0, p99=0.0),
        root_svc: _svc(root_svc, "down",
                       cpu=0.0, mem=0.0, err=100.0, p99=0.0, root=True),
        "notification-service": _svc("notification-service", "degraded",
                                      cpu=rng.uniform(28, 42), mem=rng.uniform(42, 54),
                                      err=rng.uniform(35, 48), p99=rng.uniform(750, 950)),
        "redis-cache": _svc("redis-cache", "degraded",
                            cpu=rng.uniform(48, 62), mem=rng.uniform(65, 78),
                            err=rng.uniform(5, 12), p99=rng.uniform(18, 28),
                            herring=True),
        "postgres-db": _svc("postgres-db", "healthy",
                            cpu=rng.uniform(14, 24), mem=rng.uniform(36, 48),
                            err=0.0, p99=rng.uniform(10, 18)),
    }

    logs = [
        LogEntry("2024-01-15 20:10:01", root_svc, "WARN",
                 f"Memory usage at {warn_pct}% (limit: {mem_limit})"),
        LogEntry("2024-01-15 20:11:33", root_svc, "FATAL",
                 f"Killed by signal 9 (OOMKilled). Memory limit exceeded: {mem_limit}"),
        LogEntry("2024-01-15 20:11:35", circuit_target, "ERROR",
                 f"gRPC connection to {root_svc} failed: connection refused"),
        LogEntry("2024-01-15 20:11:35", other_down, "ERROR",
                 f"gRPC connection to {root_svc} failed: connection refused"),
        LogEntry("2024-01-15 20:11:36", circuit_target, "ERROR",
                 "Unable to validate user session: upstream unavailable. Returning 503."),
        LogEntry("2024-01-15 20:11:36", other_down, "ERROR",
                 "Unable to fetch user profile: upstream unavailable. Returning 503."),
        LogEntry("2024-01-15 20:11:40", "api-gateway", "ERROR",
                 f"Upstream {circuit_target}: 503. Upstream {other_down}: 503."),
        LogEntry("2024-01-15 20:12:00", "notification-service", "WARN",
                 f"Failed to send 340 notifications ({root_svc} unreachable for preferences fetch)"),
        LogEntry("2024-01-15 20:13:00", "redis-cache", "WARN",
                 "Eviction rate elevated: 2400 keys/min (normal: ~200 keys/min)"),
        LogEntry("2024-01-15 20:14:00", "redis-cache", "WARN",
                 "Memory at 71%. LRU eviction active."),
        LogEntry("2024-01-15 20:15:00", "api-gateway", "ERROR",
                 f"{circuit_target} 100% error rate for 3min 20sec. Firing CRITICAL alert."),
        LogEntry("2024-01-15 20:15:30", "api-gateway", "WARN",
                 f"Recommendation: after fixing upstream {root_svc}, enable circuit breaker on "
                 f"{circuit_target} and {other_down} to prevent future cascades."),
    ]

    mem_lower = mem_limit.lower()
    return Scenario(
        id=4, name="Cascading Microservice Failure", difficulty=0.75,
        alert_message=(
            f"[CRITICAL] {circuit_target} and {other_down} both reporting 100% error rate. "
            "Alert fired at 2024-01-15 20:15:00 UTC. "
            "Downstream impact: API gateway unavailable. Customer-facing features offline. "
            "Multiple alerts firing simultaneously. Investigate urgently."
        ),
        services=services,
        logs=logs,
        deployment_events=[],
        root_cause_services=[root_svc],
        correct_fix_actions=[
            f"restart_service:{root_svc}",
            f"enable_circuit_breaker:{circuit_target}",
            f"enable_circuit_breaker:{other_down}",
        ],
        correct_diagnoses=[
            "oom", "out of memory", root_svc, "cascade", "cascading failure",
            "upstream", "memory", "oomkill", "memory limit", "circuit",
            "downstream", "service failure", "upstream failure", mem_lower,
        ],
        correct_diagnosis_groups=[[
            "oom", "out of memory", root_svc, "cascade", "cascading failure",
            "upstream", "memory", "oomkill", "memory limit", "circuit",
            "downstream", "service failure", "upstream failure", mem_lower,
        ]],
        max_steps=18,
        max_reward=105.0,
        escalation_correct=False,
    )


# ---------------------------------------------------------------------------
# Scenario 5 — Multi-Incident Surge  (D: 1.0)
# ---------------------------------------------------------------------------

def _make_scenario_5(rng: random.Random) -> Scenario:
    # DDoS: randomise the attacker IP and peak request rate
    ip_suffix = f"{rng.randint(1, 254)}.{rng.randint(1, 254)}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"
    ddos_ip = f"203.0.{ip_suffix}"  # kept in the 203.0.113.x documentation range
    peak_rate = rng.choice([38000, 42000, 48000, 55000])
    normal_rate = 2000

    # Config drift: randomise version labels
    old_cfg = f"config-v{rng.randint(1, 5)}"
    new_cfg = f"config-v{int(old_cfg.split('v')[1]) + 1}"

    services = {
        "api-gateway": _svc("api-gateway", "degraded",
                            cpu=rng.uniform(94, 99), mem=rng.uniform(70, 82),
                            err=rng.uniform(90, 97), p99=rng.uniform(7500, 9500),
                            root=True),
        "auth-service": _svc("auth-service", "degraded",
                             cpu=rng.uniform(38, 50), mem=rng.uniform(32, 44),
                             err=100.0, p99=rng.uniform(100, 140),
                             root=True),
        "order-service": _svc("order-service", "down",
                              cpu=rng.uniform(2, 6), mem=rng.uniform(16, 24),
                              err=100.0, p99=0.0),
        "user-service": _svc("user-service", "down",
                             cpu=rng.uniform(2, 6), mem=rng.uniform(18, 26),
                             err=100.0, p99=0.0),
        "inventory-service": _svc("inventory-service", "degraded",
                                   cpu=rng.uniform(32, 44), mem=rng.uniform(36, 46),
                                   err=rng.uniform(28, 42), p99=rng.uniform(700, 850),
                                   herring=True),
        "postgres-db": _svc("postgres-db", "healthy",
                            cpu=rng.uniform(22, 34), mem=rng.uniform(48, 62),
                            err=0.0, p99=rng.uniform(14, 22)),
        "redis-cache": _svc("redis-cache", "healthy",
                            cpu=rng.uniform(16, 28), mem=rng.uniform(38, 50),
                            err=0.0, p99=rng.uniform(2, 5)),
    }

    api_err = int(services["api-gateway"].error_rate_pct)
    logs = [
        LogEntry("2024-01-15 22:28:00", "api-gateway", "WARN",
                 f"Unusually high request rate: {peak_rate:,} req/min (normal: {normal_rate:,} req/min)"),
        LogEntry("2024-01-15 22:28:30", "api-gateway", "WARN",
                 f"Single IP {ddos_ip} responsible for {peak_rate - 3000:,} req/min"),
        LogEntry("2024-01-15 22:29:00", "api-gateway", "ERROR",
                 "Rate limit threshold breached. Dropping requests. error_rate rising."),
        LogEntry("2024-01-15 22:29:30", "api-gateway", "ERROR",
                 "CPU at 98%. Thread pool exhausted. Queuing degraded."),
        LogEntry("2024-01-15 22:25:01", "auth-service", "INFO",
                 f"Config reload triggered by configmap update: {new_cfg}"),
        LogEntry("2024-01-15 22:25:05", "auth-service", "WARN",
                 "JWT_SECRET changed in environment. All active sessions may be invalidated."),
        LogEntry("2024-01-15 22:26:00", "auth-service", "ERROR",
                 "JWT validation failed: signature verification failed (token=eyJhbGci...)"),
        LogEntry("2024-01-15 22:26:01", "auth-service", "ERROR",
                 "JWT validation failed: signature verification failed (token=eyJhbGci...)"),
        LogEntry("2024-01-15 22:27:00", "auth-service", "ERROR",
                 f"100% of JWT validation requests failing. JWT_SECRET mismatch suspected."),
        LogEntry("2024-01-15 22:29:00", "order-service", "ERROR",
                 "Authentication required but auth-service returning 401 for all requests"),
        LogEntry("2024-01-15 22:29:00", "user-service", "ERROR",
                 "Authentication required but auth-service returning 401 for all requests"),
        LogEntry("2024-01-15 22:29:30", "inventory-service", "WARN",
                 "Database query latency elevated: 780ms (normal: 50ms). Possibly resource contention."),
        LogEntry("2024-01-15 22:30:00", "api-gateway", "ERROR",
                 f"Error rate {api_err}% sustained for 90 seconds. Firing CRITICAL multi-alert."),
    ]

    ip_prefix = ".".join(ddos_ip.split(".")[:3])  # for partial-match diagnosis
    return Scenario(
        id=5, name="Multi-Incident Surge", difficulty=1.0,
        alert_message=(
            "[CRITICAL] MULTIPLE SIMULTANEOUS INCIDENTS DETECTED. "
            f"(1) api-gateway error rate: {api_err}% — possible DDoS. "
            "(2) auth-service JWT validation failures: 100% — ALL user sessions being rejected. "
            "Alert fired at 2024-01-15 22:30:00 UTC. "
            "Customer impact: complete service outage. Both issues must be resolved independently."
        ),
        services=services,
        logs=logs,
        deployment_events=[
            DeploymentEvent("2024-01-15 22:25:00", "auth-service", old_cfg, new_cfg,
                            "automated-configmap-sync"),
        ],
        root_cause_services=["api-gateway", "auth-service"],
        correct_fix_actions=[
            "enable_rate_limiting:api-gateway",
            "rollback_config:auth-service",
        ],
        correct_diagnoses=[
            "ddos", "rate limit", "ip", ip_prefix,
            "jwt", "jwt_secret", "config drift", "configmap", "auth config",
            "traffic spike", "rate limiting", "misconfiguration", "config",
            "secret", "authentication", "surge", new_cfg,
        ],
        correct_diagnosis_groups=[
            # Group 0: DDoS / traffic surge on api-gateway
            ["ddos", "rate limit", "ip", ip_prefix, "traffic spike", "rate limiting", "surge"],
            # Group 1: JWT config drift on auth-service
            ["jwt", "jwt_secret", "config drift", "configmap", "auth config",
             "misconfiguration", "config", "secret", "authentication", new_cfg],
        ],
        max_steps=22,
        max_reward=122.0,
        escalation_correct=False,
    )


# ---------------------------------------------------------------------------
# Scenario 6 — False Alarm / Planned Load Test  (D: 0.2)
# ---------------------------------------------------------------------------

def _make_scenario_6(rng: random.Random) -> Scenario:
    activity = rng.choice(_ACTIVITY_TYPES)           # e.g. "stress test"
    team = rng.choice(_LOAD_TEST_TEAMS)              # e.g. "perf-team@k8s-job"
    peak_rate = rng.choice([28000, 35000, 42000, 50000])
    normal_rate = 2000
    cpu_pct = rng.uniform(65, 80)
    latency = rng.uniform(1500, 2200)

    # Randomise the planned window hour
    start_hour = rng.choice([10, 14, 22, 2])
    start_str = f"{start_hour:02d}:00"
    end_str = f"{start_hour:02d}:30"
    ts_base = f"2024-01-15 {start_hour:02d}"

    services = {
        "api-gateway": _svc("api-gateway", "degraded",
                            cpu=cpu_pct, mem=rng.uniform(48, 62),
                            err=rng.uniform(2, 6), p99=latency),
        "user-service": _svc("user-service", "healthy",
                             cpu=rng.uniform(32, 44), mem=rng.uniform(36, 46),
                             err=0.0, p99=rng.uniform(80, 110)),
        "order-service": _svc("order-service", "healthy",
                              cpu=rng.uniform(36, 50), mem=rng.uniform(42, 54),
                              err=0.0, p99=rng.uniform(95, 125)),
        "auth-service": _svc("auth-service", "healthy",
                             cpu=rng.uniform(24, 36), mem=rng.uniform(28, 38),
                             err=0.0, p99=rng.uniform(40, 55)),
        "postgres-db": _svc("postgres-db", "healthy",
                            cpu=rng.uniform(20, 30), mem=rng.uniform(38, 50),
                            err=0.0, p99=rng.uniform(12, 20)),
    }

    logs = [
        LogEntry(f"{ts_base}:00:00", "api-gateway", "INFO",
                 f"Scheduled {activity} started by {team} (planned window: {start_str}–{end_str} UTC)"),
        LogEntry(f"{ts_base}:00:30", "api-gateway", "INFO",
                 f"Request rate climbing: 5,000 req/min → {activity} ramp-up phase"),
        LogEntry(f"{ts_base}:02:00", "api-gateway", "WARN",
                 f"Request rate: {peak_rate // 2:,} req/min. CPU at {int(cpu_pct * 0.85)}%. "
                 f"Within {activity} parameters."),
        LogEntry(f"{ts_base}:04:00", "api-gateway", "WARN",
                 f"Request rate: {peak_rate:,} req/min. CPU at {int(cpu_pct)}%. "
                 f"p99 latency elevated (expected during {activity})."),
        LogEntry(f"{ts_base}:05:00", "api-gateway", "ERROR",
                 f"Automated alert triggered: request rate exceeds threshold. "
                 f"NOTE: {activity} in progress."),
        LogEntry(f"{ts_base}:05:01", "user-service", "INFO",
                 "Handling elevated load gracefully. No errors detected."),
        LogEntry(f"{ts_base}:05:01", "order-service", "INFO",
                 "Handling elevated load gracefully. No errors detected."),
        LogEntry(f"{ts_base}:05:02", "auth-service", "INFO",
                 "1,200 authentication requests processed. All successful."),
        LogEntry(f"{ts_base}:05:10", "api-gateway", "INFO",
                 f"{team}: target throughput reached. Sustaining for 25 minutes."),
    ]

    # build synonym list so different phrasings of the activity type still match
    activity_kws = [activity]
    if "load" in activity:
        activity_kws += ["load test", "load-test"]
    if "stress" in activity:
        activity_kws += ["stress test", "stress-test"]
    if "capacity" in activity:
        activity_kws += ["capacity test", "capacity-test"]

    diagnosis_kws = activity_kws + [
        "planned", "expected", team, team.split("@")[0],
        "scheduled", "no action", "intentional", "test traffic",
    ]

    return Scenario(
        id=6, name="False Alarm: Planned Load Test", difficulty=0.2,
        alert_message=(
            f"[HIGH] api-gateway request rate spiked to {peak_rate:,} req/min "
            f"(normal: {normal_rate:,} req/min). "
            f"CPU elevated: {int(cpu_pct)}%. p99 latency: {int(latency)}ms. "
            f"Alert fired at {ts_base}:05:00 UTC. "
            "All services are currently responding. Investigate and determine if action is required."
        ),
        services=services,
        logs=logs,
        deployment_events=[],
        root_cause_services=[],   # intentional — this is expected traffic
        correct_fix_actions=[],   # no fix needed; any fix is penalised
        correct_diagnoses=diagnosis_kws,
        correct_diagnosis_groups=[diagnosis_kws],
        max_steps=10,
        max_reward=65.0,
        escalation_correct=False,
        hint=(
            f"Tip: Check api-gateway logs carefully — the alert details and recent "
            f"log entries may tell the full story."
        ),
    )


# ---------------------------------------------------------------------------
# Public dispatcher — called by environment.py on every reset()
# ---------------------------------------------------------------------------

_GENERATORS: Dict[int, Callable[[random.Random], Scenario]] = {
    1: _make_scenario_1,
    2: _make_scenario_2,
    3: _make_scenario_3,
    4: _make_scenario_4,
    5: _make_scenario_5,
    6: _make_scenario_6,
}

ALL_SCENARIO_IDS: List[int] = list(_GENERATORS.keys())


def generate_scenario(scenario_id: int, seed: Optional[int] = None) -> Scenario:
    """
    Return a freshly generated Scenario for the given ID.

    Args:
        scenario_id: Integer 1–6 identifying the scenario type.
        seed:        Optional integer seed for reproducibility. If None, each call
                     produces a different variant (random seed chosen by the caller).
    """
    if scenario_id not in _GENERATORS:
        raise ValueError(f"Unknown scenario_id {scenario_id}. Valid: {ALL_SCENARIO_IDS}")
    rng = random.Random(seed)
    return _GENERATORS[scenario_id](rng)

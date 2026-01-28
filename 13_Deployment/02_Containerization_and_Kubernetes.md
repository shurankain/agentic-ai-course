# Containerization and Kubernetes for LLM Applications

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Deployment
**Previous:** [[01_Deployment_Strategies|Deployment Strategies]]
**Next:** [[03_CI_CD_and_Self_Hosted|CI/CD and Self Hosted]]

---

## Containerization as the Foundation of Deployment

Docker ensures environment reproducibility: the application runs identically on a developer's machine, in CI, and in production. This is critical for LLM applications due to complex dependencies: provider client libraries, ML libraries, vector store clients. Containerization solves the "works on my machine" problem, but LLM systems have specifics: large images due to ML dependencies, less predictable memory consumption, increased cold start time.

**Docker Layer Theory.** An image consists of layers from bottom to top. Lower layers (base OS, runtime) are fixed. Middle layers (system packages, dependencies) change rarely. Upper layers (source code) change frequently. Typical structure: Base OS debian-slim 70MB, Java runtime 150MB, System packages 50MB, Dependencies 200MB, Application code 4MB.

**Caching Principle.** Each Dockerfile instruction creates an immutable layer with a SHA256 hash. During rebuild, Docker checks for changes: if none — uses cache, if changed — rebuilds that and all subsequent layers. Critical rule: place frequently changing instructions at the bottom, rarely changing ones at the top.

**Anti-pattern:** copying all code at once followed by building. Changing a single file invalidates the cache, forcing re-download of hundreds of megabytes of dependencies. **Correct approach:** first copy only the dependency file (pom.xml, requirements.txt), install dependencies, then copy the code. The dependency layer remains in cache. Rebuild takes seconds instead of minutes.

## Image Optimization

Instruction order determines cache usage. Base image → system dependencies → application dependencies → source code → build. Base image choice affects size and security. Alpine images are minimal but have compatibility issues with musl instead of glibc. Distroless are even smaller but harder to debug. For production, start with slim variants and transition to distroless as the application stabilizes.

**Multi-stage build** separates builder and runtime environments. Builder contains compilers, build tools, dev dependencies. Runtime contains only JRE/Python runtime and the compiled application. Reduces image size by 2-5x, lowers the attack surface.

**Java:** Spring Boot layertools splits the JAR into layers: dependencies, spring-boot-loader, snapshot-dependencies, application. When only your code changes, only the last layer is rebuilt. Use a non-root user: create a dedicated user, assign permissions, switch before ENTRYPOINT.

**Python:** use pip install --no-cache-dir, install dependencies into a virtual environment, copy only it to runtime. Compile files to bytecode via python -m compileall — speeds up first startup by 5-15 seconds.

## JVM in Containers

Java applications require special configuration. JVM historically worked poorly with cgroups: it detected the entire host memory and created too many threads. Modern versions (11+) resolve this but require flags.

The UseContainerSupport flag (enabled by default in Java 11+) forces JVM to respect limits. MaxRAMPercentage=75.0 sets the maximum heap as a percentage of available memory. Leaving 25% for non-heap (metaspace, native memory, stacks) prevents OOMKilled. G1GC balances throughput and pause times. ZGC provides sub-millisecond pauses but requires more CPU. For LLM applications where the primary latency is in API calls, G1GC is sufficient. HeapDumpOnOutOfMemoryError saves the heap state for analysis.

## Health Checks

Liveness probe answers "is the application still running?". Failure leads to restart. For LLM applications, it checks basic functionality: the application responds, main threads are not blocked. Checking external LLM API availability in liveness is dangerous — temporary provider issues will cause infinite restarts.

Readiness probe determines readiness to accept traffic. Failure removes the pod from load balancing without restart. It checks database connections, vector stores, caches. LLM provider unavailability is a borderline case: depends on whether a fallback provider exists.

Graceful shutdown: upon SIGTERM, the application stops accepting new requests, waits for current ones to complete, then exits. For agent operations lasting minutes, terminationGracePeriodSeconds should be 180-300 seconds.

## Docker Compose for Development

Compose organizes the local environment: application, database, vector store, observability. Ensures consistency between developers. Typical stack: Redis for caching and rate limiting, PostgreSQL for conversations and state, vector database for RAG, optionally Prometheus and Grafana.

Environment variables manage configuration. API keys via environment, never hardcoded. The .env file is excluded from git. Health checks ensure proper startup order: the application starts after the database is ready. Depends_on with condition: service_healthy replaces sleep scripts. Volumes: bind mounts for editing code without rebuilding, named volumes for database persistence.

**Compose file organization:** docker-compose.yml contains the base configuration. docker-compose.override.yml (applied automatically) adds development settings: bind mounts, extended logging, debug ports. docker-compose.prod.yml for production: different images, fewer ports, stricter security. Networks isolate service groups. Profiles enable services optionally: the observability stack is activated with --profile monitoring.

## Kubernetes Basics

Kubernetes orchestrates containers: manages deployments, service discovery, scaling, self-healing. For LLM, high availability and elastic scaling under variable loads are critical.

Deployment defines the desired state: replica count, image, resources. Rolling update upgrades pods gradually, ensuring zero-downtime. Resource requests and limits define guaranteed and maximum resources. Too small requests — eviction under resource pressure. Too large — inefficient cluster utilization. Pod Anti-Affinity distributes replicas across different nodes, improving availability. Annotations for Prometheus enable automatic metrics collection.

**Rolling update:** MaxSurge defines additional pods during the update. A value of 1 with 3 replicas allows a maximum of 4 pods. MaxUnavailable defines unavailable pods. A value of 0 guarantees the new pod is ready before removing the old one. TerminationGracePeriodSeconds for LLM agents should be 180-300 seconds.

**Probes:** Startup probe solves the slow start problem. Liveness and readiness are disabled until startup succeeds. Critical for LLM applications with long initialization. Liveness with failureThreshold and periodSeconds prevents restart loops. Readiness for LLM should check critical dependencies: if the vector store is unavailable and RAG is a key feature — fail. If there is a fallback source — ignore.

## Service and Ingress

Service abstracts pods behind a stable DNS name and IP. Clients connect to the service, Kubernetes balances requests. ClusterIP for internal services, LoadBalancer or Ingress for external access.

Ingress provides HTTP(S) access from outside the cluster. Ingress Controller (nginx, traefik, istio) configures the reverse proxy. For LLM, timeout settings are critical: the standard 60 seconds is insufficient for long generations. TLS termination at Ingress simplifies certificate management. Cert-manager automatically obtains certificates from Let's Encrypt.

Timeout annotations are controller-specific. Nginx: proxy-read-timeout, proxy-send-timeout. For streaming, disable buffering: proxy-buffering: "off". Without this, SSE and WebSocket work incorrectly.

**Session affinity:** for stateful interactions, sessionAffinity: ClientIP may be required — requests from the same IP go to the same pod. For production, external session storage (Redis) is preferable, allowing any pod to handle any request. Rate limiting on Ingress protects against abuse: limit-rps restricts requests per second per IP. CORS settings allow cross-origin requests: specific origins instead of wildcard for production.

## Horizontal Pod Autoscaler

HPA scales the number of pods based on metrics. CPU is not always the best metric for LLM, as most time is spent waiting for the API. Custom Metrics API allows scaling by application-specific metrics: active request count, queue size, latency. Prometheus Adapter or KEDA integrates custom metrics with HPA.

**KEDA** extends HPA with event-driven triggers. Unlike HPA, it supports scale to zero, event-driven scaling, and native queue integration. Scales by request queue depth, P95 response time, external metrics from Prometheus, messages in Kafka/RabbitMQ. For LLM services: scaling by queue depth (pending requests exceeding 10), by P95 latency (response time exceeding 5 seconds), by custom business metrics (active agent sessions, token volume per minute).

KEDA is useful for batch processing and asynchronous tasks. Tasks in a queue (Redis, RabbitMQ, Kafka) — KEDA scales workers to zero when there is no work and spins them up when work appears. Saves resources for irregular workloads.

**KEDA vs HPA comparison:** Scale to zero is KEDA's key advantage. HPA requires a minimum of 1 replica. KEDA's event-driven triggers react to events, not only metrics. External metrics in KEDA are declarative via ScaledObject. HPA requires metrics server and adapter setup. Queue-based scaling is native in KEDA for RabbitMQ, Kafka, SQS, Redis.

The behavior section of HPA controls scaling speed. Scale-up is aggressive during sudden growth. Scale-down is conservative — premature pod removal during temporary dips creates problems. Stabilization window prevents flapping — frequent changes during metric fluctuations. Minimum replicas ensure baseline capacity: 2-3 replicas for production, distributed across nodes.

## Resource Management

Requests define guaranteed resources for the scheduler. Limits restrict maximum consumption. For LLM, memory is critical: context, prompts, caches. Insufficient limits — OOMKilled. Excessive limits — inefficient utilization. Rule: requests = typical consumption, limits = peak + buffer.

CPU limits are less critical: most time is spent on network I/O to the LLM API. Throttling is less painful than OOMKilled. Some practices do not set CPU limits, only requests.

QoS classes: Guaranteed (requests = limits for both CPU and memory) — highest priority. Burstable (requests < limits) — medium. BestEffort (no requests or limits) — lowest, first to be evicted.

**Recommendations:** Typical LLM API: memory requests 512Mi-1Gi, limits 1.5-2Gi. CPU requests 250-500m, limits 1-2 CPU or no limit. Agent systems: memory 2-4Gi requests, 4-8Gi limits. ResourceQuotas per namespace restrict total consumption. LimitRanges set default and maximum values.

## ConfigMaps and Secrets

ConfigMaps store configuration: prompts, model settings, feature flags. Secrets store sensitive data: API keys, database credentials, TLS certificates. Secrets are base64-encoded but not encrypted by default. For production, enable encryption at rest and integrate with external secret management (Vault, AWS Secrets Manager).

Mounting as a volume is safer for complex configurations and secrets. Environment variables are simpler but visible in describe pod and can leak into logs.

**Automatic updates:** to restart pods when a ConfigMap changes, use a checksum annotation. Compute the ConfigMap hash and add it to the Deployment annotations. When the hash changes, Kubernetes triggers a rolling update. For dynamic reload without restart, implement file watching: Spring Boot supports config refresh via Actuator.

**Secrets security:** enable encryption at rest in etcd via EncryptionConfiguration. By default, base64-encoded Secrets are NOT encrypted. External Secrets Operator syncs from Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager. Pods receive secrets on the fly, rotation is automated. RBAC restricts access: a ServiceAccount should only have access to its own secrets.

## Key Takeaways

Docker images for LLM require optimization of size and build time. Multi-stage builds, proper layer ordering, and base image selection significantly impact efficiency. JVM in containers requires UseContainerSupport, MaxRAMPercentage, and GC selection. Health checks distinguish between liveness (application is running) and readiness (ready to accept traffic). Readiness can account for caches and connections but not external LLM API availability.

Kubernetes Deployment with proper resources, probes, and affinity ensures stability. Pod Anti-Affinity is critical for high availability. HPA with custom metrics reflects load better than CPU-based scaling. Behavior configuration prevents flapping. Ingress requires timeout adjustments for long-running operations. The standard 60 seconds is insufficient. Secrets management requires encryption at rest, external secret management, and rotation procedures.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Deployment
**Previous:** [[01_Deployment_Strategies|Deployment Strategies]]
**Next:** [[03_CI_CD_and_Self_Hosted|CI/CD and Self Hosted]]

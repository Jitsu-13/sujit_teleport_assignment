DOCUMENTS = [
    {
        "id": "doc_01",
        "title": "Horizontal Scaling and Auto-Scaling Policies",
        "text": (
            "Horizontal scaling adds more compute instances to distribute incoming traffic "
            "across a larger pool of workers. Auto-scaling policies monitor CPU utilization, "
            "request queue depth, and custom metrics to trigger scale-out events during peak "
            "load. Cloud providers expose managed instance groups or node pools that respond "
            "to scaling signals within seconds, ensuring the system absorbs sudden traffic "
            "spikes without degraded latency or dropped requests."
        ),
    },
    {
        "id": "doc_02",
        "title": "Load Balancing Strategies",
        "text": (
            "A load balancer distributes client requests across backend replicas using "
            "algorithms such as round-robin, least-connections, or consistent hashing. "
            "Layer-7 load balancers inspect HTTP headers and route traffic based on path or "
            "host, enabling blue-green deployments and canary releases. Health checks remove "
            "unhealthy nodes from the rotation, preventing cascading failures when individual "
            "instances become overloaded during high-traffic periods."
        ),
    },
    {
        "id": "doc_03",
        "title": "Caching Layers and Cache Invalidation",
        "text": (
            "In-memory caches such as Redis or Memcached store frequently accessed results "
            "close to the application layer, reducing database round-trips and response "
            "latency. A write-through cache updates the cache and the backing store "
            "simultaneously, maintaining consistency. Cache invalidation strategies — "
            "time-to-live expiry, event-driven purging, and LRU eviction — prevent stale "
            "data from being served while keeping the cache hit ratio high under sustained load."
        ),
    },
    {
        "id": "doc_04",
        "title": "Message Queues and Asynchronous Processing",
        "text": (
            "Message queues such as Pub/Sub, Kafka, or RabbitMQ decouple producers from "
            "consumers, acting as a buffer during traffic bursts. When inbound request rates "
            "exceed processing capacity, messages accumulate in the queue rather than "
            "overwhelming downstream services. Consumer groups can be scaled independently "
            "to drain backlogs, and dead-letter queues capture failed messages for later "
            "inspection and reprocessing, improving overall system resilience."
        ),
    },
    {
        "id": "doc_05",
        "title": "Circuit Breakers and Fault Tolerance",
        "text": (
            "The circuit breaker pattern prevents a failing dependency from causing a "
            "cascading failure across the entire service mesh. When error rates exceed a "
            "configurable threshold, the circuit opens and fast-fails subsequent calls, "
            "returning a fallback response instead of waiting for a timeout. After a "
            "cooldown window the circuit moves to half-open state, allowing a probe request "
            "through. If the probe succeeds the circuit closes; otherwise it reopens, "
            "protecting the system under partial outage conditions."
        ),
    },
    {
        "id": "doc_06",
        "title": "Database Sharding and Read Replicas",
        "text": (
            "Database sharding partitions a large dataset across multiple nodes by a shard "
            "key, distributing both storage and query load horizontally. Read replicas "
            "offload SELECT-heavy workloads from the primary, improving read throughput "
            "without sacrificing write consistency. During peak traffic windows, directing "
            "read queries to replicas keeps primary CPU headroom available for write "
            "operations, and connection poolers such as PgBouncer limit the number of "
            "concurrent connections to prevent resource exhaustion."
        ),
    },
    {
        "id": "doc_07",
        "title": "Rate Limiting and Throttling",
        "text": (
            "Rate limiting enforces an upper bound on the number of requests a client can "
            "make within a time window, protecting backend services from abusive traffic "
            "patterns or runaway clients. Token-bucket and sliding-window algorithms offer "
            "different burst-tolerance characteristics. When a client exceeds its quota, "
            "the gateway returns HTTP 429 responses, signalling the client to back off. "
            "Adaptive throttling adjusts limits dynamically based on downstream service "
            "health, preventing overload while maximising legitimate throughput."
        ),
    },
    {
        "id": "doc_08",
        "title": "Vector Embeddings and Semantic Search",
        "text": (
            "Dense vector embeddings encode the semantic meaning of text into high-dimensional "
            "floating-point vectors. Models such as sentence-transformers map sentences to a "
            "shared embedding space where cosine similarity reflects semantic relatedness. "
            "Approximate nearest-neighbour indexes — FAISS IVF, HNSW — retrieve the top-K "
            "most similar vectors in sub-linear time, enabling semantic search over millions "
            "of document chunks. Vertex AI Vector Search (Matching Engine) provides a "
            "managed, production-grade ANN service backed by Google's ScaNN algorithm."
        ),
    },
    {
        "id": "doc_09",
        "title": "Retrieval-Augmented Generation Architecture",
        "text": (
            "Retrieval-Augmented Generation combines a retriever component with a generative "
            "language model. At query time the retriever fetches the most semantically "
            "relevant document chunks from a vector store and injects them into the model's "
            "context window as grounding evidence. This grounds the model's output in "
            "factual, up-to-date information without requiring fine-tuning, and reduces "
            "hallucinations. The quality of the retrieved context directly determines the "
            "accuracy and faithfulness of the generated response."
        ),
    },
    {
        "id": "doc_10",
        "title": "Observability and SLO-Driven Capacity Planning",
        "text": (
            "Observability combines metrics, logs, and distributed traces to give operators "
            "a real-time view of system behaviour under load. Service-level objectives define "
            "latency and error-rate targets; when SLO burn rates spike, alerting pipelines "
            "page on-call engineers. Capacity planning uses historical traffic patterns and "
            "load-test results to forecast the compute budget required to meet SLOs at "
            "projected peak traffic, informing decisions about reserved instances, committed "
            "use discounts, and pre-warming auto-scaling groups before known high-traffic events."
        ),
    },
]

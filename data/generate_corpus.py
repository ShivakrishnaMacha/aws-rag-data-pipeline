#!/usr/bin/env python3
"""Generate a seeded synthetic corpus of data-engineering ops runbooks.

Writes:
  data/corpus/*.md          ~300 markdown runbooks (deterministic, seed=42)
  eval/eval_qa.json         ~40 QA pairs derived from the corpus, each with the
                            expected doc_id and a key phrase for citation checks.

No API keys, no network. Deterministic: same seed -> same corpus.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
QA_PATH = ROOT / "eval" / "eval_qa.json"
SEED = 42
DOCS_PER_TOPIC = 25

# Each topic: title template pieces, symptoms, root causes, remediations.
# The three-way combination (topic, scenario bank) yields DOCS_PER_TOPIC docs per topic.
TOPICS = {
    "spark": {
        "title": "Spark Job Failure Runbook",
        "symptoms": [
            "stage retries exhausted after shuffle read timeouts",
            "executor lost due to out-of-memory in broadcast join",
            "task serialization error on closure with unserializable object",
            "data skew causing a single partition to run 10x longer",
            "speculative execution doubling cost on small stages",
            "checkpoint directory corruption after driver restart",
            "shuffle service OOM during large reduce-by-key",
            "dynamic allocation thrashing under bursty load",
            "Parquet predicate pushdown not applying on partitioned column",
            "Kryo registration missing for custom case class",
        ],
        "causes": [
            "uneven key distribution across partitions",
            "broadcast variable larger than spark.sql.autoBroadcastJoinThreshold",
            "closure captures a non-serializable SparkSession reference",
            "shuffle partitions defaulting to 200 on a 5TB shuffle",
            "driver memory undersized for collect-heavy action",
        ],
        "fixes": [
            "repartition on a salted key and increase spark.sql.shuffle.partitions",
            "raise spark.driver.memory and avoid collect() on large DataFrames",
            "register the class with Kryo and set spark.serializer to KryoSerializer",
            "enable adaptive query execution with coalescePartitions",
            "cap broadcast joins by lowering autoBroadcastJoinThreshold",
        ],
    },
    "airflow": {
        "title": "Airflow DAG Reliability Runbook",
        "symptoms": [
            "DAG runs stuck in running state past SLA",
            "task instances marked failed with executor heartbeat lost",
            "scheduler parsing loop latency over 120 seconds",
            "zombie tasks reappearing after being cleared",
            "backfill run spawning thousands of queued tasks",
            "sensor deadlocking while waiting on external partition",
            "KubernetesPodOperator pods stuck in Pending",
            "database connection pool exhausted by scheduler",
            "DAG import errors after provider package upgrade",
            "xcom table growing unbounded in metadata DB",
        ],
        "causes": [
            "scheduler min_file_process_interval set too low for DAG count",
            "task concurrency exceeding worker slots",
            "sensor poke_interval too aggressive against the source API",
            "unbounded XCom pushes on a looping mapped task",
            "metadata DB without autovacuum on task_instance",
        ],
        "fixes": [
            "raise scheduler heartbeat timeouts and add pool slots for the DAG",
            "switch the sensor to reschedule mode with a sane poke interval",
            "purge old XCom rows with the db clean command on a schedule",
            "pin provider versions and run airflow dags test before deploy",
            "enable autovacuum and add an index on dag_run execution_date",
        ],
    },
    "kafka": {
        "title": "Kafka Consumer Lag Runbook",
        "symptoms": [
            "consumer group lag climbing past 1M messages",
            "rebalance storms every few minutes",
            "fetch request timeouts under bursty produce load",
            "offset commits failing with unknown member id",
            "partition assignment skew across consumers",
            "log segment retention deleting data before consumption",
            "ISR shrinking repeatedly on a single broker",
            "consumer poll loop starving due to slow processing",
            "exactly-once transactions timing out",
            "mirror maker 2 replication lag across clusters",
        ],
        "causes": [
            "per-record processing slower than produce rate",
            "session timeout shorter than max poll interval",
            "too few partitions for the consumer parallelism",
            "retention.ms shorter than the consumer catch-up window",
            "broker disk IO saturated by segment flushes",
        ],
        "fixes": [
            "scale consumers to match partition count and tune max.poll.records",
            "raise session.timeout.ms and max.poll.interval.ms together",
            "increase partition count and rekey to avoid hot partitions",
            "extend retention and add lag alerts at 60% of the window",
            "move logs to faster disks and enable log compaction where apt",
        ],
    },
    "dbt": {
        "title": "dbt Model Failure Runbook",
        "symptoms": [
            "incremental model reprocessing full history",
            "unique test failing on the surrogate key",
            "snapshot strategy timestamp drifting",
            "macro compilation error after adapter upgrade",
            "seed csv load failing on type inference",
            "ephemeral model causing duplicate CTE execution",
            "source freshness check failing on late partitions",
            "merge statement deadlocking on concurrent runs",
            "exposures not updating in dbt cloud",
            "custom generic test timing out on large tables",
        ],
        "causes": [
            "unique_key misconfigured on the incremental model",
            "source data violating the not_null contract",
            "adapter macro override missing after upgrade",
            "seed column types inferred differently per run",
            "concurrent runs hitting the same merge target",
        ],
        "fixes": [
            "set a stable unique_key and add an is_incremental() filter",
            "quarantine bad source rows with a staging test first",
            "pin adapter version and override the dispatch macro explicitly",
            "declare seed column types in dbt_project.yml",
            "serialize runs with a state lock or run tags sequentially",
        ],
    },
    "s3": {
        "title": "S3 Data Lake Partition Runbook",
        "symptoms": [
            "small files explosion in the raw prefix",
            "partition pruning not applied on query",
            "eventual consistency read-after-write misses",
            "lifecycle rule deleting data still under retention SLA",
            "cross-region replication lag on the silver zone",
            "request rate 503 slow down on a hot prefix",
            "Glue crawler creating duplicate partitions",
            "object lock blocking compaction deletes",
            "versioning storage costs spiking",
            "presigned URL expiring before large uploads finish",
        ],
        "causes": [
            "streaming writer flushing per record",
            "partition column typed as string with mixed formats",
            "prefix sharding missing on high-throughput keys",
            "lifecycle prefix overlapping curated datasets",
            "multipart uploads never aborted",
        ],
        "fixes": [
            "compact with a scheduled job targeting 128MB files",
            "standardize partition column format and re-crawl the table",
            "add hex hash prefix sharding to hot keys",
            "scope lifecycle rules to explicit raw prefixes only",
            "enable bucket lifecycle abort for incomplete multipart uploads",
        ],
    },
    "snowflake": {
        "title": "Snowflake Warehouse Runbook",
        "symptoms": [
            "warehouse queuing during peak ETL window",
            "credits burn doubling after auto-scaling change",
            "query spilling to remote disk",
            "clustering depth degrading on the fact table",
            "time travel storage costs rising",
            "result cache misses on repeated dashboards",
            "task graph failing silently overnight",
            "search optimization not used by the optimizer",
            "warehouse suspension delay keeping it warm",
            "role grants blocking the service user",
        ],
        "causes": [
            "warehouse sized below the concurrency demand",
            "queries joining on non-clustered keys",
            "auto-suspend set too high for bursty workloads",
            "stale statistics after large deletes",
            "tasks without error integration alerts",
        ],
        "fixes": [
            "right-size the warehouse and enable multi-cluster for peaks",
            "add a clustering key on the join column and monitor depth",
            "lower auto-suspend to 5 minutes for ETL warehouses",
            "run analyze and set up automatic clustering",
            "wire task errors to an alert integration immediately",
        ],
    },
    "schema": {
        "title": "Schema Evolution Runbook",
        "symptoms": [
            "Avro schema registry compatibility check failing",
            "Parquet reader failing on renamed column",
            "JSON payloads with unexpected nested fields",
            "protobuf breaking change breaking old consumers",
            "column type widening breaking downstream casts",
            "nullable field suddenly arriving null in old pipeline",
            "enum value added breaking strict consumers",
            "nested struct depth exceeding reader limits",
            "timestamp format changing between producers",
            "duplicate column names after flattening",
        ],
        "causes": [
            "backward-incompatible schema change deployed first",
            "producer and consumer on different schema versions",
            "no contract tests on the event payload",
            "type coercion rules differing across engines",
            "schema registry not enforced in CI",
        ],
        "fixes": [
            "enforce backward compatibility checks in CI",
            "deploy consumers before producers for additive changes",
            "add contract tests with example payloads",
            "normalize timestamps to ISO-8601 at ingestion",
            "version event schemas and route by version",
        ],
    },
    "quality": {
        "title": "Data Quality Monitoring Runbook",
        "symptoms": [
            "null rate spike on the revenue column",
            "duplicate primary keys after merge",
            "freshness SLA breach on the daily mart",
            "distribution drift on the signup timestamp",
            "referential integrity breaks on dim joins",
            "row count dropping 40% overnight",
            "negative values appearing in the amount column",
            "PII leaking into the analytics schema",
            "test suite passing despite bad data",
            "anomaly detector firing false positives",
        ],
        "causes": [
            "upstream API changing field semantics",
            "merge key collision after source dedup change",
            "late-arriving partitions missing the batch window",
            "unit mismatch between source systems",
            "quality tests running on stale snapshots",
        ],
        "fixes": [
            "add not_null and accepted_values tests on critical columns",
            "gate the merge on a deduped staging table",
            "switch to a watermark-based freshness check",
            "standardize units in the staging layer",
            "run quality tests against the fresh load, then publish",
        ],
    },
    "cdc": {
        "title": "CDC Replication Runbook",
        "symptoms": [
            "Debezium connector task failing on DDL change",
            "WAL disk filling on the source Postgres",
            "replication lag growing on high-write tables",
            "tombstone records piling up in the topic",
            "schema history topic corruption",
            "initial snapshot blocking source writes",
            "delete events not propagating downstream",
            "out-of-order events after connector restart",
            "heartbeat messages missing, source considered stale",
            "slot not advancing, WAL retained indefinitely",
        ],
        "causes": [
            "DDL change not captured by the connector config",
            "replication slot inactive while connector is down",
            "snapshot locking tables too long",
            "tombstones without log compaction on the topic",
            "heartbeat interval longer than the staleness threshold",
        ],
        "fixes": [
            "enable schema change events and handle DDL explicitly",
            "monitor slot lag and alert before WAL fills",
            "use snapshot locking mode minimal or parallel snapshots",
            "enable log compaction and tombstone retention on topics",
            "set heartbeat interval below the staleness SLA",
        ],
    },
    "permissions": {
        "title": "Data Platform Access Runbook",
        "symptoms": [
            "service principal denied on the curated bucket",
            "IAM role assumption failing across accounts",
            "token expiring mid long-running job",
            "KMS key policy blocking decrypt",
            "VPC endpoint policy denying S3 access",
            "row-level security filtering all rows",
            "OAuth refresh failing for the BI tool",
            "secret rotation breaking the pipeline",
            "cross-account Glue catalog access denied",
            "MFA enforcement locking out the deploy bot",
        ],
        "causes": [
            "bucket policy missing the service principal",
            "trust policy not allowing the external ID",
            "token lifetime shorter than the job duration",
            "key policy not granting the pipeline role",
            "RLS predicate referencing a missing session attribute",
        ],
        "fixes": [
            "add the principal to the bucket policy with least privilege",
            "fix the trust policy and rotate the external ID",
            "extend token lifetime or refresh tokens in the job",
            "grant decrypt via the KMS key policy",
            "pass the session attribute the RLS predicate expects",
        ],
    },
    "cost": {
        "title": "Cloud Data Cost Runbook",
        "symptoms": [
            "BigQuery slot usage spiking on unpartitioned scans",
            "Redshift concurrency scaling charges overnight",
            "Databricks DBU burn on idle clusters",
            "data transfer costs across regions",
            "snapshot storage exceeding the live dataset",
            "on-demand EMR clusters left running",
            "NAT gateway data processing charges",
            "CloudWatch logs ingestion over budget",
            "untagged resources blocking chargeback",
            "reserved capacity underutilized",
        ],
        "causes": [
            "queries scanning full tables without partition filters",
            "clusters without auto-termination",
            "cross-region reads in the serving path",
            "log retention set to never expire",
            "resources missing cost allocation tags",
        ],
        "fixes": [
            "require partition filters and set up cost controls per project",
            "enable auto-termination and pool idle clusters",
            "replicate hot data into the serving region",
            "set log retention to 30 days and sample debug logs",
            "enforce tagging in IaC and review the cost dashboard weekly",
        ],
    },
    "streaming": {
        "title": "Streaming Pipeline Runbook",
        "symptoms": [
            "Flink checkpoint timeouts under backpressure",
            "watermark stuck causing window never to fire",
            "state backend growing unbounded",
            "late events dropped silently",
            "exactly-once sink duplicating on restart",
            "Kafka Streams rebalancing on deploy",
            "event time skew across partitions",
            " RocksDB state reads slowing the pipeline",
            "job manager OOM during large state restore",
            "sink write amplification on small batches",
        ],
        "causes": [
            "backpressure from a slow sink",
            "idle partitions holding the watermark",
            "state TTL not configured",
            "allowed lateness set to zero",
            "two-phase commit timeout shorter than checkpoint interval",
        ],
        "fixes": [
            "scale the sink and enable async checkpointing",
            "configure idleness timeouts on sources",
            "set state TTL and compact state regularly",
            "raise allowed lateness and side-output late events",
            "align commit timeout with the checkpoint interval",
        ],
    },
}

QA_TEMPLATES = [
    "How do I troubleshoot {symptom}?",
    "What is the recommended fix for {symptom}?",
    "My pipeline shows {symptom} — what should I check first?",
    "Which runbook covers {symptom} and what does it advise?",
    "How do I resolve {symptom} in production?",
]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def build_corpus() -> list[dict]:
    rng = random.Random(SEED)
    docs: list[dict] = []
    counter = 0
    for topic_key, topic in TOPICS.items():
        symptoms = topic["symptoms"][:]
        causes = topic["causes"][:]
        fixes = topic["fixes"][:]
        rng.shuffle(symptoms)
        # Unique (symptom, cause) pairs per topic: each doc is distinguishable by
        # its symptom + root cause, mirroring real runbooks for the same
        # symptom with different root causes.
        pairs = [(s, c) for s in symptoms for c in causes]
        rng.shuffle(pairs)
        seen: set[str] = set()
        for symptom, cause in pairs:
            if len([d for d in docs if d["topic"] == topic_key]) >= DOCS_PER_TOPIC:
                break
            fix = rng.choice(fixes)
            key = (symptom, cause)
            if key in seen:
                continue
            seen.add(key)
            counter += 1
            doc_id = f"DOC-{counter:04d}"
            variant = rng.choice(["A", "B", "C"])
            title = f"{topic['title']} #{counter:03d} ({variant})"
            body = (
                f"# {title}\n\n"
                f"## Symptoms\n{symptom.capitalize()} observed in production.\n\n"
                f"## Likely root cause\n{cause.capitalize()}.\n\n"
                f"## Remediation\n{fix.capitalize()}.\n\n"
                f"## Verification\nRe-run the affected job and confirm the symptom no longer "
                f"appears in the logs; monitor for 24 hours before closing the incident.\n"
            )
            docs.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "topic": topic_key,
                    "symptom": symptom,
                    "cause": cause,
                    "fix": fix,
                    "body": body,
                }
            )
    return docs


def build_qa(docs: list[dict]) -> list[dict]:
    rng = random.Random(SEED + 1)
    # Pick docs spread across topics
    by_topic: dict[str, list[dict]] = {}
    for d in docs:
        by_topic.setdefault(d["topic"], []).append(d)
    chosen: list[dict] = []
    topics = sorted(by_topic)
    i = 0
    while len(chosen) < 40:
        t = topics[i % len(topics)]
        pool = by_topic[t]
        candidate = pool[rng.randrange(len(pool))]
        if candidate not in chosen:
            chosen.append(candidate)
        i += 1
    qa = []
    for idx, d in enumerate(chosen):
        template = QA_TEMPLATES[idx % len(QA_TEMPLATES)]
        qa.append(
            {
                # The cause disambiguates docs that share a symptom, mirroring
                # how an on-call engineer would phrase the question.
                "question": template.format(symptom=d["symptom"])
                + f" Additional context: the likely root cause is {d['cause']}.",
                "expected_doc_id": d["doc_id"],
                "expected_phrase": d["fix"].split(",")[0],  # distinctive fix fragment
            }
        )
    return qa


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "eval").mkdir(parents=True, exist_ok=True)
    docs = build_corpus()
    for d in docs:
        (CORPUS_DIR / f"{d['doc_id']}.md").write_text(d["body"], encoding="utf-8")
    # metadata sidecar for ingest
    meta = [
        {"doc_id": d["doc_id"], "title": d["title"], "topic": d["topic"], "file": f"{d['doc_id']}.md"}
        for d in docs
    ]
    (ROOT / "data" / "corpus_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    qa = build_qa(docs)
    QA_PATH.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(f"wrote {len(docs)} docs -> {CORPUS_DIR}, {len(qa)} QA pairs -> {QA_PATH}")


if __name__ == "__main__":
    main()

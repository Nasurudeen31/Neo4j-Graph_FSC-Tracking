"""Evaluate five configurations on the expanded 50-batch synthetic dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

from app import (
    Services, call_ollama, deterministic_answer, event_sort_key, now_iso,
    retrieve_batch, validate_answer, verify_evidence,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_expanded"
OUTPUT_DIR = ROOT / "outputs" / "expanded_baseline_evaluation"

CONFIGURATIONS = {
    "B1_CYPHER_TEMPLATE": "Pure Cypher + deterministic template",
    "B2_DOCUMENT_RAG": "Classical document-based RAG",
    "B3_GRAPHRAG_NO_VERIFY": "GraphRAG without blockchain verification",
    "B4_VERIFIED_NO_FALLBACK": "Verified GraphRAG without fallback",
    "B5_FULL_FRAMEWORK": "Full proposed framework",
}
STRATIFIED_BATCHES = [
    "BATCH-FISH-001", "BATCH-FISH-007",
    "BATCH-MEAT-014", "BATCH-MEAT-015",
    "BATCH-VEG-023", "BATCH-VEG-030",
    "BATCH-DAIRY-036", "BATCH-DAIRY-038",
    "BATCH-PROCFOOD-042", "BATCH-PROCFOOD-049",
]
CITATION_RE = re.compile(r"\[(EV-[A-Z]+-[0-9]+)\]")
TOKEN_RE = re.compile(r"[A-Za-z0-9-]+")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_cases() -> dict[str, dict]:
    batches = {row["batch_id"]: row for row in read_csv(DATA_DIR / "batches.csv")}
    truths = {row["batch_id"]: row for row in read_csv(DATA_DIR / "ground_truth.csv")}
    if set(batches) != set(truths):
        raise RuntimeError("batches.csv and ground_truth.csv contain different batch IDs")
    cases = {}
    for batch_id, batch in batches.items():
        truth = truths[batch_id]
        question = (
            f"Reconstruct the complete available traceability record for {batch_id} "
            f"({batch['product']}), report its status, and identify any missing, "
            "conflicting, inconsistent-identifier, inspection, or cold-chain evidence."
        )
        cases[batch_id] = {"batch": batch, "truth": truth, "question": question}
    return cases


def graph_evidence(graph_result: dict) -> list[dict]:
    items = []
    for item in graph_result["evidence"]:
        items.append({**item, "document": json.loads(item["canonical_json"])})
    return sorted(items, key=event_sort_key)


def all_documents(services: Services) -> list[dict]:
    query = """MATCH (e:Evidence {dataset_id:'expanded-50-v1'}) RETURN properties(e) AS e
               ORDER BY e.evidence_id"""
    with services.neo4j.session(database=services.database) as session:
        records = list(session.run(query))
    return [{**r["e"], "document": json.loads(r["e"]["canonical_json"])} for r in records]


def tokenize(text: str) -> list[str]:
    return [x.casefold() for x in TOKEN_RE.findall(text)]


def tfidf_retrieve(question: str, corpus: list[dict], top_k: int) -> list[dict]:
    docs = [tokenize(json.dumps(x["document"], sort_keys=True)) for x in corpus]
    query = tokenize(question)
    df = Counter()
    for tokens in docs: df.update(set(tokens))
    n = len(corpus)
    def vector(tokens):
        counts = Counter(tokens)
        return {t: c * (math.log((n + 1) / (df[t] + 1)) + 1) for t, c in counts.items()}
    qv = vector(query)
    def cosine(a, b):
        denom = math.sqrt(sum(v*v for v in a.values())) * math.sqrt(sum(v*v for v in b.values()))
        return sum(v * b.get(k, 0) for k, v in a.items()) / denom if denom else 0.0
    scored = [(cosine(qv, vector(tokens)), item["evidence_id"], item)
              for item, tokens in zip(corpus, docs)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [x[2] for x in scored[:top_k]]


def call_document_model(services: Services, question: str, evidence: list[dict]) -> str:
    allowed = [x["evidence_id"] for x in evidence]
    lines = [f"[{x['evidence_id']}] {json.dumps(x['document'], ensure_ascii=False, sort_keys=True)}"
             for x in evidence]
    response = requests.post(
        services.ollama_api,
        json={"model": services.ollama_model, "stream": False,
              "messages": [{"role": "user", "content":
                  "Use only these documents. Cite every factual sentence using an allowed ID. "
                  f"Allowed IDs: {allowed}\nQuestion: {question}\n" + "\n".join(lines)}],
              "options": {"temperature": 0.0, "seed": 42, "num_predict": 500}},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def retry_model_call(function, attempts: int = 3, delay_seconds: int = 10) -> str:
    """Retry transient Ollama/HTTP failures without losing experiment progress."""
    for attempt in range(1, attempts + 1):
        try:
            return function()
        except (requests.RequestException, TimeoutError, ConnectionError) as exc:
            if attempt == attempts:
                raise
            print(f"Temporary model error ({type(exc).__name__}); retry "
                  f"{attempt}/{attempts - 1} after {delay_seconds} seconds")
            time.sleep(delay_seconds)
    raise RuntimeError("Unreachable retry state")


def metric_values(retrieved: set[str], expected: set[str], answer: str) -> dict:
    cited = set(CITATION_RE.findall(answer)); correct = cited & expected
    return {
        "retrieval_completeness": len(retrieved & expected) / len(expected) if expected else 1.0,
        "citation_precision": len(correct) / len(cited) if cited else 0.0,
        "citation_recall": len(correct) / len(expected) if expected else 1.0,
        "retrieved_ids": ";".join(sorted(retrieved)),
        "expected_ids": ";".join(sorted(expected)),
        "cited_ids": ";".join(sorted(cited)),
    }


def run_one(services: Services, config_id: str, batch_id: str, case: dict,
            corpus: list[dict], start_condition: str) -> dict:
    total_start = time.perf_counter()
    start = time.perf_counter(); graph_result = retrieve_batch(services, batch_id)
    if graph_result is None: raise RuntimeError(f"Batch not found: {batch_id}")
    graph_items = graph_evidence(graph_result)
    expected_ids = set(filter(None, case["truth"]["expected_evidence_ids"].split(";")))
    if config_id == "B2_DOCUMENT_RAG":
        retrieved = tfidf_retrieve(case["question"], corpus, max(1, len(expected_ids)))
    else: retrieved = graph_items
    retrieval_ms = (time.perf_counter() - start) * 1000
    verification_ms = generation_ms = 0.0
    integrity_passed = None; validation_passed = False; fallback_used = False
    if config_id == "B1_CYPHER_TEMPLATE":
        candidate = answer = deterministic_answer(graph_result, retrieved)
        validation_passed = True
    elif config_id in {"B2_DOCUMENT_RAG", "B3_GRAPHRAG_NO_VERIFY"}:
        start = time.perf_counter()
        candidate = retry_model_call(
            lambda: call_document_model(services, case["question"], retrieved)
            if config_id == "B2_DOCUMENT_RAG"
            else call_ollama(services, case["question"], retrieved)
        )
        generation_ms = (time.perf_counter() - start) * 1000
        answer = candidate; validation_passed = validate_answer(candidate, retrieved)["passed"]
    else:
        start = time.perf_counter(); verified = verify_evidence(services, graph_result)
        verification_ms = (time.perf_counter() - start) * 1000
        integrity_passed = all(x["verified"] for x in verified)
        if not integrity_passed: raise RuntimeError(f"Integrity verification failed: {batch_id}")
        start = time.perf_counter()
        candidate = retry_model_call(
            lambda: call_ollama(services, case["question"], verified)
        )
        generation_ms = (time.perf_counter() - start) * 1000
        validation_passed = validate_answer(candidate, verified)["passed"]
        if config_id == "B5_FULL_FRAMEWORK" and not validation_passed:
            answer = deterministic_answer(graph_result, verified); fallback_used = True
        else: answer = candidate
    total_ms = (time.perf_counter() - total_start) * 1000
    return {
        "created_at": now_iso(), "configuration_id": config_id,
        "configuration": CONFIGURATIONS[config_id], "start_condition": start_condition,
        "batch_id": batch_id, "category": case["batch"]["category"],
        "scenario_type": case["batch"]["scenario_type"], "question": case["question"],
        "candidate_answer": candidate.replace("\n", " "), "released_answer": answer.replace("\n", " "),
        "integrity_passed": "" if integrity_passed is None else int(integrity_passed),
        "validation_passed": int(validation_passed), "fallback_used": int(fallback_used),
        "retrieval_ms": round(retrieval_ms, 3), "verification_ms": round(verification_ms, 3),
        "generation_ms": round(generation_ms, 3), "total_ms": round(total_ms, 3),
        **metric_values({x["evidence_id"] for x in retrieved}, expected_ids, answer),
    }


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values: return 0.0
    k = (len(values) - 1) * p; lo = math.floor(k); hi = math.ceil(k)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (k - lo)


def stats(values: list[float]) -> dict:
    n = len(values); mean = statistics.fmean(values) if values else 0.0
    sd = statistics.stdev(values) if n > 1 else 0.0
    margin = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "median": statistics.median(values), "sd": sd,
            "min": min(values), "max": max(values), "p95": percentile(values, .95),
            "ci95_low": mean - margin, "ci95_high": mean + margin}


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows: groups[(row["configuration_id"], row["start_condition"])].append(row)
    output = []
    for (config_id, condition), group in groups.items():
        latency = stats([float(x["total_ms"]) for x in group])
        output.append({
            "configuration": CONFIGURATIONS[config_id], "start_condition": condition,
            "runs": latency.pop("n"),
            "retrieval_completeness_percent": round(100*statistics.fmean(float(x["retrieval_completeness"]) for x in group), 2),
            "citation_precision_percent": round(100*statistics.fmean(float(x["citation_precision"]) for x in group), 2),
            "citation_recall_percent": round(100*statistics.fmean(float(x["citation_recall"]) for x in group), 2),
            "fallback_rate_percent": round(100*statistics.fmean(float(x["fallback_used"]) for x in group), 2),
            **{f"total_ms_{k}": round(v, 2) for k, v in latency.items()},
        })
    return output


def create_review_sheet(rows: list[dict]) -> None:
    unique = {}
    for row in rows:
        key = (row["configuration_id"], row["batch_id"], row["released_answer"])
        unique[key] = row
    review = []
    for number, row in enumerate(unique.values(), 1):
        review.append({"review_id": number, "configuration_id": row["configuration_id"],
                       "batch_id": row["batch_id"], "answer": row["released_answer"],
                       "answer_correct_0_or_1": "", "factual_claim_count": "",
                       "unsupported_claim_count": "", "reviewer_notes": ""})
    write_csv(OUTPUT_DIR / "manual_claim_review.csv", review)


def result_key(row: dict) -> tuple[str, str, str, int]:
    return (
        row["configuration_id"], row["batch_id"], row["start_condition"],
        int(row["repetition"]),
    )


def save_checkpoint(rows: list[dict]) -> None:
    """Persist progress after every successful configuration execution."""
    raw_path = OUTPUT_DIR / "baseline_raw_results.csv"
    write_csv(raw_path, rows)
    write_csv(OUTPUT_DIR / "baseline_summary.csv", summarize(rows))


def run_experiment(repetitions: int, cold_repetitions: int,
                   selected_batch: str | None, stratified: bool, fresh: bool) -> None:
    cases = load_cases()
    if selected_batch:
        if selected_batch not in cases:
            raise RuntimeError(f"Unknown batch ID: {selected_batch}")
        cases = {selected_batch: cases[selected_batch]}
    elif stratified:
        cases = {batch_id: cases[batch_id] for batch_id in STRATIFIED_BATCHES}
        print("Using stratified 10-batch subset:", ", ".join(cases))
    services = Services(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        corpus = all_documents(services)
        if len(corpus) != 310:
            raise RuntimeError(f"Expected 310 Evidence nodes, found {len(corpus)}. Run seed-expanded first.")
        raw_path = OUTPUT_DIR / "baseline_raw_results.csv"
        rows = [] if fresh or not raw_path.exists() else read_csv(raw_path)
        completed = {result_key(row) for row in rows}
        if rows:
            print(f"Resuming with {len(rows)} completed executions.")
        # Cold-start designation: the first measured pass after services are started.
        for repetition in range(1, cold_repetitions + repetitions + 1):
            condition = "cold" if repetition <= cold_repetitions else "warm"
            measured_rep = repetition if condition == "cold" else repetition - cold_repetitions
            for batch_id, case in cases.items():
                for config_id in CONFIGURATIONS:
                    key = (config_id, batch_id, condition, measured_rep)
                    if key in completed:
                        print(f"SKIP completed | {condition} {measured_rep} | "
                              f"{batch_id} | {CONFIGURATIONS[config_id]}")
                        continue
                    print(f"{condition} {measured_rep} | {batch_id} | {CONFIGURATIONS[config_id]}")
                    row = run_one(services, config_id, batch_id, case, corpus, condition)
                    row["repetition"] = measured_rep
                    rows.append(row); completed.add(key)
                    save_checkpoint(rows)
                    print(f"Checkpoint saved: {len(rows)} executions")
        save_checkpoint(rows)
        create_review_sheet(rows)
        print("Experiment completed:", OUTPUT_DIR)
    finally: services.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=10,
                        help="Warm-start repetitions per batch/configuration")
    parser.add_argument("--cold-repetitions", type=int, default=1,
                        help="Separately labelled cold-start passes")
    parser.add_argument("--batch", default=None,
                        help="Optional single batch ID for a short smoke test")
    parser.add_argument("--stratified", action="store_true",
                        help="Repeat a predefined 10-batch, five-category subset")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore prior raw results and start a new experiment")
    args = parser.parse_args()
    if args.repetitions < 1 or args.cold_repetitions < 0: parser.error("Invalid repetition count")
    if args.batch and args.stratified:
        parser.error("Use either --batch or --stratified, not both")
    run_experiment(args.repetitions, args.cold_repetitions,
                   args.batch, args.stratified, args.fresh)


if __name__ == "__main__":
    main()

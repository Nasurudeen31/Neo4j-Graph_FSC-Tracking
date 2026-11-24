"""Blockchain-ablation experiment for the FoodTraceGraphRAG prototype.

Run from the project directory after Neo4j, IPFS Desktop and Ganache are
running and after the graph has been seeded::

    python blockchain_ablation.py --batch BATCH-TUNA-001 --repetitions 10

The experiment does not modify Neo4j or the smart contract.  It works with
copies of one registered evidence record.  One altered test object is added
to IPFS so that a genuine content-derived CID can be used for the coordinated
local-alteration condition.

Outputs:
  outputs/blockchain_ablation/blockchain_ablation_trials.csv
  outputs/blockchain_ablation/blockchain_ablation_summary.csv
  outputs/blockchain_ablation/blockchain_ablation_by_fault.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from app import Services, canonical_bytes, retrieve_batch, sha256_hex


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "blockchain_ablation"

CONFIGURATIONS = (
    "Graph + IPFS",
    "Graph + IPFS + SHA-256",
    "Graph + IPFS + SHA-256 + blockchain",
    "Complete system",
)

FAULTS = (
    "modified_content",
    "altered_metadata",
    "missing_evidence",
    "incorrect_hash",
    "coordinated_content_cid_hash_change",
    "incomplete_evidence_set",
)


def percentile(values: list[float], p: float) -> float:
    """Return a linearly interpolated percentile without extra dependencies."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def make_altered_document(document: dict) -> dict:
    altered = copy.deepcopy(document)
    facts = dict(altered.get("facts", {}))
    facts["ablation_test_marker"] = "controlled-modification"
    altered["facts"] = facts
    return altered


def build_trial_state(clean_item: dict, altered_cid: str, altered_payload: bytes,
                      fault: str, expected_count: int) -> dict:
    """Create a copied trial state; never update the registered record."""
    clean_document = json.loads(clean_item["canonical_json"])
    clean_payload = canonical_bytes(clean_document)
    altered_document = json.loads(altered_payload)

    state = {
        "local_cid": clean_item["ipfs_cid"],
        "local_hash": clean_item["content_hash"],
        "local_evidence_id": clean_item["evidence_id"],
        "local_event_type": clean_item["event_type"],
        "local_batch_id": clean_item["batch_id"],
        "actual_cid": clean_item["ipfs_cid"],
        "actual_payload": clean_payload,
        "actual_document": clean_document,
        "available": True,
        "observed_count": expected_count,
        "expected_count": expected_count,
    }

    if fault == "modified_content":
        state["actual_cid"] = altered_cid
        state["actual_payload"] = altered_payload
        state["actual_document"] = altered_document
    elif fault == "altered_metadata":
        state["local_evidence_id"] = f"{clean_item['evidence_id']}-ALTERED"
    elif fault == "missing_evidence":
        state["available"] = False
        state["actual_payload"] = None
        state["actual_document"] = None
    elif fault == "incorrect_hash":
        state["local_hash"] = "0" * 64
    elif fault == "coordinated_content_cid_hash_change":
        # The mutable local record and content are changed consistently.
        # CID-only and local-hash checks can accept this state, while the
        # earlier blockchain commitment should reject it.
        state["local_cid"] = altered_cid
        state["local_hash"] = sha256_hex(altered_payload)
        state["actual_cid"] = altered_cid
        state["actual_payload"] = altered_payload
        state["actual_document"] = altered_document
    elif fault == "incomplete_evidence_set":
        state["observed_count"] = max(0, expected_count - 1)
    else:
        raise ValueError(f"Unknown fault: {fault}")

    return state


def verify_state(services: Services, state: dict, configuration: str) -> dict:
    """Apply only the checks enabled for one ablation configuration."""
    started = time.perf_counter()

    available_ok = bool(state["available"])
    completeness_ok = state["observed_count"] == state["expected_count"]
    cid_ok = False
    metadata_ok = False
    hash_ok = None
    blockchain_ok = None

    if available_ok:
        payload = state["actual_payload"]
        document = state["actual_document"]
        cid_ok = state["actual_cid"] == state["local_cid"]
        metadata_ok = (
            document["evidence_id"] == state["local_evidence_id"]
            and document["event_type"] == state["local_event_type"]
            and document["batch_id"] == state["local_batch_id"]
        )

        if configuration != "Graph + IPFS":
            hash_ok = sha256_hex(payload) == state["local_hash"]

        if configuration in (
            "Graph + IPFS + SHA-256 + blockchain",
            "Complete system",
        ):
            try:
                blockchain_ok = services.verify_anchor(
                    document, state["actual_cid"], payload
                )
            except Exception:
                blockchain_ok = False

    checks = [available_ok, completeness_ok, cid_ok, metadata_ok]
    if hash_ok is not None:
        checks.append(hash_ok)
    if blockchain_ok is not None:
        checks.append(blockchain_ok)

    accepted = all(checks)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "accepted": accepted,
        "detected": not accepted,
        "generation_blocked": not accepted,
        "available_ok": available_ok,
        "completeness_ok": completeness_ok,
        "cid_ok": cid_ok,
        "metadata_ok": metadata_ok,
        "hash_ok": hash_ok,
        "blockchain_ok": blockchain_ok,
        "verification_latency_ms": latency_ms,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(trials: list[dict]) -> tuple[list[dict], list[dict]]:
    by_configuration: dict[str, list[dict]] = defaultdict(list)
    by_fault: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in trials:
        by_configuration[row["configuration"]].append(row)
        by_fault[(row["configuration"], row["fault_type"])].append(row)

    summary_rows = []
    for configuration in CONFIGURATIONS:
        rows = by_configuration[configuration]
        injected = len(rows)
        detected = sum(int(row["detected"]) for row in rows)
        false_acceptances = sum(int(row["accepted"]) for row in rows)
        blocked = sum(int(row["generation_blocked"]) for row in rows)
        latencies = [float(row["verification_latency_ms"]) for row in rows]
        summary_rows.append({
            "configuration": configuration,
            "fault_trials": injected,
            "faults_detected": detected,
            "detection_rate_percent": round(100 * detected / injected, 2),
            "false_acceptances": false_acceptances,
            "false_acceptance_rate_percent": round(
                100 * false_acceptances / injected, 2
            ),
            "generation_blocked": f"{blocked}/{injected}",
            "mean_verification_latency_ms": round(statistics.mean(latencies), 3),
            "median_verification_latency_ms": round(statistics.median(latencies), 3),
            "p95_verification_latency_ms": round(percentile(latencies, 0.95), 3),
        })

    fault_rows = []
    for fault in FAULTS:
        row = {"fault_type": fault}
        for configuration in CONFIGURATIONS:
            trials_for_cell = by_fault[(configuration, fault)]
            detected = sum(int(item["detected"]) for item in trials_for_cell)
            total = len(trials_for_cell)
            row[configuration] = (
                f"Detected ({detected}/{total})"
                if detected == total
                else f"False acceptance ({total - detected}/{total})"
            )
        fault_rows.append(row)
    return summary_rows, fault_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="BATCH-FISH-001")
    parser.add_argument("--evidence-id", default=None)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
    )
    args = parser.parse_args()

    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")

    services = Services()
    try:
        graph_result = retrieve_batch(services, args.batch)
        if graph_result is None:
            raise RuntimeError(f"Batch not found: {args.batch}")
        evidence = graph_result["evidence"]
        if not evidence:
            raise RuntimeError(f"No evidence found for batch: {args.batch}")

        if args.evidence_id:
            matches = [
                item for item in evidence
                if item["evidence_id"] == args.evidence_id
            ]
            if not matches:
                raise RuntimeError(
                    f"Evidence ID {args.evidence_id!r} not found in {args.batch}"
                )
            clean_item = matches[0]
        else:
            clean_item = evidence[0]

        clean_document = json.loads(clean_item["canonical_json"])
        altered_document = make_altered_document(clean_document)
        altered_cid, altered_payload = services.ipfs_add(altered_document)
        expected_count = len(evidence)

        trials = []
        total = len(CONFIGURATIONS) * len(FAULTS) * args.repetitions
        index = 0
        for configuration in CONFIGURATIONS:
            for fault in FAULTS:
                for repetition in range(1, args.repetitions + 1):
                    index += 1
                    state = build_trial_state(
                        clean_item,
                        altered_cid,
                        altered_payload,
                        fault,
                        expected_count,
                    )
                    result = verify_state(services, state, configuration)
                    row = {
                        "configuration": configuration,
                        "fault_type": fault,
                        "repetition": repetition,
                        "batch_id": args.batch,
                        "evidence_id": clean_item["evidence_id"],
                        **result,
                    }
                    trials.append(row)
                    print(
                        f"[{index}/{total}] {configuration} | {fault} | "
                        f"detected={result['detected']} | "
                        f"latency={result['verification_latency_ms']:.3f} ms"
                    )

        summary_rows, fault_rows = summarize(trials)
        output_dir = args.output_dir
        write_csv(
            output_dir / "blockchain_ablation_trials.csv",
            trials,
            list(trials[0].keys()),
        )
        write_csv(
            output_dir / "blockchain_ablation_summary.csv",
            summary_rows,
            list(summary_rows[0].keys()),
        )
        write_csv(
            output_dir / "blockchain_ablation_by_fault.csv",
            fault_rows,
            ["fault_type", *CONFIGURATIONS],
        )

        print("\nManuscript-ready summary")
        for row in summary_rows:
            print(
                f"{row['configuration']}: "
                f"detected {row['faults_detected']}/{row['fault_trials']}, "
                f"FAR={row['false_acceptance_rate_percent']:.2f}%, "
                f"mean={row['mean_verification_latency_ms']:.3f} ms"
            )
        print(f"\nResults written to: {output_dir.resolve()}")
    finally:
        services.close()


if __name__ == "__main__":
    main()

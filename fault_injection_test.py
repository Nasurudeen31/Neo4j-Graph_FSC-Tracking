"""Controlled fault-injection evaluation for the expanded FoodTraceGraphRAG app.

Run only against the local synthetic ``expanded-50-v1`` dataset.  Every test
backs up the affected state and restores it in a ``finally`` block.  Results
are written to ``outputs/fault_injection`` as detailed and summary CSV files.

The test intentionally does not call Ollama: when an integrity or completeness
failure is detected, the release gate marks generation as blocked.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import requests

from app import ALLOWED_EVENT_TYPES, Services, retrieve_batch, verify_evidence


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "fault_injection"
DATASET_ID = "expanded-50-v1"


def neo4j_one(services: Services, query: str, **params) -> dict:
    with services.neo4j.session(database=services.database) as session:
        record = session.run(query, **params).single()
    if record is None:
        raise RuntimeError("Neo4j query returned no record")
    return dict(record)


def neo4j_write(services: Services, query: str, **params) -> None:
    with services.neo4j.session(database=services.database) as session:
        session.run(query, **params).consume()


def relation_ids(services: Services, batch_id: str) -> set[str]:
    """Return operational event relationships associated with a batch."""
    query = """
    MATCH (b:Batch {batch_id:$batch_id})-[:SUPPORTED_BY]->(e:Evidence)
    WITH collect(e.evidence_id) AS ids
    MATCH ()-[r]->()
    WHERE r.dataset_id=$dataset_id AND r.evidence_id IN ids
    RETURN collect(DISTINCT r.evidence_id) AS ids
    """
    row = neo4j_one(services, query, batch_id=batch_id, dataset_id=DATASET_ID)
    return set(row["ids"] or [])


def transaction_metadata_ok(services: Services, item: dict) -> bool:
    """Validate the stored transaction reference and block number."""
    try:
        receipt = services.w3.eth.get_transaction_receipt(item["transaction_hash"])
        return bool(
            receipt
            and receipt.status == 1
            and int(receipt.blockNumber) == int(item["block_number"])
        )
    except Exception:
        return False


def assess_state(
    services: Services,
    batch_id: str,
    expected_evidence_ids: set[str],
    expected_relation_ids: set[str],
) -> tuple[bool, str]:
    """Apply the integrity/completeness release gate.

    The pre-fault snapshot is the controlled-test oracle for graph completeness.
    No language generation is attempted when this function reports a fault.
    """
    reasons: list[str] = []
    graph_result = retrieve_batch(services, batch_id)
    if graph_result is None:
        return True, "batch missing from Neo4j"

    actual_ids = {x["evidence_id"] for x in graph_result["evidence"]}
    if actual_ids != expected_evidence_ids:
        missing = sorted(expected_evidence_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_evidence_ids)
        reasons.append(f"evidence-set mismatch; missing={missing}; unexpected={unexpected}")

    actual_relations = relation_ids(services, batch_id)
    if actual_relations != expected_relation_ids:
        missing = sorted(expected_relation_ids - actual_relations)
        unexpected = sorted(actual_relations - expected_relation_ids)
        reasons.append(f"path mismatch; missing={missing}; unexpected={unexpected}")

    try:
        verified = verify_evidence(services, graph_result)
        failed = sorted(x["evidence_id"] for x in verified if not x["verified"])
        if failed:
            reasons.append(f"cross-system verification failed for {failed}")
    except Exception as exc:
        reasons.append(f"evidence unavailable or unreadable: {type(exc).__name__}: {exc}")

    bad_transactions = sorted(
        x["evidence_id"]
        for x in graph_result["evidence"]
        if not transaction_metadata_ok(services, x)
    )
    if bad_transactions:
        reasons.append(f"invalid blockchain transaction metadata for {bad_transactions}")

    return bool(reasons), " | ".join(reasons) if reasons else "no fault detected"


def select_fixture(services: Services, requested_batch: str | None) -> dict:
    batch_filter = "AND b.batch_id=$batch_id" if requested_batch else ""
    query = f"""
    MATCH (b:Batch {{dataset_id:$dataset_id}})-[:SUPPORTED_BY]->(e:Evidence)
    WHERE e.ipfs_cid IS NOT NULL {batch_filter}
    RETURN b.batch_id AS batch_id, elementId(e) AS element_id,
           properties(e) AS evidence
    ORDER BY b.batch_id, e.event_order
    LIMIT 1
    """
    params = {"dataset_id": DATASET_ID}
    if requested_batch:
        params["batch_id"] = requested_batch
    fixture = neo4j_one(services, query, **params)

    other = neo4j_one(
        services,
        """
        MATCH (e:Evidence {dataset_id:$dataset_id})
        WHERE e.evidence_id <> $evidence_id
        RETURN e.ipfs_cid AS cid
        ORDER BY e.evidence_id LIMIT 1
        """,
        dataset_id=DATASET_ID,
        evidence_id=fixture["evidence"]["evidence_id"],
    )
    fixture["other_cid"] = other["cid"]
    fixture["expected_evidence_ids"] = {
        x["evidence_id"] for x in retrieve_batch(services, fixture["batch_id"])["evidence"]
    }
    fixture["expected_relation_ids"] = relation_ids(services, fixture["batch_id"])
    return fixture


def set_evidence_property(services: Services, element_id: str, name: str, value) -> None:
    allowed = {
        "ipfs_cid", "content_hash", "evidence_id", "event_type", "batch_id",
        "transaction_hash", "block_number",
    }
    if name not in allowed:
        raise ValueError(f"Unsafe property name: {name}")
    neo4j_write(
        services,
        f"MATCH (e) WHERE elementId(e)=$element_id SET e.{name}=$value",
        element_id=element_id,
        value=value,
    )


def add_altered_ipfs_object(services: Services, original_payload: bytes) -> str:
    document = json.loads(original_payload)
    document["finding"] = f"FAULT-INJECTED: {document.get('finding', '')}"
    cid, _ = services.ipfs_add(document)
    return cid


def remove_local_ipfs_block(services: Services, cid: str) -> None:
    # Remove only the selected test object.  Do not run repo/gc, which could
    # remove unrelated unpinned content.
    requests.post(
        f"{services.ipfs_api}/pin/rm", params={"arg": cid}, timeout=60
    ).raise_for_status()
    requests.post(
        f"{services.ipfs_api}/block/rm", params={"arg": cid, "force": "true"}, timeout=60
    ).raise_for_status()


def restore_ipfs_object(services: Services, payload: bytes, expected_cid: str) -> None:
    response = requests.post(
        f"{services.ipfs_api}/add",
        files={"file": ("evidence.json", payload, "application/json")},
        params={"pin": "true"},
        timeout=60,
    )
    response.raise_for_status()
    restored_cid = response.json()["Hash"]
    if restored_cid != expected_cid:
        raise RuntimeError(f"IPFS restore CID mismatch: {restored_cid} != {expected_cid}")


def remove_event_relationship(services: Services, evidence_id: str) -> dict:
    row = neo4j_one(
        services,
        """
        MATCH (s)-[r]->(o)
        WHERE r.dataset_id=$dataset_id AND r.evidence_id=$evidence_id
        RETURN elementId(s) AS source_id, elementId(o) AS target_id,
               type(r) AS rel_type, properties(r) AS rel_props
        LIMIT 1
        """,
        dataset_id=DATASET_ID,
        evidence_id=evidence_id,
    )
    if row["rel_type"] not in ALLOWED_EVENT_TYPES:
        raise RuntimeError(f"Unexpected relationship type: {row['rel_type']}")
    neo4j_write(
        services,
        """
        MATCH (s)-[r]->(o)
        WHERE elementId(s)=$source_id AND elementId(o)=$target_id
          AND r.dataset_id=$dataset_id AND r.evidence_id=$evidence_id
        DELETE r
        """,
        source_id=row["source_id"], target_id=row["target_id"],
        dataset_id=DATASET_ID, evidence_id=evidence_id,
    )
    return row


def restore_event_relationship(services: Services, row: dict) -> None:
    rel_type = row["rel_type"]
    if rel_type not in ALLOWED_EVENT_TYPES:
        raise RuntimeError(f"Unsafe relationship type: {rel_type}")
    neo4j_write(
        services,
        f"""
        MATCH (s), (o)
        WHERE elementId(s)=$source_id AND elementId(o)=$target_id
        CREATE (s)-[r:{rel_type}]->(o)
        SET r=$rel_props
        """,
        source_id=row["source_id"], target_id=row["target_id"],
        rel_props=row["rel_props"],
    )


def run_trial(
    services: Services,
    fixture: dict,
    fault_type: str,
    expected_result: str,
    inject: Callable[[], Callable[[], None]],
) -> dict:
    restore: Callable[[], None] | None = None
    detected = False
    details = ""
    status = "completed"
    try:
        restore = inject()
        detected, details = assess_state(
            services,
            fixture["batch_id"],
            fixture["expected_evidence_ids"],
            fixture["expected_relation_ids"],
        )
    except Exception as exc:
        # An exception raised by the verification path is a safe failure only
        # after injection has completed.  Setup errors remain visible as errors.
        status = "error"
        details = f"{type(exc).__name__}: {exc}"
    finally:
        if restore is not None:
            try:
                restore()
            except Exception as exc:
                raise RuntimeError(
                    f"CRITICAL: restoration failed after '{fault_type}': {exc}"
                ) from exc

    # Confirm that restoration returned the batch to a clean state.
    clean, clean_details = assess_state(
        services,
        fixture["batch_id"],
        fixture["expected_evidence_ids"],
        fixture["expected_relation_ids"],
    )
    if clean:
        raise RuntimeError(f"Restoration check failed after '{fault_type}': {clean_details}")

    generation_blocked = detected
    return {
        "fault_type": fault_type,
        "batch_id": fixture["batch_id"],
        "evidence_id": fixture["evidence"]["evidence_id"],
        "expected_result": expected_result,
        "detected": "Yes" if detected else "No",
        "generation_blocked": "Yes" if generation_blocked else "No",
        "system_response": "Integrity error; generation blocked" if detected else "Fault not detected",
        "details": details,
        "trial_status": status,
    }


def build_trials(services: Services, fixture: dict):
    evidence = fixture["evidence"]
    element_id = fixture["element_id"]
    original_cid = evidence["ipfs_cid"]
    original_payload = services.ipfs_cat(original_cid)

    def property_trial(name: str, bad_value):
        original = evidence[name]
        def inject():
            set_evidence_property(services, element_id, name, bad_value)
            return lambda: set_evidence_property(services, element_id, name, original)
        return inject

    def altered_ipfs():
        altered_cid = add_altered_ipfs_object(services, original_payload)
        set_evidence_property(services, element_id, "ipfs_cid", altered_cid)
        return lambda: set_evidence_property(services, element_id, "ipfs_cid", original_cid)

    def unavailable_ipfs():
        remove_local_ipfs_block(services, original_cid)
        return lambda: restore_ipfs_object(services, original_payload, original_cid)

    def remove_evidence_label():
        neo4j_write(
            services,
            "MATCH (e) WHERE elementId(e)=$element_id REMOVE e:Evidence",
            element_id=element_id,
        )
        return lambda: neo4j_write(
            services,
            "MATCH (e) WHERE elementId(e)=$element_id SET e:Evidence",
            element_id=element_id,
        )

    def incomplete_path():
        row = remove_event_relationship(services, evidence["evidence_id"])
        return lambda: restore_event_relationship(services, row)

    return [
        ("Altered IPFS evidence object", "CID/hash/anchor mismatch detected", altered_ipfs),
        ("Modified CID stored in Neo4j", "CID/anchor mismatch detected",
         property_trial("ipfs_cid", fixture["other_cid"])),
        ("Modified SHA-256 digest stored in Neo4j", "Digest mismatch detected",
         property_trial("content_hash", "0" * 64)),
        ("Unavailable local IPFS object", "Retrieval failure detected", unavailable_ipfs),
        ("Modified evidence identifier", "Identifier/anchor mismatch detected",
         property_trial("evidence_id", evidence["evidence_id"] + "-FAULT")),
        ("Altered blockchain-linked metadata", "Invalid transaction reference detected",
         property_trial("transaction_hash", "0x" + "0" * 64)),
        ("Evidence node removed from graph", "Evidence-set mismatch detected", remove_evidence_label),
        ("Inconsistent graph and IPFS records", "Graph/IPFS metadata mismatch detected",
         property_trial("event_type", "FAULT_EVENT")),
        ("Incomplete batch path", "Path-completeness failure detected", incomplete_path),
    ]


def write_results(rows: list[dict]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed_path = OUTPUT_DIR / "fault_injection_detailed.csv"
    with detailed_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["fault_type"]].append(row)
    summary_rows = []
    for fault_type, items in grouped.items():
        completed = [x for x in items if x["trial_status"] == "completed"]
        detected = sum(x["detected"] == "Yes" for x in completed)
        blocked = sum(x["generation_blocked"] == "Yes" for x in completed)
        injected = len(completed)
        summary_rows.append({
            "Fault type": fault_type,
            "Number injected": injected,
            "Number detected": detected,
            "Detection rate": f"{(100 * detected / injected):.2f}%" if injected else "N/A",
            "Generation blocked": f"{blocked}/{injected}" if injected else "N/A",
        })
    total_injected = sum(x["Number injected"] for x in summary_rows)
    total_detected = sum(x["Number detected"] for x in summary_rows)
    total_blocked = sum(int(x["Generation blocked"].split("/")[0]) for x in summary_rows if x["Generation blocked"] != "N/A")
    summary_rows.append({
        "Fault type": "Overall",
        "Number injected": total_injected,
        "Number detected": total_detected,
        "Detection rate": f"{(100 * total_detected / total_injected):.2f}%" if total_injected else "N/A",
        "Generation blocked": f"{total_blocked}/{total_injected}" if total_injected else "N/A",
    })
    summary_path = OUTPUT_DIR / "fault_injection_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    return detailed_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled fault-injection evaluation")
    parser.add_argument("--batch", help="Synthetic batch to test; default is the first available batch")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Number of trials per fault type (default: 1)")
    parser.add_argument("--confirm-synthetic-test", action="store_true",
                        help="Required safety confirmation")
    args = parser.parse_args()
    if not args.confirm_synthetic_test:
        parser.error("Add --confirm-synthetic-test after confirming this is the local synthetic dataset")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    services = Services()
    rows: list[dict] = []
    try:
        fixture = select_fixture(services, args.batch)
        detected, details = assess_state(
            services, fixture["batch_id"], fixture["expected_evidence_ids"],
            fixture["expected_relation_ids"],
        )
        if detected:
            raise RuntimeError(f"Baseline is not clean; fault test stopped: {details}")
        print(f"Clean baseline confirmed for {fixture['batch_id']}")

        for repetition in range(1, args.repetitions + 1):
            for fault_type, expected, inject in build_trials(services, fixture):
                print(f"Trial {repetition}: {fault_type}")
                row = run_trial(services, fixture, fault_type, expected, inject)
                row["repetition"] = repetition
                # Keep a stable column order with repetition first.
                row = {"repetition": row.pop("repetition"), **row}
                rows.append(row)
                print(f"  detected={row['detected']}; blocked={row['generation_blocked']}")
                time.sleep(0.1)
        detailed, summary = write_results(rows)
        print(f"\nDetailed results: {detailed}")
        print(f"Table summary:    {summary}")
    finally:
        services.close()


if __name__ == "__main__":
    main()

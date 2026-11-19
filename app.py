"""Expanded blockchain-anchored food traceability application.

Loads the 50-batch synthetic dataset from data_expanded.  Ground truth is
deliberately excluded from the operational graph and is used only by the
evaluation program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from web3 import Web3


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

RELATIONSHIP_ORDER = [
    "PRODUCED", "PROCESSED_AT", "STORED_AT", "TRANSFERRED_TO",
    "DISPATCHED_AS", "DELIVERED_TO", "HAS_INSPECTION",
]
ALLOWED_EVENT_TYPES = set(RELATIONSHIP_ORDER)


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def event_sort_key(item: dict) -> tuple:
    document = item.get("document") or json.loads(item["canonical_json"])
    return (int(document.get("event_order", 9999)), document["evidence_id"])


class Services:
    def __init__(self):
        self.neo4j = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
        )
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.ipfs_api = os.getenv("IPFS_API", "http://127.0.0.1:5001/api/v0")
        self.ollama_api = os.getenv("OLLAMA_API", "http://127.0.0.1:11434/api/chat")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

        provider = os.getenv("WEB3_PROVIDER", "http://127.0.0.1:7545")
        self.w3 = Web3(Web3.HTTPProvider(provider))
        if not self.w3.is_connected():
            raise RuntimeError(f"Cannot connect to Ganache at {provider}")
        artifact_path = ROOT / os.getenv("CONTRACT_ARTIFACT", "build/EvidenceAnchor.json")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        address = os.getenv("CONTRACT_ADDRESS") or artifact["address"]
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(address), abi=artifact["abi"]
        )
        self.account = self.w3.eth.accounts[int(os.getenv("BLOCKCHAIN_ACCOUNT_INDEX", "0"))]

    def close(self) -> None:
        self.neo4j.close()

    def ipfs_add(self, document: dict) -> tuple[str, bytes]:
        payload = canonical_bytes(document)
        response = requests.post(
            f"{self.ipfs_api}/add",
            files={"file": ("evidence.json", payload, "application/json")},
            params={"pin": "true"}, timeout=60,
        )
        response.raise_for_status()
        return response.json()["Hash"], payload

    def ipfs_cat(self, cid: str) -> bytes:
        response = requests.post(f"{self.ipfs_api}/cat", params={"arg": cid}, timeout=60)
        response.raise_for_status()
        return response.content

    def anchor(self, document: dict, cid: str, payload: bytes) -> tuple[str, int]:
        digest = bytes.fromhex(sha256_hex(payload))
        tx_hash = self.contract.functions.anchorEvidence(
            document["batch_id"], document["evidence_id"],
            document["event_type"], cid, digest,
        ).transact({"from": self.account})
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_hash.hex(), receipt.blockNumber

    def verify_anchor(self, document: dict, cid: str, payload: bytes) -> bool:
        digest = bytes.fromhex(sha256_hex(payload))
        return self.contract.functions.verifyEvidence(
            document["batch_id"], document["evidence_id"],
            document["event_type"], cid, digest,
        ).call()


def make_document(row: dict) -> dict:
    if row["event_type"] not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"Unsupported event_type: {row['event_type']}")
    temperature = row["temperature_c"].strip()
    return {
        "schema_version": "2.0",
        "synthetic": True,
        "evidence_id": row["evidence_id"],
        "batch_id": row["batch_id"],
        "event_order": int(row["event_order"]),
        "event_type": row["event_type"],
        "subject_id": row["subject_id"],
        "object_id": row["object_id"],
        "timestamp_utc": row["timestamp_utc"],
        "location": row["location"],
        "temperature_c": float(temperature) if temperature else None,
        "inspection_result": row["inspection_result"] or None,
        "finding": row["finding"],
        "record_state": row["record_state"],
    }


def clear_expanded_graph(tx) -> None:
    tx.run("MATCH (n {dataset_id:'expanded-50-v1'}) DETACH DELETE n")


def upsert_batch(tx, batch: dict) -> None:
    tx.run(
        """
        MERGE (b:Entity:Batch {entity_id:$batch_id})
        SET b.batch_id=$batch_id, b.category=$category, b.product=$product,
            b.scenario_type=$scenario_type, b.path_structure=$path_structure,
            b.status=$expected_status, b.producer_country=$producer_country,
            b.synthetic=true, b.dataset_id='expanded-50-v1'
        """, **batch,
    )


def upsert_event(tx, document: dict, anchored: dict) -> None:
    event_type = document["event_type"]
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"Unsafe relationship type: {event_type}")
    params = {**document, **anchored}
    tx.run(
        f"""
        MERGE (s:Entity {{entity_id:$subject_id}})
          ON CREATE SET s.dataset_id='expanded-50-v1', s.synthetic=true
        MERGE (o:Entity {{entity_id:$object_id}})
          ON CREATE SET o.dataset_id='expanded-50-v1', o.synthetic=true
        MERGE (s)-[r:{event_type} {{evidence_id:$evidence_id}}]->(o)
        SET r.event_order=$event_order, r.timestamp_utc=$timestamp_utc,
            r.location=$location, r.record_state=$record_state,
            r.dataset_id='expanded-50-v1'
        MERGE (e:Evidence {{evidence_id:$evidence_id}})
        SET e.batch_id=$batch_id, e.event_order=$event_order,
            e.event_type=$event_type, e.subject_id=$subject_id,
            e.object_id=$object_id, e.record_state=$record_state,
            e.ipfs_cid=$ipfs_cid, e.content_hash=$content_hash,
            e.transaction_hash=$transaction_hash, e.block_number=$block_number,
            e.canonical_json=$canonical_json, e.integrity_status='ANCHORED',
            e.synthetic=true, e.dataset_id='expanded-50-v1'
        WITH e
        MATCH (b:Batch {{batch_id:$batch_id}})
        MERGE (b)-[:SUPPORTED_BY]->(e)
        """, **params,
    )


def seed_expanded(services: Services, data_dir: Path, keep_existing: bool = False) -> None:
    batches = read_csv(data_dir / "batches.csv")
    events = read_csv(data_dir / "events.csv")
    if len(batches) != 50 or len(events) != 310:
        raise RuntimeError(f"Expected 50 batches and 310 events; got {len(batches)} and {len(events)}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in events:
        grouped[row["batch_id"]].append(row)

    with services.neo4j.session(database=services.database) as session:
        if not keep_existing:
            session.execute_write(clear_expanded_graph)
        for batch in batches:
            session.execute_write(upsert_batch, batch)
            anchored_count = 0
            for row in sorted(grouped[batch["batch_id"]], key=lambda x: int(x["event_order"])):
                document = make_document(row)
                cid, payload = services.ipfs_add(document)
                tx_hash, block_number = services.anchor(document, cid, payload)
                anchored = {
                    "ipfs_cid": cid,
                    "content_hash": sha256_hex(payload),
                    "transaction_hash": tx_hash,
                    "block_number": block_number,
                    "canonical_json": payload.decode("utf-8"),
                }
                session.execute_write(upsert_event, document, anchored)
                anchored_count += 1
            print(f"Anchored and indexed {batch['batch_id']} ({anchored_count} present events)")


def retrieve_batch(services: Services, batch_id: str) -> dict | None:
    query = """
    MATCH (batch:Batch {batch_id:$batch_id})
    OPTIONAL MATCH (batch)-[:SUPPORTED_BY]->(evidence:Evidence)
    WITH batch, evidence ORDER BY evidence.event_order, evidence.evidence_id
    RETURN properties(batch) AS batch,
           [e IN collect(evidence) WHERE e IS NOT NULL | properties(e)] AS evidence
    """
    with services.neo4j.session(database=services.database) as session:
        record = session.run(query, batch_id=batch_id).single()
    return None if record is None else {"batch": record["batch"], "evidence": record["evidence"]}


def verify_evidence(services: Services, graph_result: dict) -> list[dict]:
    verified = []
    for item in graph_result["evidence"]:
        stored_payload = item["canonical_json"].encode("utf-8")
        ipfs_payload = services.ipfs_cat(item["ipfs_cid"])
        document = json.loads(ipfs_payload)
        checks = {
            "neo4j_hash_ok": sha256_hex(stored_payload) == item["content_hash"],
            "ipfs_hash_ok": sha256_hex(ipfs_payload) == item["content_hash"],
            "blockchain_ok": services.verify_anchor(document, item["ipfs_cid"], ipfs_payload),
            "graph_metadata_ok": (
                document["evidence_id"] == item["evidence_id"]
                and document["event_type"] == item["event_type"]
                and document["batch_id"] == item["batch_id"]
            ),
        }
        verified.append({**item, **checks, "verified": all(checks.values()), "document": document})
    return sorted(verified, key=event_sort_key)


def evidence_lines(items: list[dict]) -> list[str]:
    lines = []
    for item in items:
        doc = item["document"]
        details = {k: doc.get(k) for k in (
            "timestamp_utc", "location", "temperature_c", "inspection_result",
            "finding", "record_state") if doc.get(k) not in (None, "")}
        lines.append(
            f"[{doc['evidence_id']}] order={doc['event_order']} {doc['event_type']}: "
            f"{doc['subject_id']} -> {doc['object_id']}; "
            f"details={json.dumps(details, ensure_ascii=False, sort_keys=True)}"
        )
    return lines


def deterministic_answer(graph_result: dict, verified: list[dict]) -> str:
    batch = graph_result["batch"]
    statements = [
        f"Batch {batch['batch_id']} is synthetic {batch['product']} in the "
        f"{batch['category']} category with recorded status {batch['status']}."
    ]
    for item in verified:
        doc = item["document"]
        detail = doc.get("finding") or doc.get("inspection_result") or doc.get("location") or "recorded event"
        statements.append(
            f"Event {doc['event_order']} records {doc['event_type']} from "
            f"{doc['subject_id']} to {doc['object_id']}: {detail} "
            f"[{doc['evidence_id']}]."
        )
    statements.append("All cited available evidence passed IPFS and blockchain integrity verification.")
    return " ".join(statements)


def call_ollama(services: Services, question: str, verified: list[dict]) -> str:
    allowed = [item["evidence_id"] for item in verified]
    prompt = f"""You are a food-traceability reporting assistant.
Use only the verified evidence. Preserve event_order, including branches and repeated inspections.
Do not invent missing events, entities, causes, or identifiers.
Explicitly report CONFLICTING or INCONSISTENT_ID record states.
Place a supporting evidence ID after every event-level factual sentence.
Allowed evidence IDs: {allowed}

Question: {question}

Verified evidence:
{chr(10).join(evidence_lines(verified))}
"""
    response = requests.post(
        services.ollama_api,
        json={"model": services.ollama_model, "stream": False,
              "messages": [{"role": "system", "content": "Return grounded facts only."},
                           {"role": "user", "content": prompt}],
              "options": {"temperature": 0.0, "seed": 42, "num_predict": 500}},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def validate_answer(answer: str, verified: list[dict]) -> dict:
    expected = {item["evidence_id"] for item in verified}
    cited = set(re.findall(r"\[(EV-[A-Z]+-[0-9]+)\]", answer))
    return {"expected": sorted(expected), "cited": sorted(cited),
            "missing": sorted(expected - cited), "unknown": sorted(cited - expected),
            "passed": cited == expected}


def ask(services: Services, batch_id: str, question: str) -> None:
    total_start = time.perf_counter()
    start = time.perf_counter(); graph_result = retrieve_batch(services, batch_id)
    retrieval_ms = (time.perf_counter() - start) * 1000
    if graph_result is None:
        print(f"Safe fallback: batch {batch_id} was not found in Neo4j."); return
    start = time.perf_counter(); verified = verify_evidence(services, graph_result)
    verification_ms = (time.perf_counter() - start) * 1000
    failures = [x["evidence_id"] for x in verified if not x["verified"]]
    if failures:
        print("INTEGRITY WARNING: generation blocked.")
        print("Failed evidence:", ", ".join(failures)); return
    start = time.perf_counter(); candidate = call_ollama(services, question, verified)
    generation_ms = (time.perf_counter() - start) * 1000
    validation = validate_answer(candidate, verified)
    released = candidate if validation["passed"] else deterministic_answer(graph_result, verified)
    release_mode = "LLM_VALIDATED" if validation["passed"] else "DETERMINISTIC_SAFE_FALLBACK"
    total_ms = (time.perf_counter() - total_start) * 1000
    report = {"batch_id": batch_id, "question": question, "candidate_answer": candidate,
              "candidate_validation": validation, "released_answer": released,
              "release_mode": release_mode,
              "verified_evidence_ids": [x["evidence_id"] for x in verified],
              "retrieval_ms": round(retrieval_ms, 2),
              "blockchain_verification_ms": round(verification_ms, 2),
              "generation_ms": round(generation_ms, 2), "total_ms": round(total_ms, 2),
              "created_at": now_iso()}
    output_dir = ROOT / "outputs"; output_dir.mkdir(exist_ok=True)
    path = output_dir / f"trace_{batch_id}_{int(time.time())}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nRELEASED ANSWER\n" + "-" * 70 + "\n" + released)
    print("\nRelease mode:", release_mode)
    print(f"Retrieval: {retrieval_ms:.2f} ms")
    print(f"Blockchain/IPFS verification: {verification_ms:.2f} ms")
    print(f"Generation: {generation_ms:.2f} ms")
    print(f"Total: {total_ms:.2f} ms\nExperiment log: {path}")


def list_batches(services: Services) -> None:
    query = """MATCH (b:Batch {dataset_id:'expanded-50-v1'})
               RETURN b.batch_id AS batch_id, b.category AS category,
                      b.product AS product, b.status AS status
               ORDER BY b.category, b.batch_id"""
    with services.neo4j.session(database=services.database) as session:
        rows = list(session.run(query))
    for row in rows:
        print(f"{row['batch_id']:<24} {row['category']:<18} {row['product']:<24} {row['status']}")
    print(f"Total batches: {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanded blockchain-anchored GraphRAG")
    sub = parser.add_subparsers(dest="command", required=True)
    seed_cmd = sub.add_parser("seed-expanded")
    seed_cmd.add_argument("--data-dir", default="data_expanded")
    seed_cmd.add_argument("--keep-existing", action="store_true")
    sub.add_parser("list")
    ask_cmd = sub.add_parser("ask")
    ask_cmd.add_argument("--batch", required=True); ask_cmd.add_argument("--question", required=True)
    args = parser.parse_args()
    services = Services()
    try:
        if args.command == "seed-expanded":
            seed_expanded(services, ROOT / args.data_dir, args.keep_existing)
        elif args.command == "list": list_batches(services)
        elif args.command == "ask": ask(services, args.batch, args.question)
    finally:
        services.close()


if __name__ == "__main__":
    main()

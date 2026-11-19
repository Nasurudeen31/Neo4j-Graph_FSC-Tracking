"""Claim-level factual support evaluation for the expanded baseline experiment.

This program reuses saved answers; it does not rerun retrieval, blockchain,
IPFS, or answer generation.  It selects repetition 1 for every batch and
configuration (50 x 5 = 250 answers), asks the local Ollama model to decompose
each answer into atomic claims, and judges each claim against the structured
batch/evidence records.  Checkpoints make the run resumable.

Outputs:
  outputs/claim_level_evaluation/claim_level_answers.csv
  outputs/claim_level_evaluation/claim_level_claims.csv
  outputs/claim_level_evaluation/claim_level_summary.csv
  outputs/claim_level_evaluation/claim_level_checkpoint.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "claim_level_evaluation"
LABELS = {"supported", "unsupported", "contradicted", "incomplete", "misattributed"}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_text": {"type": "string"},
                    "cited_evidence_ids": {
                        "type": "array", "items": {"type": "string"}
                    },
                    "label": {"type": "string", "enum": sorted(LABELS)},
                    "supporting_evidence_ids": {
                        "type": "array", "items": {"type": "string"}
                    },
                    "required_units_covered": {
                        "type": "array", "items": {"type": "string"}
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "claim_text", "cited_evidence_ids", "label",
                    "supporting_evidence_ids", "required_units_covered", "rationale",
                ],
            },
        },
        "answer_notes": {"type": "string"},
    },
    "required": ["claims", "answer_notes"],
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_id(value: str) -> str:
    return value.strip().upper()


def split_ids(value: str) -> set[str]:
    return {normalize_id(x) for x in (value or "").split(";") if x.strip()}


def as_bool(value: str) -> bool | None:
    value = str(value or "").strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    return None


def pct(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if not denominator else round(100.0 * numerator / denominator, 2)


def select_effectiveness_rows(raw_rows: list[dict]) -> list[dict]:
    first = [row for row in raw_rows if str(row.get("repetition", "1")).strip() == "1"]
    unique: dict[tuple[str, str], dict] = {}
    for row in first:
        key = (row["configuration_id"], row["batch_id"])
        unique.setdefault(key, row)
    rows = sorted(unique.values(), key=lambda x: (x["configuration_id"], x["batch_id"]))
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["configuration_id"]] += 1
    if len(rows) != 250 or len(counts) != 5 or any(v != 50 for v in counts.values()):
        raise RuntimeError(
            f"Expected 250 unique first-run answers (50 per configuration); got "
            f"{len(rows)} with counts {dict(counts)}"
        )
    return rows


def load_dataset(data_dir: Path) -> tuple[dict[str, dict], dict[str, list[dict]], dict[str, dict]]:
    batches = {row["batch_id"]: row for row in read_csv(data_dir / "batches.csv")}
    events: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(data_dir / "events.csv"):
        events[row["batch_id"]].append(row)
    for batch_id in events:
        events[batch_id].sort(key=lambda x: (int(x["event_order"]), x["evidence_id"]))
    truth = {row["batch_id"]: row for row in read_csv(data_dir / "ground_truth.csv")}
    if len(batches) != 50 or len(events) != 50 or len(truth) != 50:
        raise RuntimeError("Expanded dataset must contain 50 batches in all three CSV sources")
    return batches, events, truth


def factual_package(
    row: dict,
    batch: dict,
    events: list[dict],
    truth: dict,
) -> dict:
    event_map = {e["evidence_id"]: e for e in events}
    expected = split_ids(row.get("expected_ids", ""))
    if not expected:
        expected = split_ids(truth.get("expected_evidence_ids", ""))
    # Missing ground-truth records do not have evidence objects and therefore
    # are not counted as present event-claim units.  Their absence can still be
    # reported as a batch-level anomaly/status claim.
    present_expected = sorted(x for x in expected if x in event_map)
    required_units = ["BATCH_STATUS", *present_expected]
    anomaly = str(truth.get("anomaly_type", "")).strip().lower()
    if anomaly and anomaly not in {"normal", "none", "no_anomaly"}:
        required_units.append("ANOMALY")
    return {
        "batch": batch,
        "evidence": events,
        "ground_truth": truth,
        "run_context": {
            "configuration_id": row["configuration_id"],
            "configuration": row["configuration"],
            "integrity_passed": as_bool(row.get("integrity_passed", "")),
            "validation_passed": as_bool(row.get("validation_passed", "")),
            "fallback_used": as_bool(row.get("fallback_used", "")),
            "retrieved_ids": sorted(split_ids(row.get("retrieved_ids", ""))),
        },
        "required_units": required_units,
    }


def judge_prompt(answer: str, package: dict) -> str:
    return f"""You are performing a strict claim-level factual audit for a synthetic
food-traceability experiment. Use ONLY the supplied reference package.

TASK
1. Split the answer into atomic factual claims. One claim must express one
   independently verifiable proposition. Split sentences containing multiple
   events, entities, statuses, measurements, locations, or findings. Keep the
   audit compact: use one central claim for each reported event plus separate
   claims for batch status and anomaly conclusions. Return no more than 15
   claims for an answer.
2. Ignore headings, formatting, field-name lists, instructions, questions,
   refusals, recommendations, and generic procedural guidance when they make
   no factual assertion about this batch.
3. Extract the evidence ID(s) visibly cited with each claim.
4. Assign exactly one label:
   - supported: the claim is fully entailed by the cited evidence or supplied
     batch/run metadata, with no material omission.
   - unsupported: the claim is not established by the supplied references.
   - contradicted: supplied evidence directly conflicts with the claim.
   - incomplete: the claim is partly supported but omits a material qualifier
     that changes its interpretation.
   - misattributed: the claim may be supported by the package, but its cited
     evidence ID does not support that claim.
5. supporting_evidence_ids must contain only evidence IDs that actually support
   the claim. Use an empty list for a batch-metadata-only claim.
6. required_units_covered may contain BATCH_STATUS and/or evidence IDs from the
   supplied required_units. Add a unit only when the claim correctly reports
   its central fact. BATCH_STATUS represents the reported batch status and
   ANOMALY represents the correct reporting of the ground-truth anomaly or
   missing/conflicting condition. Do not add a unit for unsupported, contradicted,
   incomplete, or misattributed claims.
7. A valid evidence ID appearing somewhere in the answer is not sufficient.
   It must be attached to the claim that it supports.
8. Do not invent claims that are absent from the answer.
9. Keep each rationale under 12 words. Do not repeat evidence content in the
   rationale or answer_notes.

ANSWER TO AUDIT
{answer}

REFERENCE PACKAGE
{json.dumps(package, ensure_ascii=False, sort_keys=True)}

Return only JSON conforming to the requested schema."""


def call_judge(
    ollama_url: str,
    model: str,
    prompt: str,
    timeout: int,
    retries: int = 2,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                ollama_url,
                json={
                    "model": model,
                    "stream": False,
                    "format": JUDGE_SCHEMA,
                    "messages": [
                        {"role": "system", "content": "Audit claims conservatively. Output JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {
                        "temperature": 0.0,
                        "seed": 42,
                        "num_predict": 700,
                        "num_ctx": 8192,
                    },
                },
                timeout=timeout,
            )
            response.raise_for_status()
            result = json.loads(response.json()["message"]["content"])
            validate_judgment(result)
            return result
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Ollama claim judge failed after {retries} attempts: {last_error}")


def validate_judgment(result: dict) -> None:
    if not isinstance(result, dict) or not isinstance(result.get("claims"), list):
        raise ValueError("Judge result does not contain a claims array")
    for claim in result["claims"]:
        if claim.get("label") not in LABELS:
            raise ValueError(f"Invalid claim label: {claim.get('label')}")
        for key in ("claim_text", "rationale"):
            if not isinstance(claim.get(key), str):
                raise ValueError(f"Claim field {key} must be text")
        for key in ("cited_evidence_ids", "supporting_evidence_ids", "required_units_covered"):
            if not isinstance(claim.get(key), list):
                raise ValueError(f"Claim field {key} must be a list")


def sanitize_judgment(result: dict, package: dict) -> dict:
    valid_ids = {e["evidence_id"] for e in package["evidence"]}
    valid_units = set(package["required_units"])
    for claim in result["claims"]:
        claim["cited_evidence_ids"] = sorted({
            normalize_id(x) for x in claim["cited_evidence_ids"]
            if normalize_id(x) in valid_ids
        })
        claim["supporting_evidence_ids"] = sorted({
            normalize_id(x) for x in claim["supporting_evidence_ids"]
            if normalize_id(x) in valid_ids
        })
        if claim["label"] != "supported":
            claim["required_units_covered"] = []
        else:
            claim["required_units_covered"] = sorted({
                normalize_id(x)
                for x in claim["required_units_covered"]
                if normalize_id(x) in valid_units
            })
    return result


def _norm(value) -> str:
    value = str(value or "").lower().replace("_", " ")
    return re.sub(r"[^a-z0-9.]+", " ", value).strip()


def _citation_context(answer: str, evidence_id: str, radius: int = 350) -> str:
    match = re.search(rf"\[{re.escape(evidence_id)}\]", answer, flags=re.I)
    if not match:
        return ""
    return answer[max(0, match.start() - radius):min(len(answer), match.end() + radius)]


def rule_based_judgment(answer: str, package: dict) -> dict:
    """Fast, reproducible claim audit using structured evidence fields.

    The unit of analysis is one central event claim per cited evidence record,
    plus separate batch-status, anomaly, and integrity assertions.  This mode
    avoids using an LLM as its own factual judge.
    """
    claims: list[dict] = []
    event_map = {e["evidence_id"]: e for e in package["evidence"]}
    cited_in_answer = [normalize_id(x) for x in re.findall(r"\[(EV-[A-Z]+-[0-9]+)\]", answer)]
    seen: set[str] = set()

    def add(text, cited, label, supporting, units, rationale):
        claims.append({
            "claim_text": text.strip(),
            "cited_evidence_ids": cited,
            "label": label,
            "supporting_evidence_ids": supporting,
            "required_units_covered": units if label == "supported" else [],
            "rationale": rationale,
        })

    # Evaluate every distinct claim-citation association locally.  A citation
    # is supported only when its nearby statement contains central fields from
    # that same evidence object.
    for evidence_id in cited_in_answer:
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        context = _citation_context(answer, evidence_id)
        context_n = _norm(context)
        event = event_map.get(evidence_id)
        if event is None:
            add(context, [evidence_id], "unsupported", [], [], "Citation is absent from the batch evidence.")
            continue

        central = {
            "event_type": _norm(event.get("event_type")),
            "subject_id": _norm(event.get("subject_id")),
            "object_id": _norm(event.get("object_id")),
            "finding": _norm(event.get("finding")),
        }
        secondary = {
            "location": _norm(event.get("location")),
            "temperature": _norm(event.get("temperature_c")),
            "inspection": _norm(event.get("inspection_result")),
            "record_state": _norm(event.get("record_state")),
        }
        central_hits = {k for k, v in central.items() if v and v in context_n}
        secondary_hits = {k for k, v in secondary.items() if v and v in context_n}

        other_best = 0
        for other_id, other in event_map.items():
            if other_id == evidence_id:
                continue
            values = [_norm(other.get(k)) for k in ("event_type", "subject_id", "object_id", "finding")]
            other_best = max(other_best, sum(bool(v and v in context_n) for v in values))

        own_score = len(central_hits)
        if other_best > own_score and other_best >= 2:
            add(context, [evidence_id], "misattributed", [], [], "Nearby statement matches a different evidence record.")
        elif "finding" in central_hits or {"subject_id", "object_id"} <= central_hits or own_score >= 2:
            add(
                context, [evidence_id], "supported", [evidence_id],
                [evidence_id] if evidence_id in package["required_units"] else [],
                "Central event fields match the cited evidence.",
            )
        elif own_score + len(secondary_hits) >= 1:
            add(context, [evidence_id], "incomplete", [evidence_id], [], "Only part of the cited event is established.")
        else:
            add(context, [evidence_id], "unsupported", [], [], "Nearby statement lacks matching evidence fields.")

    answer_n = _norm(answer)
    batch = package["batch"]
    expected_status = _norm(batch.get("expected_status") or batch.get("status"))
    status_values = {"released", "hold", "recalled", "review required"}
    mentioned_statuses = {x for x in status_values if x in answer_n}
    if expected_status and expected_status in mentioned_statuses:
        add(
            f"Batch status is {expected_status}.", [], "supported", [], ["BATCH_STATUS"],
            "Status matches the batch ground truth.",
        )
    elif mentioned_statuses:
        add(
            f"Reported batch status: {sorted(mentioned_statuses)}", [], "contradicted", [], [],
            "Reported status conflicts with the batch ground truth.",
        )

    # Report correct product/category metadata as a supported batch claim, but
    # it is not an additional required unit.
    product = _norm(batch.get("product"))
    category = _norm(batch.get("category"))
    if product and product in answer_n:
        add(
            f"Batch product is {batch.get('product')}.", [], "supported", [], [],
            "Product matches batch metadata.",
        )
    if category and category in answer_n:
        add(
            f"Batch category is {batch.get('category')}.", [], "supported", [], [],
            "Category matches batch metadata.",
        )

    if "ANOMALY" in package["required_units"]:
        anomaly = _norm(package["ground_truth"].get("anomaly_type"))
        anomaly_terms = {
            "cold chain excursion": ("cold chain", "excursion", "temperature"),
            "missing record": ("missing", "absent", "incomplete"),
            "conflicting inspection": ("conflict", "conflicting"),
            "inconsistent identifier": ("inconsistent", "identifier"),
        }
        terms = next((v for k, v in anomaly_terms.items() if k in anomaly), tuple(anomaly.split()))
        if any(term and term in answer_n for term in terms):
            add(
                f"Reported anomaly: {package['ground_truth'].get('anomaly_type')}", [],
                "supported", [], ["ANOMALY"], "Anomaly matches the ground truth.",
            )

    integrity_asserted = bool(
        re.search(r"(?:passed|successful|verified).{0,80}(?:ipfs|blockchain|integrity)|"
                  r"(?:ipfs|blockchain|integrity).{0,80}(?:passed|successful|verified)",
                  answer, flags=re.I | re.S)
    )
    if integrity_asserted:
        integrity = package["run_context"].get("integrity_passed")
        label = "supported" if integrity is True else "unsupported"
        add(
            "Available evidence passed integrity verification.", [], label, [], [],
            "Run context confirms verification." if label == "supported"
            else "This configuration did not establish integrity verification.",
        )

    return {"claims": claims, "answer_notes": "Deterministic structured-evidence audit."}


def load_checkpoint(path: Path) -> dict[str, dict]:
    saved: dict[str, dict] = {}
    if not path.exists():
        return saved
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                saved[item["answer_id"]] = item
    return saved


def append_checkpoint(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def answer_metrics(item: dict) -> tuple[dict, list[dict]]:
    row = item["source"]
    result = item["judgment"]
    required = set(item["required_units"])
    claims = result["claims"]
    labels = defaultdict(int)
    covered: set[str] = set()
    citation_pairs = 0
    correct_citation_pairs = 0
    citable_supported_claims = 0
    cited_supported_claims = 0
    claim_rows: list[dict] = []

    for index, claim in enumerate(claims, 1):
        label = claim["label"]
        labels[label] += 1
        cited = set(claim["cited_evidence_ids"])
        supporting = set(claim["supporting_evidence_ids"])
        covered.update(claim["required_units_covered"])
        citation_pairs += len(cited)
        correct = cited & supporting if label == "supported" else set()
        correct_citation_pairs += len(correct)
        if label == "supported" and supporting:
            citable_supported_claims += 1
            if correct:
                cited_supported_claims += 1
        claim_rows.append({
            "answer_id": item["answer_id"],
            "configuration_id": row["configuration_id"],
            "configuration": row["configuration"],
            "batch_id": row["batch_id"],
            "claim_number": index,
            "claim_text": claim["claim_text"],
            "cited_evidence_ids": ";".join(sorted(cited)),
            "supporting_evidence_ids": ";".join(sorted(supporting)),
            "required_units_covered": ";".join(claim["required_units_covered"]),
            "label": label,
            "rationale": claim["rationale"],
        })

    total = len(claims)
    supported = labels["supported"]
    fully_covered = covered >= required
    answer_valid = bool(
        total > 0
        and supported == total
        and fully_covered
        and correct_citation_pairs == citation_pairs
    )
    metrics = {
        "answer_id": item["answer_id"],
        "configuration_id": row["configuration_id"],
        "configuration": row["configuration"],
        "batch_id": row["batch_id"],
        "answer_field": item["answer_field"],
        "total_claims": total,
        "supported_claims": supported,
        "unsupported_claims": labels["unsupported"],
        "contradicted_claims": labels["contradicted"],
        "incomplete_claims": labels["incomplete"],
        "misattributed_claims": labels["misattributed"],
        "required_units": len(required),
        "required_units_covered": len(covered & required),
        "citation_pairs": citation_pairs,
        "correct_citation_pairs": correct_citation_pairs,
        "citable_supported_claims": citable_supported_claims,
        "cited_supported_claims": cited_supported_claims,
        "claim_precision_percent": pct(supported, total),
        "claim_recall_percent": pct(len(covered & required), len(required)),
        "citation_precision_percent": pct(correct_citation_pairs, citation_pairs),
        "citation_recall_percent": pct(cited_supported_claims, citable_supported_claims),
        "unsupported_claim_rate_percent": pct(labels["unsupported"], total),
        "contradiction_rate_percent": pct(labels["contradicted"], total),
        "incomplete_claim_rate_percent": pct(labels["incomplete"], total),
        "misattribution_rate_percent": pct(labels["misattributed"], total),
        "answer_valid": 1 if answer_valid else 0,
        "answer_notes": result.get("answer_notes", ""),
    }
    return metrics, claim_rows


def aggregate(answer_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in answer_rows:
        grouped[(row["configuration_id"], row["configuration"])].append(row)
    summaries = []
    for (config_id, config), rows in sorted(grouped.items()):
        sums = {key: sum(int(r[key]) for r in rows) for key in (
            "total_claims", "supported_claims", "unsupported_claims",
            "contradicted_claims", "incomplete_claims", "misattributed_claims",
            "required_units", "required_units_covered", "citation_pairs",
            "correct_citation_pairs", "citable_supported_claims",
            "cited_supported_claims", "answer_valid",
        )}
        summaries.append({
            "Configuration": config,
            "Answers": len(rows),
            "Claims": sums["total_claims"],
            "Claim precision (%)": pct(sums["supported_claims"], sums["total_claims"]),
            "Claim recall (%)": pct(sums["required_units_covered"], sums["required_units"]),
            "Citation precision (%)": pct(sums["correct_citation_pairs"], sums["citation_pairs"]),
            "Citation recall (%)": pct(sums["cited_supported_claims"], sums["citable_supported_claims"]),
            "Unsupported (%)": pct(sums["unsupported_claims"], sums["total_claims"]),
            "Contradicted (%)": pct(sums["contradicted_claims"], sums["total_claims"]),
            "Incomplete (%)": pct(sums["incomplete_claims"], sums["total_claims"]),
            "Misattributed (%)": pct(sums["misattributed_claims"], sums["total_claims"]),
            "Answer validation (%)": pct(sums["answer_valid"], len(rows)),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim-level factual evaluation")
    parser.add_argument("--raw-results", default="outputs/expanded_baseline_evaluation/baseline_raw_results.csv")
    parser.add_argument("--data-dir", default="data_expanded")
    parser.add_argument("--answer-field", choices=["candidate_answer", "released_answer"],
                        default="released_answer")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--limit", type=int, help="Test only the first N answers")
    parser.add_argument(
        "--stratified-per-category", type=int,
        help="Evaluate N batches per food category for every configuration "
             "(for example, 2 gives 50 answers total)",
    )
    parser.add_argument(
        "--rules-only", action="store_true",
        help="Use the fast deterministic structured-evidence evaluator; no Ollama calls",
    )
    args = parser.parse_args()

    raw_path = ROOT / args.raw_results
    data_dir = ROOT / args.data_dir
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw results not found: {raw_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    effectiveness = select_effectiveness_rows(read_csv(raw_path))
    if args.stratified_per_category:
        if args.stratified_per_category < 1:
            parser.error("--stratified-per-category must be at least 1")
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in effectiveness:
            grouped[(row["configuration_id"], row["category"])].append(row)
        effectiveness = []
        for key in sorted(grouped):
            effectiveness.extend(grouped[key][:args.stratified_per_category])
        print(
            f"Using stratified claim sample: {len(effectiveness)} answers "
            f"({args.stratified_per_category} per category/configuration)"
        )
    if args.limit:
        effectiveness = effectiveness[:args.limit]
    batches, events, truth = load_dataset(data_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mode_name = "rules" if args.rules_only else "ollama"
    checkpoint_path = OUTPUT_DIR / f"claim_level_{args.answer_field}_{mode_name}_checkpoint.jsonl"
    checkpoint = load_checkpoint(checkpoint_path)

    for index, row in enumerate(effectiveness, 1):
        answer_id = f"{args.answer_field}|{row['configuration_id']}|{row['batch_id']}"
        if answer_id in checkpoint:
            print(f"[{index}/{len(effectiveness)}] Resume: {answer_id}")
            continue
        package = factual_package(row, batches[row["batch_id"]], events[row["batch_id"]], truth[row["batch_id"]])
        answer = row.get(args.answer_field, "").strip()
        print(f"[{index}/{len(effectiveness)}] Judging {answer_id}")
        if args.rules_only:
            judgment = rule_based_judgment(answer, package)
        else:
            judgment = call_judge(args.ollama_url, args.model, judge_prompt(answer, package), args.timeout)
        judgment = sanitize_judgment(judgment, package)
        item = {
            "answer_id": answer_id,
            "answer_field": args.answer_field,
            "required_units": package["required_units"],
            "source": {
                "configuration_id": row["configuration_id"],
                "configuration": row["configuration"],
                "batch_id": row["batch_id"],
            },
            "judgment": judgment,
        }
        append_checkpoint(checkpoint_path, item)
        checkpoint[answer_id] = item

    expected_ids = {
        f"{args.answer_field}|{r['configuration_id']}|{r['batch_id']}" for r in effectiveness
    }
    items = [checkpoint[x] for x in sorted(expected_ids) if x in checkpoint]
    if len(items) != len(effectiveness):
        raise RuntimeError(f"Only {len(items)} of {len(effectiveness)} answers were completed")

    answer_rows: list[dict] = []
    claim_rows: list[dict] = []
    for item in items:
        answer_metrics_row, claims = answer_metrics(item)
        answer_rows.append(answer_metrics_row)
        claim_rows.extend(claims)
    summary_rows = aggregate(answer_rows)

    answer_path = OUTPUT_DIR / f"claim_level_{args.answer_field}_{mode_name}_answers.csv"
    claim_path = OUTPUT_DIR / f"claim_level_{args.answer_field}_{mode_name}_claims.csv"
    summary_path = OUTPUT_DIR / f"claim_level_{args.answer_field}_{mode_name}_summary.csv"
    write_csv(answer_path, answer_rows, list(answer_rows[0]))
    write_csv(claim_path, claim_rows, list(claim_rows[0]))
    write_csv(summary_path, summary_rows, list(summary_rows[0]))
    print(f"\nAnswer-level audit: {answer_path}")
    print(f"Claim-level audit:  {claim_path}")
    print(f"Completed table:    {summary_path}")


if __name__ == "__main__":
    main()

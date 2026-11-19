"""Controlled local-model comparison for the FoodTraceGraphRAG framework.

The script holds retrieval, verified evidence, prompt, temperature, seed,
validation, and fallback constant while varying only the Ollama model.
Quality metrics are calculated on the direct candidate answer so deterministic
fallback cannot mask differences between models.  Every measured execution is
checkpointed and the completed journal table is written automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import requests

from app import (
    Services,
    call_ollama,
    deterministic_answer,
    retrieve_batch,
    validate_answer,
    verify_evidence,
)
from claim_level_evaluation import (
    answer_metrics,
    factual_package,
    load_dataset,
    rule_based_judgment,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_expanded"
OUTPUT_DIR = ROOT / "outputs" / "model_comparison"

MODELS = [
    ("llama3.2:1b", "Llama 3.2:1B", "1B"),
    ("llama3.2:3b", "Llama 3.2:3B", "3B"),
    ("qwen2.5:1.5b-instruct", "Qwen2.5:1.5B-Instruct", "1.5B"),
]

STRATIFIED_BATCHES = [
    "BATCH-FISH-001", "BATCH-FISH-007",
    "BATCH-MEAT-014", "BATCH-MEAT-015",
    "BATCH-VEG-023", "BATCH-VEG-030",
    "BATCH-DAIRY-036", "BATCH-DAIRY-038",
    "BATCH-PROCFOOD-042", "BATCH-PROCFOOD-049",
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], proportion: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * proportion
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def ollama_base(chat_url: str) -> str:
    return chat_url.rsplit("/", 1)[0]


def installed_models(chat_url: str) -> set[str]:
    response = requests.get(f"{ollama_base(chat_url)}/tags", timeout=30)
    response.raise_for_status()
    names = set()
    for model in response.json().get("models", []):
        names.add(model.get("name", ""))
        names.add(model.get("model", ""))
    return {x for x in names if x}


def require_models(chat_url: str) -> None:
    installed = installed_models(chat_url)
    missing = [tag for tag, _, _ in MODELS if tag not in installed]
    if missing:
        commands = "\n".join(f"ollama pull {tag}" for tag in missing)
        raise RuntimeError(f"Required Ollama model(s) missing: {missing}\nRun:\n{commands}")


def loaded_model_bytes(chat_url: str, model_tag: str) -> tuple[int, int]:
    """Return Ollama loaded size and VRAM size, when exposed by /api/ps."""
    try:
        response = requests.get(f"{ollama_base(chat_url)}/ps", timeout=30)
        response.raise_for_status()
        for model in response.json().get("models", []):
            name = model.get("name") or model.get("model") or ""
            if name == model_tag or name.split(":")[0] == model_tag.split(":")[0]:
                return int(model.get("size", 0) or 0), int(model.get("size_vram", 0) or 0)
    except requests.RequestException:
        pass
    return 0, 0


def unload_model(chat_url: str, model_tag: str) -> None:
    try:
        requests.post(
            f"{ollama_base(chat_url)}/generate",
            json={"model": model_tag, "keep_alive": 0},
            timeout=60,
        ).raise_for_status()
    except requests.RequestException:
        pass


def make_question(batch: dict) -> str:
    return (
        f"Reconstruct the complete available traceability record for {batch['batch_id']} "
        f"({batch['product']}), report its status, and identify any missing, "
        "conflicting, inconsistent-identifier, inspection, or cold-chain evidence."
    )


def verified_cases(services: Services, batches: dict, truths: dict) -> dict[str, dict]:
    cases = {}
    for batch_id in STRATIFIED_BATCHES:
        graph_result = retrieve_batch(services, batch_id)
        if graph_result is None:
            raise RuntimeError(f"Batch not found in Neo4j: {batch_id}")
        verified = verify_evidence(services, graph_result)
        if not verified or not all(item["verified"] for item in verified):
            raise RuntimeError(f"Integrity verification failed before comparison: {batch_id}")
        cases[batch_id] = {
            "batch": batches[batch_id],
            "truth": truths[batch_id],
            "graph_result": graph_result,
            "verified": verified,
            "question": make_question(batches[batch_id]),
        }
    return cases


def retry_generation(function, attempts: int = 3) -> str:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return function()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < attempts:
                print(f"Temporary Ollama error; retrying {attempt}/{attempts - 1}")
                time.sleep(5)
    raise RuntimeError(f"Generation failed after {attempts} attempts: {last_error}")


def claim_metrics_for_candidate(
    candidate: str,
    model_tag: str,
    batch_id: str,
    case: dict,
    events_by_batch: dict[str, list[dict]],
) -> dict:
    expected_ids = case["truth"]["expected_evidence_ids"]
    verified_ids = ";".join(sorted(item["evidence_id"] for item in case["verified"]))
    raw_like = {
        "configuration_id": model_tag,
        "configuration": model_tag,
        "batch_id": batch_id,
        "expected_ids": expected_ids,
        "retrieved_ids": verified_ids,
        "integrity_passed": "1",
        "validation_passed": "",
        "fallback_used": "",
    }
    package = factual_package(
        raw_like, case["batch"], events_by_batch[batch_id], case["truth"]
    )
    judgment = rule_based_judgment(candidate, package)
    item = {
        "answer_id": f"candidate|{model_tag}|{batch_id}",
        "answer_field": "candidate_answer",
        "required_units": package["required_units"],
        "source": {
            "configuration_id": model_tag,
            "configuration": model_tag,
            "batch_id": batch_id,
        },
        "judgment": judgment,
    }
    metrics, _ = answer_metrics(item)
    return metrics


def run_measured(
    services: Services,
    model_tag: str,
    model_name: str,
    parameters: str,
    batch_id: str,
    repetition: int,
    case: dict,
    events_by_batch: dict[str, list[dict]],
) -> dict:
    services.ollama_model = model_tag
    start = time.perf_counter()
    candidate = retry_generation(
        lambda: call_ollama(services, case["question"], case["verified"])
    )
    generation_ms = (time.perf_counter() - start) * 1000
    validation = validate_answer(candidate, case["verified"])
    direct_valid = bool(validation["passed"])
    fallback_used = not direct_valid
    released = (
        deterministic_answer(case["graph_result"], case["verified"])
        if fallback_used else candidate
    )
    size_bytes, vram_bytes = loaded_model_bytes(services.ollama_api, model_tag)
    claim = claim_metrics_for_candidate(
        candidate, model_tag, batch_id, case, events_by_batch
    )
    return {
        "model_tag": model_tag,
        "model": model_name,
        "parameters": parameters,
        "batch_id": batch_id,
        "category": case["batch"]["category"],
        "repetition": repetition,
        "direct_validation_passed": int(direct_valid),
        "fallback_used": int(fallback_used),
        "generation_ms": round(generation_ms, 3),
        "loaded_model_size_bytes": size_bytes,
        "loaded_vram_bytes": vram_bytes,
        "candidate_answer": candidate.replace("\n", " "),
        "released_answer": released.replace("\n", " "),
        "total_claims": claim["total_claims"],
        "supported_claims": claim["supported_claims"],
        "unsupported_claims": claim["unsupported_claims"],
        "contradicted_claims": claim["contradicted_claims"],
        "incomplete_claims": claim["incomplete_claims"],
        "misattributed_claims": claim["misattributed_claims"],
        "citation_pairs": claim["citation_pairs"],
        "correct_citation_pairs": claim["correct_citation_pairs"],
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["model_tag"]].append(row)
    output = []
    order = {tag: index for index, (tag, _, _) in enumerate(MODELS)}
    for model_tag, group in sorted(groups.items(), key=lambda x: order[x[0]]):
        generation = [float(x["generation_ms"]) for x in group]
        claims = sum(int(x["total_claims"]) for x in group)
        supported = sum(int(x["supported_claims"]) for x in group)
        unsupported = sum(int(x["unsupported_claims"]) for x in group)
        citation_pairs = sum(int(x["citation_pairs"]) for x in group)
        correct_pairs = sum(int(x["correct_citation_pairs"]) for x in group)
        loaded = max(int(x["loaded_model_size_bytes"]) for x in group)
        vram = max(int(x["loaded_vram_bytes"]) for x in group)
        output.append({
            "Model": group[0]["model"],
            "Parameters": group[0]["parameters"],
            "Answers": len(group),
            "Direct validation (%)": round(
                100 * statistics.fmean(int(x["direct_validation_passed"]) for x in group), 2
            ),
            "Claim precision (%)": round(100 * supported / claims, 2) if claims else 0.0,
            "Unsupported claims (%)": round(100 * unsupported / claims, 2) if claims else 0.0,
            "Citation precision (%)": round(100 * correct_pairs / citation_pairs, 2)
            if citation_pairs else 0.0,
            "Fallback (%)": round(
                100 * statistics.fmean(int(x["fallback_used"]) for x in group), 2
            ),
            "Median generation latency (ms)": round(statistics.median(generation), 2),
            "p95 generation latency (ms)": round(percentile(generation, 0.95), 2),
            "Loaded model size (GB)": round(loaded / (1024 ** 3), 2) if loaded else "N/A",
            "Loaded VRAM (GB)": round(vram / (1024 ** 3), 2) if vram else 0.0,
        })
    return output


def save(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "model_comparison_raw.csv", rows)
    write_csv(OUTPUT_DIR / "model_comparison_summary.csv", summarize(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Ollama model comparison")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run one batch once per model before the full experiment")
    parser.add_argument("--fresh", action="store_true",
                        help="Start a new comparison instead of resuming")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    require_models("http://127.0.0.1:11434/api/chat")

    batches, events_by_batch, truths = load_dataset(DATA_DIR)
    services = Services()
    raw_path = OUTPUT_DIR / "model_comparison_raw.csv"
    rows = [] if args.fresh else read_csv(raw_path)
    completed = {
        (x["model_tag"], x["batch_id"], int(x["repetition"])) for x in rows
    }
    try:
        cases = verified_cases(services, batches, truths)
        selected_batches = STRATIFIED_BATCHES[:1] if args.smoke_test else STRATIFIED_BATCHES
        repetitions = 1 if args.smoke_test else args.repetitions
        for model_tag, model_name, parameters in MODELS:
            services.ollama_model = model_tag
            print(f"\nLoading and warming up {model_name} ({model_tag})")
            retry_generation(
                lambda: call_ollama(
                    services,
                    cases[selected_batches[0]]["question"],
                    cases[selected_batches[0]]["verified"],
                )
            )
            for repetition in range(1, repetitions + 1):
                for batch_id in selected_batches:
                    key = (model_tag, batch_id, repetition)
                    if key in completed:
                        print(f"SKIP completed | {model_tag} | {batch_id} | {repetition}")
                        continue
                    print(
                        f"{model_tag} | repetition {repetition}/{repetitions} | {batch_id}"
                    )
                    row = run_measured(
                        services, model_tag, model_name, parameters, batch_id,
                        repetition, cases[batch_id], events_by_batch,
                    )
                    rows.append(row)
                    completed.add(key)
                    save(rows)
                    print(f"Checkpoint saved: {len(rows)} measured generations")
            unload_model(services.ollama_api, model_tag)
        save(rows)
        print(f"\nRaw results: {OUTPUT_DIR / 'model_comparison_raw.csv'}")
        print(f"Completed table: {OUTPUT_DIR / 'model_comparison_summary.csv'}")
    finally:
        services.close()


if __name__ == "__main__":
    main()

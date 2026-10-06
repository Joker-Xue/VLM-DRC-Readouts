"""Evaluate completed corrected-debate controls from their immutable saved records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.agents.audited_debate import atomic_json, file_hash
from experiments.eval.audited_paper_metrics import bootstrap, metrics


READOUTS = ("single", "r1", "own_fixed", "peer_fixed", "own_early", "peer_early")


def extract(configuration: dict, name: str) -> dict:
    if name in ("single", "r1"):
        return configuration[name]
    branch, policy = name.split("_", 1)
    return configuration["branches"][branch][policy]


def unique_messages(configuration: dict) -> list[dict]:
    own = configuration["branches"]["own"]["rounds"]
    peer = configuration["branches"]["peer"]["rounds"]
    return own[0] + own[1] + own[2] + peer[1] + peer[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=3000)
    args = parser.parse_args()

    manifest_path = args.run / "manifest.json"
    per_region_path = args.run / "per_region.json"
    manifest = json.loads(manifest_path.read_text())
    rows = json.loads(per_region_path.read_text())
    expected = manifest["candidate_ids"]
    actual = [row["patch_id"] for row in rows]
    if actual != expected:
        raise RuntimeError(f"Run is incomplete or reordered: expected {len(expected)}, found {len(actual)}")

    labels = [row["is_hotspot"] for row in rows]
    if len(set(labels)) < 2:
        raise ValueError("ROC-AUC requires both classes")

    report = {
        "run": str(args.run),
        "manifest_sha256": file_hash(manifest_path),
        "per_region_sha256": file_hash(per_region_path),
        "n": len(rows),
        "positives": sum(labels),
        "interpretation": (
            "Fixed own-history versus peer-history is the prespecified communication comparison. "
            "Early stopping is an operational secondary readout. Nominal region-bootstrap intervals "
            "do not capture dependence across designs or model/runtime configurations."
        ),
        "configurations": {},
    }

    for config in manifest["configurations"]:
        scores = {
            name: [extract(row["configurations"][config], name)["score"] for row in rows]
            for name in READOUTS
        }
        decisions = {
            name: [extract(row["configurations"][config], name)["decision"] for row in rows]
            for name in READOUTS
        }
        messages = [
            message
            for row in rows
            for message in unique_messages(row["configurations"][config])
        ]
        call_records = {record for message in messages for record in message["call_records"]}
        config_report = {
            "readouts": {
                name: metrics(scores[name], labels, decisions[name]) for name in READOUTS
            },
            "provenance": {
                "logical_messages": len(messages),
                "generation_attempt_records": len(call_records),
                "messages_with_retry": sum(len(message["call_records"]) > 1 for message in messages),
                "fallback_messages": sum(bool(message.get("fallback")) for message in messages),
            },
        }
        if args.bootstrap:
            config_report["paired_bootstrap"] = {
                "peer_fixed_minus_own_fixed": bootstrap(
                    scores["own_fixed"], scores["peer_fixed"], labels, args.bootstrap
                ),
                "peer_early_minus_own_early": bootstrap(
                    scores["own_early"], scores["peer_early"], labels, args.bootstrap
                ),
            }
        report["configurations"][config] = config_report

    atomic_json(args.out, report)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

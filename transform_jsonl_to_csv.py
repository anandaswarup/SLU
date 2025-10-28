"""
Transform JSONL in the following format:

{"audio_filepath": ".../594/128329/594-128329-0025.wav", "transcript": "...", "annotations": {"action":"none","object":"none","location":"none"}}

into CSV with following columns:

path, transcription, action, slot_name, slot_value
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path):
    """
    Load a JSONL file and yield each record as a dictionary.
    """
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"[WARN] Skipping line {lineno}: JSON decode error: {e}",
                    file=sys.stderr,
                )


def main():
    """
    Main method to transform JSONL to CSV.
    """
    # Setup command line argument parser
    parser = argparse.ArgumentParser(
        description="Transform JSONL (audio/transcript/annotations) to CSV."
    )
    parser.add_argument("--input_jsonl", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)

    # Parse and get arguments
    args = parser.parse_args()

    # Process each record in the JSONL
    rows = []
    for rec in load_jsonl(args.input_jsonl):
        audio_path = (
            rec.get("audio_filepath") or rec.get("path") or rec.get("audio") or ""
        )
        if not audio_path:
            # Skip records without a usable path
            continue

        transcript = (
            rec.get("transcript") or rec.get("text") or rec.get("transcription") or ""
        )

        ann = rec.get("annotations") or {}

        # normalize keys
        action = (ann.get("action") or "none").strip()
        obj = (ann.get("object") or "none").strip()
        loc = (ann.get("location") or "none").strip()

        rows.append(
            {
                "path": audio_path,
                "transcription": transcript,
                "action": action,
                "slot_name": obj,
                "slot_value": loc,
            }
        )

    if not rows:
        print("[ERROR] No valid records found in input JSONL.", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(
        rows,
        columns=["path", "transcription", "action", "slot_name", "slot_value"],
    )

    # Write with an index column to match your shown CSV (leading comma at header)
    df.to_csv(args.output_csv, index=True)
    print(f"[OK] Wrote {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()

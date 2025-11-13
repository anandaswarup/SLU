"""
Build CSV manifests from SLU jsonl datasets.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import jsonlines
import pandas as pd


def _infer_split_from_name(filename: str) -> Optional[str]:
    """
    Return the canonical dataset split inferred from a jsonl filename.
    """

    normalized = filename.lower().replace("-", "_")
    base_name = normalized.split("/")[-1]

    direct_matches = {"train", "dev", "test", "unseen"}
    if base_name in direct_matches:
        return base_name

    for token in base_name.split("_"):
        if token in direct_matches:
            return token
    return None


def _collect_jsonl_paths(data_root: Path) -> Dict[str, List[Path]]:
    """
    Group jsonl files by split name based on the directory structure.
    """

    split_to_paths: Dict[str, List[Path]] = {}
    for path in data_root.rglob("*.jsonl"):
        split_name = _infer_split_from_name(path.stem)
        if split_name is None:
            continue
        split_to_paths.setdefault(split_name, []).append(path)

    for paths in split_to_paths.values():
        paths.sort()
    return split_to_paths


def _read_jsonl_records(file_path: Path, next_id: int) -> Tuple[List[dict], int]:
    """
    Load entries from a jsonl file and convert them to manifest rows.
    """

    rows: List[dict] = []
    current_id = next_id

    with jsonlines.open(file_path) as reader:
        for sample in reader:
            try:
                audio_path = sample["audio_filepath"]
            except KeyError as exc:
                raise ValueError(f"Missing audio filepath in {file_path}") from exc

            duration_raw: Union[str, float, int, None] = sample.get("duration")
            if duration_raw in ("", None):
                duration: Union[str, float] = ""
            else:
                try:
                    duration = float(duration_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid duration value in {file_path}: {duration_raw}"
                    ) from exc

            semantics = _normalise_semantics(
                sample.get("semantics") or sample.get("text") or ""
            )

            transcript = sample.get("transcript", "")

            rows.append(
                {
                    "ID": current_id,
                    "duration": duration,
                    "wav": audio_path,
                    "semantics": semantics,
                    "transcript": transcript,
                }
            )
            current_id += 1

    return rows, current_id


def _normalise_semantics(raw_semantics: Union[str, dict]) -> str:
    """
    Return a readable string representation of the semantics dictionary.
    """

    if isinstance(raw_semantics, dict):
        return _format_semantics(raw_semantics)

    if not isinstance(raw_semantics, str):
        return str(raw_semantics)

    candidate = raw_semantics.strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return raw_semantics
        if isinstance(parsed, dict):
            return _format_semantics(parsed)

    return raw_semantics


def _format_semantics(value: Union[dict, list, str, int, float, None]) -> str:
    """
    Recursively format semantics data using a lightweight dict-style string.
    """

    if isinstance(value, dict):
        parts = []
        for key, inner in value.items():
            formatted_inner = _format_semantics(inner)
            parts.append(f"{key}: {formatted_inner}")
        return "{" + ", ".join(parts) + "}"

    if isinstance(value, list):
        items = ", ".join(_format_semantics(item) for item in value)
        return f"[{items}]"

    if value is None:
        return "None"

    if isinstance(value, str):
        return value if value != "" else "null"

    return str(value)


def _ordered_splits(splits: Iterable[str]) -> List[str]:
    """
    Return dataset splits ordered by standard SLU evaluation preference.
    """

    priority = {"train": 0, "dev": 1, "test": 2, "unseen": 3}

    return sorted(splits, key=lambda item: (priority.get(item, 99), item))


def prepare_SLU_dataset(data_folder, save_folder, skip_prep=False):
    """
    Create CSV manifests for SLU experiments from jsonl sources.
    """

    if skip_prep:
        return

    data_root = Path(data_folder)
    save_root = Path(save_folder)
    save_root.mkdir(parents=True, exist_ok=True)

    split_to_paths = _collect_jsonl_paths(data_root)
    if not split_to_paths:
        raise FileNotFoundError(
            f"No jsonl files discovered under {data_root} — cannot prepare manifests"
        )

    next_id = 0
    for split in _ordered_splits(split_to_paths.keys()):
        output_path = save_root / f"{split}.csv"
        if output_path.exists():
            continue

        print(f"Preparing {output_path.name}...")

        rows: List[dict] = []
        for jsonl_path in split_to_paths[split]:
            parsed_rows, next_id = _read_jsonl_records(jsonl_path, next_id)
            rows.extend(parsed_rows)

        if not rows:
            continue

        df = pd.DataFrame(
            rows,
            columns=["ID", "duration", "wav", "semantics", "transcript"],
        )
        df.to_csv(output_path, index=False)

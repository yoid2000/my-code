"""
Distributed labeling utility for AOL query data.

Modes:
- make_distinct: build distinct_queries.parquet from raw.parquet Query values
- <i>: label chunk i (0-based) of distinct queries and write label_work/i.parquet
- create: combine label_work/*.parquet with raw.parquet into labeled.parquet
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import usaddress
from gliner import GLiNER


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_PATH = BASE_DIR / "raw.parquet"
DEFAULT_OUTPUT_PATH = BASE_DIR / "labeled.parquet"
DEFAULT_DISTINCT_PATH = BASE_DIR / "distinct_queries.parquet"
DEFAULT_LABEL_WORK_DIR = BASE_DIR / "label_work"
DEFAULT_NUM_CHUNKS = 200

TARGET_LABELS = [
    "full_name",
    "street_city_address",
    "place_name",
    "profession",
    "social_security_number",
    "email_address",
    "credit_card_number",
    "phone_number",
    "other_alpha_numeric_string",
]

LOCATION_LABELS = {"street_city_address", "place_name"}
STREET_COMPONENT_KEYS = {
    "AddressNumber",
    "StreetName",
    "StreetNamePreDirectional",
    "StreetNamePreModifier",
    "StreetNamePreType",
    "StreetNamePostDirectional",
    "StreetNamePostModifier",
    "StreetNamePostType",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distributed labeler for raw.parquet. Modes: make_distinct, <chunk_index>, create."
        )
    )
    parser.add_argument(
        "mode",
        help="One of: make_distinct, create, or chunk index i (0-based integer).",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help=f"Input parquet path (default: {DEFAULT_RAW_PATH}).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--distinct-path",
        type=Path,
        default=DEFAULT_DISTINCT_PATH,
        help=f"Distinct query parquet path (default: {DEFAULT_DISTINCT_PATH}).",
    )
    parser.add_argument(
        "--label-work-dir",
        type=Path,
        default=DEFAULT_LABEL_WORK_DIR,
        help=f"Directory for chunk label outputs (default: {DEFAULT_LABEL_WORK_DIR}).",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help=f"Number of chunks for distributed labeling (default: {DEFAULT_NUM_CHUNKS}).",
    )
    parser.add_argument(
        "--model-id",
        default="gliner-community/gliner_medium-v2.5",
        help="GLiNER model id for from_pretrained().",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum GLiNER confidence score (default: 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for GLiNER inference (default: 128).",
    )
    return parser.parse_args()


def usaddress_components(text: str) -> set[str]:
    """Return component tag names found by usaddress for a location-like text."""
    try:
        tagged, _ = usaddress.tag(text)
        return set(tagged.keys())
    except usaddress.RepeatedLabelError as exc:
        return {label for _, label in exc.parsed_string}
    except Exception:
        return set()


def normalize_location_label(span_text: str) -> str | None:
    """
    Normalize location labels to:
    - street_city_address when street + city are present
    - place_name when street is absent
    Returns None when street exists but city is missing.
    """
    components = usaddress_components(span_text)
    has_street = bool(components & STREET_COMPONENT_KEYS)
    has_city = "PlaceName" in components

    if has_street and has_city:
        return "street_city_address"
    if has_street and not has_city:
        return None
    return "place_name"


def normalize_entity(entity: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one GLiNER entity dict and enforce location constraints."""
    label = str(entity["label"])
    text = str(entity["text"])

    if label in LOCATION_LABELS:
        normalized = normalize_location_label(text)
        if normalized is None:
            return None
        label = normalized

    return {
        "label": label,
        "text": text,
        "start": int(entity["start"]),
        "end": int(entity["end"]),
        "score": float(entity["score"]),
    }


def label_queries(queries: list[str], model: GLiNER, threshold: float, batch_size: int) -> list[list[dict[str, Any]]]:
    """Label all distinct queries and return mapping query -> list of span labels."""
    all_labels: list[list[dict[str, Any]]] = []

    total = len(queries)
    for offset in range(0, total, batch_size):
        batch = queries[offset : offset + batch_size]
        batch_entities = predict_entities_batch(
            model=model,
            texts=batch,
            labels=TARGET_LABELS,
            threshold=threshold,
        )

        for entities in batch_entities:
            normalized_entities: list[dict[str, Any]] = []
            for entity in entities:
                normalized = normalize_entity(entity)
                if normalized is not None:
                    normalized_entities.append(normalized)
            all_labels.append(normalized_entities)

        processed = min(offset + batch_size, total)
        if processed % 100_000 == 0 or processed == total:
            print(f"Labeled {processed:,}/{total:,} distinct queries")

    return all_labels


def predict_entities_batch(
    model: GLiNER, texts: list[str], labels: list[str], threshold: float
) -> list[list[dict[str, Any]]]:
    """
    Predict entities for a batch across GLiNER API variants.

    Supported in priority order:
    - model.inference(...)
    - model.run(...)
    - model.batch_predict_entities(...)
    - repeated model.predict_entities(...)
    """
    if hasattr(model, "inference"):
        try:
            return model.inference(texts, labels, threshold=threshold)
        except TypeError:
            return model.inference(texts, labels)

    if hasattr(model, "run"):
        return model.run(texts, labels=labels, threshold=threshold)

    if hasattr(model, "batch_predict_entities"):
        try:
            return model.batch_predict_entities(texts, labels, threshold=threshold)
        except TypeError:
            return model.batch_predict_entities(texts, labels)

    if hasattr(model, "predict_entities"):
        out: list[list[dict[str, Any]]] = []
        for text in texts:
            try:
                out.append(model.predict_entities(text, labels, threshold=threshold))
            except TypeError:
                out.append(model.predict_entities(text, labels))
        return out

    raise AttributeError("Unsupported GLiNER API on model instance.")


def build_distinct_queries(raw_path: Path, distinct_path: Path) -> None:
    """Write a parquet file containing one distinct Query per row."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {raw_path}")

    print(f"Loading data from: {raw_path}")
    df = pd.read_parquet(raw_path, columns=["Query"])
    query_series = df["Query"].astype("string").fillna("")
    distinct_df = pd.DataFrame({"Query": query_series.drop_duplicates().tolist()})

    distinct_path.parent.mkdir(parents=True, exist_ok=True)
    distinct_df.to_parquet(distinct_path, index=False)
    print(f"Wrote {len(distinct_df):,} distinct queries to: {distinct_path}")


def chunk_bounds(total: int, index: int, num_chunks: int) -> tuple[int, int]:
    """Return [start, end) indices for chunk index under floor-based partitioning."""
    start = (index * total) // num_chunks
    end = ((index + 1) * total) // num_chunks
    return start, end


def run_chunk_labeling(
    chunk_index: int,
    distinct_path: Path,
    label_work_dir: Path,
    num_chunks: int,
    model_id: str,
    threshold: float,
    batch_size: int,
) -> None:
    """Label a single distinct-query chunk and write label_work/<i>.parquet."""
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError(f"Chunk index must be in [0, {num_chunks - 1}], got {chunk_index}.")
    if not distinct_path.exists():
        raise FileNotFoundError(
            f"Distinct query file not found: {distinct_path}. Run mode make_distinct first."
        )

    distinct_df = pd.read_parquet(distinct_path, columns=["Query"])
    distinct_queries = distinct_df["Query"].astype("string").fillna("").tolist()
    total = len(distinct_queries)
    start, end = chunk_bounds(total, chunk_index, num_chunks)
    queries = distinct_queries[start:end]

    print(
        f"Chunk {chunk_index}/{num_chunks - 1}: rows [{start:,}, {end:,}) "
        f"({len(queries):,} queries)"
    )
    print(f"Loading GLiNER model: {model_id}")
    model = GLiNER.from_pretrained(model_id)

    labels = label_queries(queries, model=model, threshold=threshold, batch_size=batch_size)

    out_df = pd.DataFrame({"Query": queries, "QueryLabels": labels})
    label_work_dir.mkdir(parents=True, exist_ok=True)
    out_path = label_work_dir / f"{chunk_index}.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote chunk labels: {out_path}")

    label_counter: Counter[str] = Counter()
    rows_with_labels = 0
    for row_labels in labels:
        if row_labels:
            rows_with_labels += 1
        for span in row_labels:
            label_counter[str(span["label"])] += 1

    print(f"Chunk queries: {len(queries):,}")
    print(f"Chunk rows with at least one label: {rows_with_labels:,}")
    print("Chunk span counts by label:")
    for label in TARGET_LABELS:
        print(f"  {label}: {label_counter[label]:,}")


def create_labeled_parquet(
    raw_path: Path,
    label_work_dir: Path,
    output_path: Path,
    num_chunks: int,
) -> None:
    """Join all label chunk files to raw.parquet and write labeled.parquet."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {raw_path}")
    if not label_work_dir.exists():
        raise FileNotFoundError(f"Label work directory not found: {label_work_dir}")

    expected_files = [label_work_dir / f"{i}.parquet" for i in range(num_chunks)]
    missing_files = [str(path) for path in expected_files if not path.exists()]
    if missing_files:
        preview = ", ".join(missing_files[:10])
        suffix = "..." if len(missing_files) > 10 else ""
        raise FileNotFoundError(
            f"Missing {len(missing_files)} chunk files in {label_work_dir}: {preview}{suffix}"
        )

    label_frames = [pd.read_parquet(path, columns=["Query", "QueryLabels"]) for path in expected_files]
    labels_df = pd.concat(label_frames, ignore_index=True)
    labels_df = labels_df.drop_duplicates(subset=["Query"], keep="last")

    print(f"Loading raw data from: {raw_path}")
    raw_df = pd.read_parquet(raw_path)
    labeled_df = raw_df.merge(labels_df, on="Query", how="left")
    labeled_df["QueryLabels"] = labeled_df["QueryLabels"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_parquet(output_path, index=False)
    print(f"Wrote merged labeled parquet: {output_path}")
    print(f"Rows: {len(labeled_df):,}")


def main() -> None:
    args = parse_args()
    mode = args.mode.strip().lower()

    if mode == "make_distinct":
        build_distinct_queries(args.raw_path, args.distinct_path)
        return

    if mode == "create":
        create_labeled_parquet(
            raw_path=args.raw_path,
            label_work_dir=args.label_work_dir,
            output_path=args.output_path,
            num_chunks=args.num_chunks,
        )
        return

    try:
        chunk_index = int(mode)
    except ValueError as exc:
        raise ValueError(
            "Mode must be make_distinct, create, or an integer chunk index."
        ) from exc

    run_chunk_labeling(
        chunk_index=chunk_index,
        distinct_path=args.distinct_path,
        label_work_dir=args.label_work_dir,
        num_chunks=args.num_chunks,
        model_id=args.model_id,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

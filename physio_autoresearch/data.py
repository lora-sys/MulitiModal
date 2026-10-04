"""Explicit ingestion, data QC and subject-level splits."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ResearchError, digest


def parse_data(raw: bytes, contract: dict, prediction_only: bool = False) -> pd.DataFrame:
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue  # Match pandas' blank-line handling.
            if len(row) != len(header):
                raise ResearchError("CORRUPTED_DATA", f"CSV row ending at line {reader.line_num} has {len(row)} fields; expected {len(header)}")
    except (UnicodeError, StopIteration, csv.Error) as exc:
        raise ResearchError("CORRUPTED_DATA", str(exc)) from exc
    if len(set(header)) != len(header):
        raise ResearchError("SCHEMA_DRIFT", "duplicate column names")
    fields = contract["features"] + ([] if prediction_only else contract["targets"])
    required = [contract["record_id"], contract["subject_id"]] + [f["name"] for f in fields]
    if set(header) != set(required):
        raise ResearchError("SCHEMA_DRIFT", f"missing={sorted(set(required)-set(header))}; unknown={sorted(set(header)-set(required))}")
    try:
        frame = pd.read_csv(io.StringIO(text), dtype={contract["record_id"]: "string", contract["subject_id"]: "string"})
    except (ValueError, pd.errors.ParserError) as exc:
        raise ResearchError("CORRUPTED_DATA", str(exc)) from exc
    if frame.empty:
        raise ResearchError("EMPTY_DATA", "no records")
    for col in (contract["record_id"], contract["subject_id"]):
        if frame[col].isna().any() or frame[col].str.strip().eq("").any():
            raise ResearchError("MISSING_ID", col)
        if frame[col].str.strip().ne(frame[col]).any():
            raise ResearchError("BAD_ID", "leading/trailing whitespace is forbidden")
    if frame[contract["record_id"]].duplicated().any():
        raise ResearchError("DUPLICATE_RECORD", "record_id must identify a unique session/measurement")
    for field in fields:
        name = field["name"]
        try:
            frame[name] = pd.to_numeric(frame[name], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ResearchError("NONNUMERIC_DATA", name) from exc
        values = frame[name].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ResearchError("NONFINITE_DATA", name)
        if field["phase"] == "post" and np.isnan(values).any():
            raise ResearchError("MISSING_TARGET", name)
        if "range" in field:
            low, high = field["range"]
            if ((values < low) | (values > high)).any():
                raise ResearchError("OUT_OF_RANGE", name)
    return frame


def load_data(path: Path, contract: dict) -> pd.DataFrame:
    return parse_data(path.read_bytes(), contract)


def subject_folds(frame: pd.DataFrame, contract: dict) -> list[dict]:
    subjects = sorted(frame[contract["subject_id"]].unique().tolist())
    if len(subjects) < 2:
        return []  # One person can prove fitting only; never emit a validation score.
    record, subject = contract["record_id"], contract["subject_id"]
    return [{"train_ids": frame.loc[frame[subject] != held, record].tolist(),
             "validation_ids": frame.loc[frame[subject] == held, record].tolist(),
             "validation_subject": held} for held in subjects]


def check_fold(frame: pd.DataFrame, contract: dict, fold: dict):
    indexed = frame.set_index(contract["record_id"])
    train, valid = fold["train_ids"], fold["validation_ids"]
    if not train or not valid or set(train) & set(valid):
        raise ResearchError("SUBJECT_LEAKAGE", "empty or overlapping split")
    if len(set(train)) != len(train) or len(set(valid)) != len(valid) or set(train) | set(valid) != set(indexed.index):
        raise ResearchError("INCOMPLETE_SPLIT", "every record must be present exactly once")
    subject = contract["subject_id"]
    if set(indexed.loc[train, subject]) & set(indexed.loc[valid, subject]):
        raise ResearchError("SUBJECT_LEAKAGE", "sessions from one person cross the split")


def summary(frame: pd.DataFrame, contract: dict) -> dict:
    return {"rows": len(frame), "subjects": frame[contract["subject_id"]].nunique(),
            "missing_features": {f["name"]: int(frame[f["name"]].isna().sum()) for f in contract["features"]},
            "constant_features": [f["name"] for f in contract["features"] if frame[f["name"]].nunique() <= 1],
            "split_sha256": digest(subject_folds(frame, contract))}

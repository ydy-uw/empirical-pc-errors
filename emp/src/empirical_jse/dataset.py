"""Readers for verified annual CRSP snapshots and derived Parquet panels."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = (
    "permco",
    "dlyprc",
    "dlycap",
    "dlyret",
    "dlyretx",
    "dlyreti",
    "dlyvol",
    "dlycumfacpr",
    "dlycumfacshr",
    "shrout",
)

CATEGORICAL_COLUMNS = (
    "sharetype",
    "securitytype",
    "securitysubtype",
    "usincflg",
    "issuertype",
    "primaryexch",
    "conditionaltype",
    "tradingstatusflg",
    "dlydelflg",
    "dlyprcflg",
    "dlycapflg",
    "dlyretmissflg",
    "dlyretdurflg",
    "dlydistretflg",
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_child(root: Path, relative: str | Path) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"manifest path escapes its snapshot root: {relative}")
    return candidate


def verify_file_records(
    root: str | Path,
    records: Iterable[dict[str, object]],
) -> None:
    """Verify size and SHA-256 fields in an immutable artifact manifest."""
    base = Path(root)
    for record in records:
        path = _safe_child(base, str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"manifest artifact is missing: {path}")
        if "bytes" in record and path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"manifest byte-size mismatch: {path}")
        expected = str(record.get("sha256", ""))
        if not expected:
            raise ValueError(f"manifest artifact has no SHA-256 digest: {path}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"manifest SHA-256 mismatch: {path}")


def verify_reference_snapshot(snapshot: str | Path) -> dict[str, object]:
    """Verify all artifacts in a CRSP reference snapshot and return its manifest."""
    root = Path(snapshot)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing reference manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"reference manifest contains no files: {manifest_path}")
    verify_file_records(root, records)
    listed = {str(record["path"]).replace("\\", "/") for record in records}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if listed != actual:
        missing = sorted(listed - actual)
        extra = sorted(actual - listed)
        raise RuntimeError(
            "reference manifest membership mismatch; "
            f"missing={missing}, extra={extra}"
        )
    return manifest


def normalize_ciz_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize CSV-inferred CIZ dtypes without changing return values."""
    result = frame.copy()
    result = result.assign(
        dlycaldt=pd.to_datetime(
            result["dlycaldt"],
            errors="raise",
        ).dt.normalize()
    )
    permno = pd.to_numeric(result["permno"], errors="raise")
    if permno.isna().any():
        raise ValueError("PERMNO cannot be missing")
    conversions: dict[str, pd.Series] = {
        "permno": permno.astype("int64"),
    }
    conversions.update(
        {
            column: pd.to_numeric(result[column], errors="coerce")
            for column in NUMERIC_COLUMNS
            if column in result
        }
    )
    conversions.update(
        {
            column: result[column].astype("string").astype("category")
            for column in CATEGORICAL_COLUMNS
            if column in result
        }
    )
    conversions.update(
        {
            column: result[column].astype("string")
            for column in ("ticker", "cusip")
            if column in result
        }
    )
    result = result.assign(**conversions)
    return result


def load_raw_annual_csv(path: str | Path) -> pd.DataFrame:
    """Load one immutable annual gzip CSV while preserving literal ``NA`` flags."""
    frame = pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    return normalize_ciz_frame(frame)


def trading_calendar_from_reference(snapshot: str | Path) -> pd.DatetimeIndex:
    """Load the authoritative CIZ trading calendar from a reference snapshot."""
    root = Path(snapshot)
    manifest = verify_reference_snapshot(root)
    calendar_records = [
        record
        for record in manifest["files"]
        if record["path"] == "calendar/calendar.csv.gz"
    ]
    if len(calendar_records) != 1:
        raise RuntimeError(
            "reference manifest must contain exactly one CIZ calendar artifact"
        )
    path = _safe_child(root, str(calendar_records[0]["path"]))
    frame = pd.read_csv(path, keep_default_na=False)
    trading = frame.loc[
        frame["tradingflg"].eq("Y") & frame["exchangegroup"].eq("CIZ"),
        "caldt",
    ]
    index = pd.DatetimeIndex(pd.to_datetime(trading, errors="raise")).normalize()
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("CIZ reference calendar is not unique and increasing")
    return index


def monthly_formation_dates(
    calendar: pd.DatetimeIndex,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    lookback: int = 63,
) -> tuple[pd.Timestamp, ...]:
    """Return month-final trading dates with lookback and next-month OOS data."""
    dates = pd.DatetimeIndex(calendar).normalize()
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("calendar must be unique and increasing")
    series = pd.Series(dates, index=dates)
    month_ends = series.groupby(dates.to_period("M")).max()
    candidates: list[pd.Timestamp] = []
    for formation in month_ends:
        formation = pd.Timestamp(formation)
        if not start_ts <= formation <= end_ts:
            continue
        position = int(dates.searchsorted(formation))
        if position < lookback - 1:
            continue
        next_period = formation.to_period("M") + 1
        if not (dates.to_period("M") == next_period).any():
            continue
        candidates.append(formation)
    return tuple(candidates)


class ParquetCIZCollection:
    """Read date-bounded frames from a verified year-partitioned collection."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        collection = self.root / "collection_manifest.json"
        if not collection.is_file():
            raise FileNotFoundError(f"missing collection manifest: {collection}")
        self.manifest_path = collection
        self.manifest_sha256 = sha256_file(collection)
        self.manifest = json.loads(collection.read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 2:
            raise ValueError(
                "Parquet collection requires manifest format_version=2"
            )
        spec_path = self.root / "dataset_spec.json"
        if not spec_path.is_file():
            raise FileNotFoundError(f"missing dataset spec: {spec_path}")
        expected_spec_file = self.manifest.get("dataset_spec_file_sha256")
        expected_spec = self.manifest.get("dataset_spec_sha256")
        if not expected_spec_file or sha256_file(spec_path) != expected_spec_file:
            raise RuntimeError("dataset-spec file hash mismatch")
        self.dataset_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        canonical_spec = json.dumps(
            self.dataset_spec,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        actual_spec = hashlib.sha256(canonical_spec).hexdigest()
        if not expected_spec or actual_spec != expected_spec:
            raise RuntimeError("dataset-spec canonical hash mismatch")
        records = self.manifest.get("years")
        if not isinstance(records, list) or not records:
            raise ValueError(f"collection manifest contains no years: {collection}")

        year_paths: dict[int, Path] = {}
        for record in records:
            year = int(record["year"])
            if year in year_paths:
                raise ValueError(f"duplicate year {year} in {collection}")
            path = _safe_child(self.root, str(record["relative_path"]))
            if not path.is_file():
                raise FileNotFoundError(f"missing Parquet year {year}: {path}")
            expected_bytes = record.get("bytes")
            if expected_bytes is None or path.stat().st_size != int(expected_bytes):
                raise RuntimeError(f"Parquet byte-size mismatch for {year}")
            expected = str(record.get("parquet_sha256", ""))
            if not expected or sha256_file(path) != expected:
                raise RuntimeError(f"Parquet collection hash mismatch for {year}")

            year_manifest_path = path.parent / "manifest.json"
            if not year_manifest_path.is_file():
                raise FileNotFoundError(
                    f"missing Parquet year manifest: {year_manifest_path}"
                )
            expected_year_manifest = record.get("year_manifest_sha256")
            if (
                not expected_year_manifest
                or sha256_file(year_manifest_path) != expected_year_manifest
            ):
                raise RuntimeError(
                    f"Parquet year-manifest hash mismatch for {year}"
                )
            year_manifest = json.loads(
                year_manifest_path.read_text(encoding="utf-8")
            )
            if int(year_manifest["year"]) != year:
                raise RuntimeError(f"Parquet year-manifest identity mismatch: {year}")
            if year_manifest.get("parquet_sha256") != expected:
                raise RuntimeError(
                    f"Parquet year and collection disagree on hash: {year}"
                )
            expected_source = record.get("source_manifest_sha256")
            if (
                not expected_source
                or year_manifest.get("source_manifest_sha256")
                != expected_source
            ):
                raise RuntimeError(
                    f"Parquet source-manifest lineage mismatch for {year}"
                )
            if year_manifest.get("dataset_spec_sha256") != expected_spec:
                raise RuntimeError(
                    f"Parquet dataset-spec lineage mismatch for {year}"
                )
            year_paths[year] = path
        self._year_paths = year_paths

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(sorted(self._year_paths))

    def load_years(self, years: Iterable[int]) -> pd.DataFrame:
        frames = []
        for year in sorted(set(int(value) for value in years)):
            try:
                path = self._year_paths[year]
            except KeyError as exc:
                raise KeyError(f"year {year} is not in the Parquet collection") from exc
            frames.append(pd.read_parquet(path))
        if not frames:
            raise ValueError("at least one year is required")
        return pd.concat(frames, ignore_index=True)

    def load_range(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            raise ValueError("end precedes start")
        frame = self.load_years(range(start_ts.year, end_ts.year + 1))
        dates = pd.to_datetime(frame["dlycaldt"]).dt.normalize()
        return frame.loc[dates.between(start_ts, end_ts)].copy()

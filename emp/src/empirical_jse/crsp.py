"""CRSP CIZ extraction and point-in-time rolling-panel utilities.

The module deliberately separates I/O from panel logic:

* SQL builders are static and parameterize every date value.
* :class:`ImmutableSnapshotBuilder` is the only raw-data writer.
* Universe and panel builders are pure with respect to their inputs.
* CIZ ``dlyret`` is used as supplied.  ``stkdelists.delret`` is never
  compounded onto it because CIZ already incorporates the terminal return.

No function in this module deletes or overwrites an existing file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import pandas as pd


CIZ_DAILY_COLUMNS: tuple[str, ...] = (
    "permno",
    "permco",
    "sharetype",
    "securitytype",
    "securitysubtype",
    "usincflg",
    "issuertype",
    "primaryexch",
    "conditionaltype",
    "tradingstatusflg",
    "dlycaldt",
    "dlydelflg",
    "dlyprc",
    "dlyprcflg",
    "dlycap",
    "dlycapflg",
    "dlyret",
    "dlyretmissflg",
    "dlyretdurflg",
    "dlydistretflg",
    "dlyvol",
)

COMMON_SHARE_FILTERS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "sharetype": frozenset({"NS"}),
        "securitytype": frozenset({"EQTY"}),
        "securitysubtype": frozenset({"COM"}),
        "usincflg": frozenset({"Y"}),
        "issuertype": frozenset({"ACOR", "CORP"}),
    }
)
PRIMARY_EXCHANGES: tuple[str, ...] = ("N", "A", "Q")
VALID_SINGLE_PERIOD_DURATIONS: frozenset[str] = frozenset(
    {"D1", "D2", "D3", "D4", "DU"}
)
INVALID_CAP_FLAGS: frozenset[str] = frozenset(
    {"AD", "DE", "MP", "MS", "NT"}
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RawSQLConnection(Protocol):
    """Minimal WRDS connection interface used by the extractor."""

    def raw_sql(
        self,
        sql: str,
        *,
        params: Mapping[str, Any],
        date_cols: Sequence[str],
        chunksize: int,
        return_iter: bool,
    ) -> pd.DataFrame | Iterable[pd.DataFrame]:
        """Execute a read-only parameterized query."""


class SnapshotError(RuntimeError):
    """Base class for immutable-snapshot failures."""


class MissingDelistingReturnError(ValueError):
    """Raised when a CIZ terminal row has no usable ``dlyret``."""


@dataclass(frozen=True)
class AnnualSlice:
    """A bounded annual interval with an exclusive stop date."""

    year: int
    start: date
    stop: date

    def __post_init__(self) -> None:
        if self.start >= self.stop:
            raise ValueError("AnnualSlice.start must be before stop")
        if self.start.year != self.year:
            raise ValueError("AnnualSlice.year must equal start.year")


@dataclass(frozen=True)
class QuerySpec:
    """A parameterized SQL statement and its parsing metadata."""

    sql: str
    params: Mapping[str, Any]
    date_columns: tuple[str, ...] = ("dlycaldt",)


@dataclass(frozen=True)
class WindowDates:
    """Formation, 63-day estimation, and next-calendar-month OOS dates."""

    formation: pd.Timestamp
    estimation: tuple[pd.Timestamp, ...]
    out_of_sample: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class EligibilityResult:
    """Point-in-time eligible company representatives at one formation date."""

    window: WindowDates
    representatives: pd.DataFrame
    eligible_permnos: tuple[int, ...]
    counts: Mapping[str, int]


@dataclass(frozen=True)
class DelistingCashResult:
    """OOS return matrix after terminal CIZ returns become cash sleeves."""

    returns: pd.DataFrame
    return_flags: pd.DataFrame
    terminal_dates: Mapping[int, pd.Timestamp]


@dataclass(frozen=True)
class ReturnPanel:
    """One rolling estimation/OOS panel."""

    window: WindowDates
    permnos: tuple[int, ...]
    raw_estimation: pd.DataFrame
    centered_estimation: pd.DataFrame
    out_of_sample: pd.DataFrame
    terminal_dates: Mapping[int, pd.Timestamp]
    out_of_sample_return_flags: pd.DataFrame | None = None


@dataclass(frozen=True)
class SnapshotVerification:
    """Result of verifying every file listed in a snapshot manifest."""

    ok: bool
    errors: tuple[str, ...]


def _as_date(value: date | datetime | str | pd.Timestamp) -> date:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Date cannot be missing")
    return timestamp.date()


def _as_timestamp(value: date | datetime | str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Date cannot be missing")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def iter_annual_slices(
    start_date: date | datetime | str | pd.Timestamp,
    end_date: date | datetime | str | pd.Timestamp,
) -> tuple[AnnualSlice, ...]:
    """Split an inclusive date range into calendar-year query intervals.

    Each returned interval is ``[start, stop)`` so adjacent slices cannot
    duplicate a row at the year boundary.
    """

    start = _as_date(start_date)
    end = _as_date(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    final_stop = end + timedelta(days=1)
    cursor = start
    slices: list[AnnualSlice] = []
    while cursor < final_stop:
        year_stop = date(cursor.year + 1, 1, 1)
        stop = min(year_stop, final_stop)
        slices.append(AnnualSlice(cursor.year, cursor, stop))
        cursor = stop
    return tuple(slices)


def _validated_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return identifier


def build_dsf_v2_query(
    period: AnnualSlice,
    *,
    schema: str = "crsp_a_stock",
    table: str = "dsf_v2",
    columns: Sequence[str] = CIZ_DAILY_COLUMNS,
    restrict_to_conventional_candidates: bool = True,
) -> QuerySpec:
    """Build the bounded, read-only CIZ daily query for one annual slice.

    Security-status filters are intentionally absent from the returned daily
    rows: filtering them to active securities would discard halt and terminal
    records needed after portfolio formation. To avoid an expensive correlated
    lookup, the default materializes conventional-common PERMNOs once from the
    compact security-history table over the current/prior-year protection
    window, then returns *all* daily rows for those PERMNOs in the requested
    slice. Exact point-in-time filters are still applied locally by
    :func:`build_eligible_universe`; the wider candidate window only protects
    lookback and OOS records at year boundaries.
    """

    safe_schema = _validated_identifier(schema)
    safe_table = _validated_identifier(table)
    safe_columns = tuple(_validated_identifier(column) for column in columns)
    if not safe_columns:
        raise ValueError("At least one column is required")
    if len(set(safe_columns)) != len(safe_columns):
        raise ValueError("Query columns must be unique")
    if "dlycaldt" not in safe_columns:
        raise ValueError("CIZ query must include dlycaldt")
    if "dlyret" not in safe_columns:
        raise ValueError("CIZ query must include dlyret")

    selected = ",\n    ".join(f"d.{column}" for column in safe_columns)
    cte = ""
    candidate_join = ""
    candidate_start = (
        pd.Timestamp(period.start) - pd.DateOffset(years=1)
    ).date()
    if restrict_to_conventional_candidates:
        cte = (
            "WITH common_permnos AS MATERIALIZED (\n"
            "    SELECT DISTINCT permno\n"
            f"    FROM {safe_schema}.stksecurityinfohist\n"
            "    WHERE secinfostartdt < %(stop_date)s\n"
            "      AND secinfoenddt >= %(candidate_start_date)s\n"
            "      AND sharetype = 'NS'\n"
            "      AND securitytype = 'EQTY'\n"
            "      AND securitysubtype = 'COM'\n"
            "      AND usincflg = 'Y'\n"
            "      AND issuertype IN ('ACOR', 'CORP')\n"
            "      AND primaryexch IN ('N', 'A', 'Q')\n"
            ")\n"
        )
        candidate_join = "\nJOIN common_permnos AS c ON c.permno = d.permno"
    sql = (
        f"{cte}SELECT\n"
        f"    {selected}\n"
        f"FROM {safe_schema}.{safe_table} AS d"
        f"{candidate_join}\n"
        "WHERE d.dlycaldt >= %(start_date)s\n"
        "  AND d.dlycaldt < %(stop_date)s"
        "\n"
    )
    return QuerySpec(
        sql=sql,
        params=MappingProxyType(
            {
                "candidate_start_date": candidate_start,
                "start_date": period.start,
                "stop_date": period.stop,
            }
        ),
    )


def _safe_relative_path(relative_path: str | PurePosixPath) -> PurePosixPath:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts:
        raise ValueError("Snapshot artifact path must be nonempty and relative")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe snapshot artifact path: {relative_path!r}")
    if relative.name == "manifest.json":
        raise ValueError("manifest.json is reserved for the snapshot manifest")
    return relative


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class ImmutableSnapshotBuilder:
    """Incrementally create a raw snapshot without overwriting any path.

    Artifacts are written into a uniquely named staging directory.  A complete
    manifest is added last, then the staging directory is atomically renamed
    to ``raw_root/snapshot_id``.  If an exception interrupts construction, the
    staging directory is deliberately retained for diagnosis; it is never
    deleted by this class.
    """

    def __init__(
        self,
        raw_root: str | Path,
        snapshot_id: str,
        *,
        created_at_utc: str | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", snapshot_id):
            raise ValueError("snapshot_id contains unsupported characters")

        self.raw_root = Path(raw_root)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.snapshot_id = snapshot_id
        self.final_path = self.raw_root / snapshot_id
        if self.final_path.exists():
            raise FileExistsError(
                f"Immutable snapshot already exists: {self.final_path}"
            )
        self.staging_path = (
            self.raw_root / f".{snapshot_id}.staging-{uuid4().hex}"
        )
        self.staging_path.mkdir(parents=False, exist_ok=False)
        self.created_at_utc = created_at_utc or datetime.now(
            timezone.utc
        ).isoformat()
        self._files: dict[str, dict[str, Any]] = {}
        self._closed = False

    def _destination(self, relative_path: str | PurePosixPath) -> tuple[str, Path]:
        if self._closed:
            raise SnapshotError("Snapshot builder has already been finalized")
        relative = _safe_relative_path(relative_path)
        key = relative.as_posix()
        if key in self._files:
            raise FileExistsError(f"Snapshot artifact already added: {key}")
        destination = self.staging_path.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Snapshot path already exists: {destination}")
        return key, destination

    def _record(self, key: str, destination: Path) -> dict[str, Any]:
        record = {
            "path": key,
            "bytes": destination.stat().st_size,
            "sha256": _sha256_file(destination),
        }
        self._files[key] = record
        return record

    def add_bytes(
        self,
        relative_path: str | PurePosixPath,
        content: bytes,
    ) -> Mapping[str, Any]:
        """Add one immutable byte artifact."""

        key, destination = self._destination(relative_path)
        with destination.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        return MappingProxyType(self._record(key, destination))

    def add_dataframe_chunks_csv_gz(
        self,
        relative_path: str | PurePosixPath,
        chunks: pd.DataFrame | Iterable[pd.DataFrame],
        *,
        expected_columns: Sequence[str] = CIZ_DAILY_COLUMNS,
        date_column: str = "dlycaldt",
    ) -> Mapping[str, Any]:
        """Stream DataFrame chunks into a deterministic gzip CSV artifact."""

        key, destination = self._destination(relative_path)
        expected = tuple(expected_columns)
        iterable: Iterable[pd.DataFrame]
        if isinstance(chunks, pd.DataFrame):
            iterable = (chunks,)
        else:
            iterable = chunks

        row_count = 0
        minimum_date: pd.Timestamp | None = None
        maximum_date: pd.Timestamp | None = None
        wrote_header = False

        with destination.open("xb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(
                    compressed,
                    encoding="utf-8",
                    newline="",
                ) as text:
                    for chunk in iterable:
                        if not isinstance(chunk, pd.DataFrame):
                            raise TypeError("Query chunks must be pandas DataFrames")
                        missing = [c for c in expected if c not in chunk.columns]
                        if missing:
                            raise ValueError(
                                f"CIZ query chunk is missing columns: {missing}"
                            )
                        ordered = chunk.loc[:, expected]
                        ordered.to_csv(
                            text,
                            index=False,
                            header=not wrote_header,
                            lineterminator="\n",
                            date_format="%Y-%m-%d",
                        )
                        wrote_header = True
                        row_count += len(ordered)
                        if len(ordered) and date_column in ordered.columns:
                            dates = pd.to_datetime(
                                ordered[date_column], errors="raise"
                            )
                            chunk_min = dates.min()
                            chunk_max = dates.max()
                            minimum_date = (
                                chunk_min
                                if minimum_date is None
                                else min(minimum_date, chunk_min)
                            )
                            maximum_date = (
                                chunk_max
                                if maximum_date is None
                                else max(maximum_date, chunk_max)
                            )
                    if not wrote_header:
                        pd.DataFrame(columns=expected).to_csv(
                            text,
                            index=False,
                            lineterminator="\n",
                        )

        record = self._record(key, destination)
        record.update(
            {
                "rows": row_count,
                "min_date": (
                    None if minimum_date is None else minimum_date.date().isoformat()
                ),
                "max_date": (
                    None if maximum_date is None else maximum_date.date().isoformat()
                ),
                "format": "csv.gz",
            }
        )
        return MappingProxyType(record)

    def add_binary_chunks(
        self,
        relative_path: str | PurePosixPath,
        chunks: Iterable[bytes],
        *,
        inspect: Callable[[Path], Mapping[str, Any]] | None = None,
    ) -> Mapping[str, Any]:
        """Stream an existing binary artifact into the immutable snapshot.

        ``inspect`` runs only after the file is fully flushed and can return
        format-specific metadata such as row counts or date bounds. Reserved
        manifest keys cannot be replaced by inspector output.
        """

        key, destination = self._destination(relative_path)
        with destination.open("xb") as output:
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError("Binary artifact chunks must be bytes")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        inspected = dict(inspect(destination)) if inspect is not None else {}
        reserved = {"path", "bytes", "sha256"}
        collisions = reserved.intersection(inspected)
        if collisions:
            raise ValueError(
                "Inspector metadata uses reserved keys: "
                + ", ".join(sorted(collisions))
            )
        record = self._record(key, destination)
        record.update(_json_safe(inspected))
        return MappingProxyType(record)

    def finalize(self, metadata: Mapping[str, Any]) -> Path:
        """Write the manifest and publish the immutable snapshot."""

        if self._closed:
            raise SnapshotError("Snapshot builder has already been finalized")
        if self.final_path.exists():
            raise FileExistsError(
                f"Immutable snapshot already exists: {self.final_path}"
            )

        manifest = {
            "format_version": 1,
            "snapshot_id": self.snapshot_id,
            "created_at_utc": self.created_at_utc,
            "metadata": _json_safe(metadata),
            "files": [self._files[key] for key in sorted(self._files)],
        }
        manifest_path = self.staging_path / "manifest.json"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(
                manifest,
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())

        self.staging_path.rename(self.final_path)
        self._closed = True
        return self.final_path


def verify_snapshot(snapshot_path: str | Path) -> SnapshotVerification:
    """Verify manifest membership, byte lengths, and SHA-256 hashes."""

    root = Path(snapshot_path)
    errors: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return SnapshotVerification(False, ("manifest.json is missing",))

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return SnapshotVerification(False, (f"manifest read failed: {exc}",))

    if manifest.get("snapshot_id") != root.name:
        errors.append("manifest snapshot_id does not match directory name")

    listed: set[str] = set()
    for record in manifest.get("files", []):
        try:
            relative = _safe_relative_path(record["path"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid manifest path: {exc}")
            continue
        key = relative.as_posix()
        listed.add(key)
        artifact = root.joinpath(*relative.parts)
        if not artifact.is_file():
            errors.append(f"missing artifact: {key}")
            continue
        size = artifact.stat().st_size
        if size != record.get("bytes"):
            errors.append(f"byte length mismatch: {key}")
        if _sha256_file(artifact) != record.get("sha256"):
            errors.append(f"sha256 mismatch: {key}")

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    for extra in sorted(actual - listed):
        errors.append(f"unlisted artifact: {extra}")
    for absent in sorted(listed - actual):
        if not any(error.endswith(absent) for error in errors):
            errors.append(f"listed artifact absent: {absent}")

    return SnapshotVerification(not errors, tuple(errors))


def extract_ciz_annual_snapshot(
    connection: RawSQLConnection,
    *,
    raw_root: str | Path,
    snapshot_id: str,
    start_date: date | datetime | str | pd.Timestamp,
    end_date: date | datetime | str | pd.Timestamp,
    schema: str = "crsp_a_stock",
    table: str = "dsf_v2",
    chunksize: int = 250_000,
    metadata: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
) -> Path:
    """Download bounded annual CIZ partitions into one immutable snapshot.

    This function performs no implicit full-history request: both dates are
    required and every SQL call is limited to one annual slice.
    """

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    builder = ImmutableSnapshotBuilder(
        raw_root,
        snapshot_id,
        created_at_utc=created_at_utc,
    )
    queries: list[dict[str, Any]] = []
    for period in iter_annual_slices(start_date, end_date):
        query = build_dsf_v2_query(
            period,
            schema=schema,
            table=table,
            restrict_to_conventional_candidates=True,
        )
        chunks = connection.raw_sql(
            query.sql,
            # psycopg2 accepts a real dict for ``%(name)s`` parameters but not
            # MappingProxyType, which QuerySpec uses to prevent mutation.
            params=dict(query.params),
            date_cols=query.date_columns,
            chunksize=chunksize,
            return_iter=True,
        )
        relative = f"daily/year={period.year}/dsf_v2.csv.gz"
        record = builder.add_dataframe_chunks_csv_gz(relative, chunks)
        queries.append(
            {
                "year": period.year,
                "sql": query.sql,
                "params": _json_safe(query.params),
                "artifact": relative,
                "rows": record["rows"],
                "min_date": record["min_date"],
                "max_date": record["max_date"],
            }
        )

    manifest_metadata = {
        "source": "WRDS CRSP CIZ",
        "schema": schema,
        "table": table,
        "date_range_inclusive": {
            "start": _as_date(start_date),
            "end": _as_date(end_date),
        },
        "columns": CIZ_DAILY_COLUMNS,
        "raw_query_security_restriction": (
            "current_or_prior_year_materialized_security_history_join"
        ),
        "chunksize": chunksize,
        "queries": queries,
        "user_metadata": dict(metadata or {}),
    }
    return builder.finalize(manifest_metadata)


def _trading_index(
    calendar: Sequence[Any] | pd.Series | pd.DataFrame | pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    if isinstance(calendar, pd.DataFrame):
        if "caldt" in calendar.columns:
            frame = calendar
            if "tradingflg" in frame.columns:
                frame = frame.loc[
                    frame["tradingflg"].astype("string").str.upper().eq("Y")
                ]
            values = frame["caldt"]
        elif "dlycaldt" in calendar.columns:
            values = calendar["dlycaldt"]
        else:
            raise ValueError("Calendar DataFrame needs caldt or dlycaldt")
    else:
        values = calendar

    index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise")).normalize()
    if index.hasnans:
        raise ValueError("Calendar contains missing dates")
    index = index.sort_values()
    if index.has_duplicates:
        raise ValueError("Calendar dates must be unique")
    return index


def formation_and_oos_dates(
    calendar: Sequence[Any] | pd.Series | pd.DataFrame | pd.DatetimeIndex,
    formation_date: date | datetime | str | pd.Timestamp,
    *,
    lookback: int = 63,
    require_month_end: bool = True,
) -> WindowDates:
    """Return the trailing window and all trading dates in the next month."""

    if lookback < 2:
        raise ValueError("lookback must be at least two")
    dates = _trading_index(calendar)
    formation = _as_timestamp(formation_date)
    positions = np.flatnonzero(dates == formation)
    if len(positions) != 1:
        raise ValueError("formation_date must occur exactly once in calendar")
    position = int(positions[0])
    if position + 1 < len(dates):
        next_trading = dates[position + 1]
        if (
            require_month_end
            and next_trading.year == formation.year
            and next_trading.month == formation.month
        ):
            raise ValueError("formation_date is not the final trading date of month")
    if position + 1 < lookback:
        raise ValueError("Calendar does not contain the full lookback window")

    estimation = tuple(dates[position - lookback + 1 : position + 1])
    next_month_start = (formation + pd.offsets.MonthBegin(1)).normalize()
    following_month_start = (
        next_month_start + pd.offsets.MonthBegin(1)
    ).normalize()
    oos = tuple(
        dates[
            (dates >= next_month_start) & (dates < following_month_start)
        ]
    )
    if not oos:
        raise ValueError("Calendar has no complete next-month OOS dates")
    return WindowDates(formation, estimation, oos)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def conventional_common_mask(
    frame: pd.DataFrame,
    *,
    exchanges: Sequence[str] = PRIMARY_EXCHANGES,
) -> pd.Series:
    """Point-in-time conventional U.S. common-share classification mask."""

    required = tuple(COMMON_SHARE_FILTERS) + ("primaryexch",)
    _require_columns(frame, required)
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for column, accepted in COMMON_SHARE_FILTERS.items():
        mask &= frame[column].astype("string").isin(accepted)
    mask &= frame["primaryexch"].astype("string").isin(tuple(exchanges))
    return mask


def formation_candidate_mask(
    frame: pd.DataFrame,
    *,
    exchanges: Sequence[str] = PRIMARY_EXCHANGES,
) -> pd.Series:
    """Tradable formation-date mask without using any future information."""

    _require_columns(
        frame,
        (
            "conditionaltype",
            "tradingstatusflg",
            "dlycap",
            "dlycapflg",
        ),
    )
    cap = pd.to_numeric(frame["dlycap"], errors="coerce")
    cap_flag = frame["dlycapflg"].astype("string")
    return (
        conventional_common_mask(frame, exchanges=exchanges)
        & frame["conditionaltype"].astype("string").eq("RW")
        & frame["tradingstatusflg"].astype("string").eq("A")
        & cap.notna()
        & np.isfinite(cap)
        & cap.gt(0)
        & cap_flag.notna()
        & ~cap_flag.isin(INVALID_CAP_FLAGS)
    )


def choose_permco_representatives(
    common_formation_rows: pd.DataFrame,
    eligible_permnos: Sequence[int],
) -> pd.DataFrame:
    """Aggregate PERMCO cap and retain its largest eligible PERMNO.

    Company capitalization is summed across all conventional common share
    classes at formation.  The representative is chosen only from securities
    that passed the historical-return eligibility checks.
    """

    _require_columns(
        common_formation_rows,
        ("permno", "permco", "dlycap", "dlycapflg"),
    )
    if common_formation_rows["permno"].duplicated().any():
        raise ValueError("Formation rows contain duplicate PERMNOs")
    if common_formation_rows["permco"].isna().any():
        raise ValueError("PERMCO is required for company-level selection")

    rows = common_formation_rows.copy()
    rows = rows.assign(
        permno=pd.to_numeric(rows["permno"], errors="raise").astype(int),
        permco=pd.to_numeric(rows["permco"], errors="raise").astype(int),
        _security_cap=pd.to_numeric(rows["dlycap"], errors="coerce"),
    )
    cap_flag = rows["dlycapflg"].astype("string")
    cap_rows = rows.loc[
        rows["_security_cap"].notna()
        & np.isfinite(rows["_security_cap"])
        & rows["_security_cap"].gt(0)
        & cap_flag.notna()
        & ~cap_flag.isin(INVALID_CAP_FLAGS)
    ]
    company_cap = cap_rows.groupby("permco", sort=False)[
        "_security_cap"
    ].sum(min_count=1)

    eligible = rows.loc[rows["permno"].isin(tuple(eligible_permnos))].copy()
    eligible = eligible.assign(
        company_cap=eligible["permco"].map(company_cap)
    )
    eligible = eligible.loc[
        eligible["company_cap"].notna() & eligible["company_cap"].gt(0)
    ]
    eligible = eligible.sort_values(
        ["permco", "_security_cap", "permno"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    representatives = eligible.drop_duplicates("permco", keep="first")
    representatives = representatives.sort_values(
        ["company_cap", "_security_cap", "permno"],
        ascending=[False, False, True],
        kind="mergesort",
    ).drop(columns="_security_cap")
    return representatives.reset_index(drop=True)


def build_eligible_universe(
    daily: pd.DataFrame,
    calendar: Sequence[Any] | pd.Series | pd.DataFrame | pd.DatetimeIndex,
    formation_date: date | datetime | str | pd.Timestamp,
    *,
    lookback: int = 63,
    exchanges: Sequence[str] = PRIMARY_EXCHANGES,
    allowed_return_durations: Sequence[str] = tuple(
        VALID_SINGLE_PERIOD_DURATIONS
    ),
) -> EligibilityResult:
    """Build a complete-history, point-in-time, company-level universe."""

    required = (
        "permno",
        "permco",
        "dlycaldt",
        "dlyret",
        "dlyretmissflg",
        "dlyretdurflg",
        "dlycap",
        "dlycapflg",
        "conditionaltype",
        "tradingstatusflg",
        "sharetype",
        "securitytype",
        "securitysubtype",
        "usincflg",
        "issuertype",
        "primaryexch",
    )
    _require_columns(daily, required)
    window = formation_and_oos_dates(
        calendar,
        formation_date,
        lookback=lookback,
    )

    frame = daily.copy()
    frame = frame.assign(
        dlycaldt=pd.to_datetime(
            frame["dlycaldt"],
            errors="raise",
        ).dt.normalize(),
        permno=pd.to_numeric(
            frame["permno"],
            errors="raise",
        ).astype(int),
    )

    # This is the explicit no-lookahead barrier.
    past = frame.loc[frame["dlycaldt"].le(window.formation)].copy()
    formation_rows = past.loc[
        past["dlycaldt"].eq(window.formation)
    ].copy()
    if formation_rows["permno"].duplicated().any():
        raise ValueError("Duplicate PERMNO rows at formation date")

    common_rows = formation_rows.loc[
        conventional_common_mask(formation_rows, exchanges=exchanges)
    ].copy()
    candidate_rows = formation_rows.loc[
        formation_candidate_mask(formation_rows, exchanges=exchanges)
    ].copy()
    candidate_permnos = tuple(candidate_rows["permno"].astype(int))

    history = past.loc[
        past["permno"].isin(candidate_permnos)
        & past["dlycaldt"].isin(window.estimation)
    ].copy()
    if history.duplicated(["permno", "dlycaldt"]).any():
        raise ValueError("Duplicate PERMNO-date rows in estimation window")

    history = history.assign(
        _return=pd.to_numeric(
            history["dlyret"],
            errors="coerce",
        )
    )
    history = history.assign(
        _valid=(
            history["_return"].notna()
            & np.isfinite(history["_return"])
            & history["dlyretmissflg"].astype("string").eq("NA")
            & history["dlyretdurflg"]
            .astype("string")
            .isin(tuple(allowed_return_durations))
        )
    )

    grouped = history.groupby("permno", sort=False)
    stats = grouped.agg(
        rows=("dlycaldt", "size"),
        unique_dates=("dlycaldt", "nunique"),
        all_valid=("_valid", "all"),
        return_variance=("_return", "var"),
    ).reindex(candidate_permnos)

    complete = (
        stats["rows"].eq(lookback)
        & stats["unique_dates"].eq(lookback)
        & stats["all_valid"].fillna(False)
    )
    nonzero = stats["return_variance"].gt(0) & np.isfinite(
        stats["return_variance"]
    )
    eligible = tuple(
        int(value) for value in stats.index[complete & nonzero].tolist()
    )
    representatives = choose_permco_representatives(common_rows, eligible)

    counts = MappingProxyType(
        {
            "formation_rows": len(formation_rows),
            "common_share_rows": len(common_rows),
            "tradable_candidates": len(candidate_rows),
            "complete_history": int(complete.sum()),
            "nonzero_variance": int((complete & nonzero).sum()),
            "eligible_permnos": len(eligible),
            "representative_permcos": len(representatives),
        }
    )
    return EligibilityResult(
        window=window,
        representatives=representatives,
        eligible_permnos=eligible,
        counts=counts,
    )


def select_top_cap(
    representatives: pd.DataFrame,
    n: int,
) -> tuple[int, ...]:
    """Select the top-N representatives with deterministic tie breaking."""

    _require_columns(representatives, ("permno", "company_cap"))
    if n <= 0:
        raise ValueError("n must be positive")
    if len(representatives) < n:
        raise ValueError(
            f"Only {len(representatives)} representatives are available for N={n}"
        )
    ordered = representatives.sort_values(
        ["company_cap", "permno"],
        ascending=[False, True],
        kind="mergesort",
    )
    return tuple(ordered["permno"].head(n).astype(int))


def nested_random_samples(
    representatives: pd.DataFrame,
    sizes: Sequence[int],
    *,
    seed: int | Sequence[int],
) -> Mapping[int, tuple[int, ...]]:
    """Return nested random samples from one seeded permutation."""

    _require_columns(representatives, ("permno",))
    normalized_sizes = tuple(sorted({int(size) for size in sizes}))
    if not normalized_sizes or normalized_sizes[0] <= 0:
        raise ValueError("sizes must contain positive integers")
    if normalized_sizes[-1] > len(representatives):
        raise ValueError("Largest requested sample exceeds representatives")

    # Canonicalize the input before applying the seeded permutation.  This
    # makes the draw invariant to incidental DataFrame row order while
    # preserving a uniform sample without replacement.
    permnos = np.sort(
        representatives["permno"].astype(int).to_numpy(copy=True)
    )
    permutation = np.random.default_rng(seed).permutation(permnos)
    return MappingProxyType(
        {
            size: tuple(int(value) for value in permutation[:size])
            for size in normalized_sizes
        }
    )


def select_nested_random_top_cap(
    representatives: pd.DataFrame,
    sizes: Sequence[int],
    *,
    outer_n: int,
    seed: int | Sequence[int],
) -> Mapping[int, tuple[int, ...]]:
    """Draw nested samples only after freezing the top-cap outer universe."""

    normalized_sizes = tuple(sorted({int(size) for size in sizes}))
    if not normalized_sizes or normalized_sizes[0] <= 0:
        raise ValueError("sizes must contain positive integers")
    if outer_n <= 0:
        raise ValueError("outer_n must be positive")
    if normalized_sizes[-1] != outer_n:
        raise ValueError("largest requested size must equal outer_n")

    outer_permnos = select_top_cap(representatives, outer_n)
    outer = (
        representatives.set_index("permno", drop=False)
        .loc[list(outer_permnos)]
        .reset_index(drop=True)
    )
    samples = nested_random_samples(outer, normalized_sizes, seed=seed)
    if set(samples[outer_n]) != set(outer_permnos):
        raise RuntimeError("random outer sample does not equal top-cap universe")
    for smaller, larger in zip(normalized_sizes, normalized_sizes[1:]):
        if not set(samples[smaller]).issubset(samples[larger]):
            raise RuntimeError("random samples are not nested")
    return samples


def apply_delisting_cash_policy(
    oos_daily: pd.DataFrame,
    oos_dates: Sequence[Any],
    permnos: Sequence[int],
) -> DelistingCashResult:
    """Use the CIZ terminal return once, then set that sleeve to cash.

    ``dlyret`` is never combined with a legacy or event-table ``delret``.
    Missing nonterminal observations are preserved as missing for an explicit
    downstream policy; only dates strictly after a valid terminal return are
    changed to zero.
    """

    _require_columns(
        oos_daily,
        (
            "permno",
            "dlycaldt",
            "dlyret",
            "dlyretmissflg",
            "dlydelflg",
        ),
    )
    frame = oos_daily.copy()
    frame = frame.assign(
        dlycaldt=pd.to_datetime(
            frame["dlycaldt"],
            errors="raise",
        ).dt.normalize(),
        permno=pd.to_numeric(
            frame["permno"],
            errors="raise",
        ).astype(int),
    )
    if frame.duplicated(["permno", "dlycaldt"]).any():
        raise ValueError("Duplicate PERMNO-date rows in OOS data")

    dates = pd.DatetimeIndex(pd.to_datetime(oos_dates, errors="raise")).normalize()
    if dates.has_duplicates:
        raise ValueError("OOS dates must be unique")
    dates = dates.sort_values()
    ordered_permnos = tuple(int(value) for value in permnos)

    selected = frame.loc[
        frame["permno"].isin(ordered_permnos)
        & frame["dlycaldt"].isin(dates)
    ]
    returns = selected.pivot(
        index="dlycaldt",
        columns="permno",
        values="dlyret",
    ).reindex(index=dates, columns=ordered_permnos)
    returns = returns.apply(pd.to_numeric, errors="coerce").astype(float)
    return_flags = selected.pivot(
        index="dlycaldt",
        columns="permno",
        values="dlyretmissflg",
    ).reindex(index=dates, columns=ordered_permnos).astype("string")
    # A numeric value accompanied by an RM flag such as MV is not a complete
    # CIZ return.  Treat it as missing; for a terminal row this is a fatal
    # primary-sample event rather than a zero-return imputation.
    incomplete_return = ~return_flags.eq("NA")
    returns = returns.mask(incomplete_return)
    flags = selected.pivot(
        index="dlycaldt",
        columns="permno",
        values="dlydelflg",
    ).reindex(index=dates, columns=ordered_permnos)

    terminal_dates: dict[int, pd.Timestamp] = {}
    for permno in ordered_permnos:
        terminal_mask = flags[permno].astype("string").eq("Y")
        terminal = dates[
            terminal_mask.to_numpy(dtype=bool, na_value=False)
        ]
        if len(terminal) > 1:
            raise ValueError(f"Multiple CIZ terminal rows for PERMNO {permno}")
        if not len(terminal):
            continue
        terminal_date = pd.Timestamp(terminal[0])
        terminal_return = returns.at[terminal_date, permno]
        if pd.isna(terminal_return) or not np.isfinite(terminal_return):
            missing_reason = return_flags.at[terminal_date, permno]
            raise MissingDelistingReturnError(
                f"PERMNO {permno} has an incomplete CIZ terminal dlyret "
                f"(dlyretmissflg={missing_reason!r})"
            )
        terminal_dates[permno] = terminal_date
        returns.loc[returns.index > terminal_date, permno] = 0.0
        return_flags.loc[return_flags.index > terminal_date, permno] = "CASH"

    return DelistingCashResult(
        returns=returns,
        return_flags=return_flags,
        terminal_dates=MappingProxyType(terminal_dates),
    )


def build_return_panel(
    daily: pd.DataFrame,
    window: WindowDates,
    permnos: Sequence[int],
) -> ReturnPanel:
    """Build a complete 63-day panel and next-month CIZ OOS matrix."""

    estimation, centered = build_estimation_panel(daily, window, permnos)
    ordered_permnos = tuple(int(value) for value in permnos)

    _require_columns(
        daily,
        ("permno", "dlycaldt", "dlyret", "dlydelflg"),
    )

    frame = daily.copy()
    frame = frame.assign(
        dlycaldt=pd.to_datetime(
            frame["dlycaldt"],
            errors="raise",
        ).dt.normalize(),
        permno=pd.to_numeric(
            frame["permno"],
            errors="raise",
        ).astype(int),
    )
    relevant = frame.loc[
        frame["permno"].isin(ordered_permnos)
        & frame["dlycaldt"].isin(
            tuple(window.estimation) + tuple(window.out_of_sample)
        )
    ]
    if relevant.duplicated(["permno", "dlycaldt"]).any():
        raise ValueError("Duplicate PERMNO-date rows in requested panel")

    oos_long = relevant.loc[
        relevant["dlycaldt"].isin(window.out_of_sample)
    ]
    cash = apply_delisting_cash_policy(
        oos_long,
        window.out_of_sample,
        ordered_permnos,
    )
    return ReturnPanel(
        window=window,
        permnos=ordered_permnos,
        raw_estimation=estimation,
        centered_estimation=centered,
        out_of_sample=cash.returns,
        terminal_dates=cash.terminal_dates,
        out_of_sample_return_flags=cash.return_flags,
    )


def build_estimation_panel(
    daily: pd.DataFrame,
    window: WindowDates,
    permnos: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the finite formation matrix without inspecting OOS returns.

    This deliberately isolates the 63-day information set used by the Jung
    factor-count procedure.  In particular, an incomplete terminal return in
    the following month cannot condition whether a formation-month factor
    count is computed or retained.
    """

    _require_columns(daily, ("permno", "dlycaldt", "dlyret"))
    ordered_permnos = tuple(int(value) for value in permnos)
    if not ordered_permnos or len(set(ordered_permnos)) != len(ordered_permnos):
        raise ValueError("permnos must be a nonempty unique sequence")

    frame = daily.loc[
        :,
        ["permno", "dlycaldt", "dlyret"],
    ].copy()
    frame = frame.assign(
        dlycaldt=pd.to_datetime(
            frame["dlycaldt"],
            errors="raise",
        ).dt.normalize(),
        permno=pd.to_numeric(
            frame["permno"],
            errors="raise",
        ).astype(int),
    )
    relevant = frame.loc[
        frame["permno"].isin(ordered_permnos)
        & frame["dlycaldt"].isin(window.estimation)
    ]
    if relevant.duplicated(["permno", "dlycaldt"]).any():
        raise ValueError("Duplicate PERMNO-date rows in estimation panel")
    estimation = relevant.pivot(
        index="dlycaldt",
        columns="permno",
        values="dlyret",
    ).reindex(
        index=pd.DatetimeIndex(window.estimation),
        columns=ordered_permnos,
    )
    estimation = estimation.apply(pd.to_numeric, errors="coerce").astype(float)
    if estimation.isna().any().any() or not np.isfinite(
        estimation.to_numpy()
    ).all():
        raise ValueError("Estimation panel must have finite 63/63 returns")
    if (estimation.var(axis=0, ddof=1) <= 0).any():
        raise ValueError("Estimation panel contains a zero-variance security")
    centered = estimation - estimation.mean(axis=0)
    return estimation, centered


__all__ = [
    "AnnualSlice",
    "CIZ_DAILY_COLUMNS",
    "COMMON_SHARE_FILTERS",
    "DelistingCashResult",
    "EligibilityResult",
    "ImmutableSnapshotBuilder",
    "INVALID_CAP_FLAGS",
    "MissingDelistingReturnError",
    "PRIMARY_EXCHANGES",
    "QuerySpec",
    "ReturnPanel",
    "SnapshotError",
    "SnapshotVerification",
    "VALID_SINGLE_PERIOD_DURATIONS",
    "WindowDates",
    "apply_delisting_cash_policy",
    "build_dsf_v2_query",
    "build_eligible_universe",
    "build_estimation_panel",
    "build_return_panel",
    "choose_permco_representatives",
    "conventional_common_mask",
    "extract_ciz_annual_snapshot",
    "formation_and_oos_dates",
    "formation_candidate_mask",
    "iter_annual_slices",
    "nested_random_samples",
    "select_nested_random_top_cap",
    "select_top_cap",
    "verify_snapshot",
]

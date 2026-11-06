"""Download small CRSP CIZ reference/audit tables as one immutable snapshot."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.empirical_jse.crsp import ImmutableSnapshotBuilder, verify_snapshot
from wrds_client import connect


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data" / "raw" / "wrds" / "crsp_ciz_reference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root if args.raw_root.is_absolute() else ROOT / args.raw_root
    builder = ImmutableSnapshotBuilder(raw_root, args.snapshot_id)
    bounds = {"start_date": args.start, "end_date": args.end}
    queries = {
        "calendar/calendar.csv.gz": (
            """
            select caldt, tradingflg, holidayflg, weekendflg,
                   datestatusflg, weekday, exchangegroup
            from crsp_a_stock.metaexchangecalendar
            where caldt >= %(start_date)s and caldt <= %(end_date)s
            order by caldt, exchangegroup
            """,
            ("caldt", "tradingflg", "holidayflg", "weekendflg",
             "datestatusflg", "weekday", "exchangegroup"),
            "caldt",
        ),
        "delists/stkdelists.csv.gz": (
            """
            select permno, delistingdt, deldtprc, deldtprcflg,
                   delactiontype, delstatustype, delreasontype,
                   delpaymenttype, delpermno, delpermco, delret,
                   delretmisstype, delnextdt, delnextprc, delnextprcflg,
                   delamtdt, deldivamt, deldistype, deldlydt
            from crsp_a_stock.stkdelists
            where delistingdt >= %(start_date)s and delistingdt <= %(end_date)s
            order by delistingdt, permno
            """,
            (
                "permno", "delistingdt", "deldtprc", "deldtprcflg",
                "delactiontype", "delstatustype", "delreasontype",
                "delpaymenttype", "delpermno", "delpermco", "delret",
                "delretmisstype", "delnextdt", "delnextprc", "delnextprcflg",
                "delamtdt", "deldivamt", "deldistype", "deldlydt",
            ),
            "delistingdt",
        ),
        "market/daily_index.csv.gz": (
            """
            select dlycaldt, vwretd, vwretx, vwtotval, vwusdval,
                   vwtotcnt, vwusdcnt, ewretd, ewretx, ewtotval,
                   ewusdval, ewtotcnt, ewusdcnt, sprtrn, spindx
            from crsp_a_stock.wrds_dailyindexret_query
            where dlycaldt >= %(start_date)s and dlycaldt <= %(end_date)s
            order by dlycaldt
            """,
            (
                "dlycaldt", "vwretd", "vwretx", "vwtotval", "vwusdval",
                "vwtotcnt", "vwusdcnt", "ewretd", "ewretx", "ewtotval",
                "ewusdval", "ewtotcnt", "ewusdcnt", "sprtrn", "spindx",
            ),
            "dlycaldt",
        ),
    }
    query_records = []
    db = connect()
    try:
        flags = db.raw_sql(
            """
            select flagtype, flagvalue, flagtypedesc, flagdesc,
                   flagdef, flagcoverageflg, flagkey
            from crsp_a_stock.metaflaginfo
            order by flagtype, flagvalue
            """
        )
        flag_record = builder.add_dataframe_chunks_csv_gz(
            "flags/metaflaginfo.csv.gz",
            flags,
            expected_columns=(
                "flagtype", "flagvalue", "flagtypedesc", "flagdesc",
                "flagdef", "flagcoverageflg", "flagkey",
            ),
            date_column="",
        )
        query_records.append(
            {"artifact": "flags/metaflaginfo.csv.gz", "rows": flag_record["rows"]}
        )

        for artifact, (sql, columns, date_column) in queries.items():
            frame = db.raw_sql(
                sql,
                params=bounds,
                date_cols=[date_column],
            )
            record = builder.add_dataframe_chunks_csv_gz(
                artifact,
                frame,
                expected_columns=columns,
                date_column=date_column,
            )
            query_records.append(
                {
                    "artifact": artifact,
                    "rows": record["rows"],
                    "sql": sql,
                    "params": {
                        "start_date": args.start.isoformat(),
                        "end_date": args.end.isoformat(),
                    },
                }
            )
    finally:
        db.close()

    snapshot = builder.finalize(
        {
            "source": "WRDS CRSP CIZ reference tables",
            "date_range_inclusive": {
                "start": args.start,
                "end": args.end,
            },
            "queries": query_records,
        }
    )
    verification = verify_snapshot(snapshot)
    if not verification.ok:
        raise RuntimeError("; ".join(verification.errors))
    print(f"Verified immutable reference snapshot: {snapshot}")


if __name__ == "__main__":
    main()

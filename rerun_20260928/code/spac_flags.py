"""SPAC identification for the 28 September 2026 rerun.

A security (PERMNO) is treated as a SPAC on date t when, on the CRSP CIZ security-information
row valid at t, either

  (A) its issuer name has the form of a SPAC name (NAME_PATTERN: "... ACQUISITION CORP",
      "... ACQ CORP", "... MERGER", "BLANK CHECK", "SPECIAL PURPOSE", ...), except for the
      names in NOT_SPAC, which were checked by hand; or
  (B) all of the following hold:
      (B1) the security is linked to a SPAC: its 6-digit issuer CUSIP (on any of its CIZ rows)
           matches an SDC New Issues IPO whose business description is a blank-check or
           special-purpose-acquisition company (SDC's `cusip` or `cusip9`), or CRSP records SIC
           6770 (blank checks) for it on some row;
      (B2) it started trading as a SPAC: its first CIZ row has SIC 6770 or a SPAC-form name,
           or its first trading date is between 30 days before and 120 days after the date of
           an SDC SPAC IPO with a matching CUSIP (SPAC shares start trading separately from
           the units about 52 days after the IPO);
      (B3) its issuer name at t still begins with the same word as the name under which it was
           first listed (the merger with the target, the "de-SPAC", almost always renames the
           company; comparing the first word ignores cosmetic changes such as an added "DE"); and
      (B4) if the name never changes, t is within MAX_LIFE of the first trading date (SPAC
           charters require a merger or liquidation within 2-3 years, sometimes extended), so the few SPACs that
           kept their name after the merger are not flagged for ever.

CRSP's SIC and the SPAC's CUSIP sometimes survive the merger (Jamba, Global Eagle
Entertainment) and some CRSP securities begin only after the merger (RAM Energy Resources,
Rand Logistics), which is why no single source is used on its own.  Compustat's historical SIC
(9995 for SPACs) was tried as a further link and rejected: every security it alone linked was
an operating company (e.g. Primerica, Getty Images), because Compustat keeps the SPAC's GVKEY
for the merged company.
Inputs: emp/data/raw/wrds/crsp_names_20260928/ (written by r01_fetch_names.py and
r01d_fetch_sdc_ipos.py).
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NAMES = ROOT / "emp" / "data" / "raw" / "wrds" / "crsp_names_20260928"
NAME_PATTERN = re.compile(
    r"\bACQUISITIONS?\s+(CORP|CO|COMPANY|CORPORATION|HOLDINGS|HLDGS|LTD|LIMITED|INC|GROUP|GRP)\b"
    r"|\bACQ\b|\bACQUIRCO\b|\bMERGER\b|\bBLANK\s+CHECK\b|\bSPAC\b|\bSPECIAL\s+PURPOSE\b|\bSPEC\s+PUR\b")
# checked by hand: SPAC-form names of companies that are not SPACs
NOT_SPAC = {
    "CAESARS ACQUISITION CO",   # holding company of Caesars Growth Partners (2013-2017)
}
SDC_BUSINESS = r"blank check|special purpose acq|\bspac\b"
IPO_WINDOW = (pd.Timedelta(days=-30), pd.Timedelta(days=120))   # first trade - IPO date
MAX_LIFE = pd.Timedelta(days=4 * 365 + 1)


def _sdc() -> pd.DataFrame:
    sd = pd.read_parquet(NAMES / "sdc_spac_ipos.parquet")
    sd = sd[sd.businesss.str.contains(SDC_BUSINESS, case=False, na=False, regex=True)].copy()
    ipo = pd.to_datetime(sd.ipodate).fillna(pd.to_datetime(sd.master_deal_date))
    rows = []
    for col in ("cusip", "cusip9"):
        k = sd[col].astype("string").str.upper().str.strip().str[:6]
        rows.append(pd.DataFrame({"cusip6": k.to_numpy(), "ipodate": ipo.to_numpy()}))
    out = pd.concat(rows).dropna(subset=["cusip6"])
    return out[out.cusip6.str.fullmatch(r"[0-9A-Z]{6}")].drop_duplicates()


def _first_word(name: str) -> str:
    """First word of an issuer name, with CRSP's spaced initials glued ("G T Y" -> "GTY")."""
    s = re.sub(r"[^A-Z0-9 ]", " ", name.upper())
    s = re.sub(r"\b([A-Z0-9]) (?=[A-Z0-9]\b)", r"\1", s)
    w = s.split()
    return w[0] if w else ""


def security_history() -> pd.DataFrame:
    """CIZ security-information rows with the SPAC indicators above."""
    c = pd.read_parquet(NAMES / "stksecurityinfohist.parquet")
    c = c[["permno", "secinfostartdt", "secinfoenddt", "securitybegdt", "cusip", "issuernm", "siccd"]].copy()
    c["permno"] = c.permno.astype(int)
    c["start"] = pd.to_datetime(c.secinfostartdt); c["end"] = pd.to_datetime(c.secinfoenddt)
    c["begdt"] = pd.to_datetime(c.securitybegdt)
    c["siccd"] = pd.to_numeric(c.siccd, errors="coerce")
    c["issuernm"] = (c.issuernm.fillna("").str.upper()
                     .str.replace(r"\s*\(LAST KNOWN\)", "", regex=True).str.strip())
    c["cusip6"] = c.cusip.astype("string").str.upper().str[:6]
    c = c.sort_values(["permno", "start"]).reset_index(drop=True)
    c["name_pattern"] = c.issuernm.map(lambda s: bool(NAME_PATTERN.search(s))) & ~c.issuernm.isin(NOT_SPAC)

    # (B1) links
    sdc_hit = c[["permno", "cusip6"]].dropna().drop_duplicates().merge(_sdc(), on="cusip6")
    c["link_sdc"] = c.permno.isin(set(sdc_hit.permno))
    c["link_sic"] = c.permno.isin(set(c.loc[c.siccd.eq(6770), "permno"]))
    c["linked"] = c.link_sdc | c.link_sic

    # (B2) started trading as a SPAC
    first = c.groupby("permno").first()
    beg = first.begdt.fillna(first.start)
    c["first_trade"] = c.permno.map(beg)
    s = sdc_hit.assign(gap=sdc_hit.permno.map(beg) - sdc_hit.ipodate)
    s = s[(s.gap >= IPO_WINDOW[0]) & (s.gap <= IPO_WINDOW[1])]
    began = set(s.permno) | set(first.index[first.siccd.eq(6770) | first.name_pattern])
    c["began_as_spac"] = c.permno.isin(began)

    # (B3), (B4)
    c["first_word"] = c.issuernm.map(_first_word)
    c["listing_name"] = c.first_word.eq(c.permno.map(c.groupby("permno").first_word.first()))
    c["name_ever_changes"] = c.permno.map(c.groupby("permno").issuernm.nunique() > 1)
    c["spac_A"] = c.name_pattern
    c["spac_B"] = c.linked & c.began_as_spac & c.listing_name
    return c


def spac_permnos_at(hist: pd.DataFrame, date) -> set[int]:
    d = pd.Timestamp(date)
    h = hist[(hist.start <= d) & (hist.end >= d)]
    b4 = h.name_ever_changes | (d - h.first_trade <= MAX_LIFE)
    return {int(p) for p in h.loc[h.spac_A | (h.spac_B & b4), "permno"]}

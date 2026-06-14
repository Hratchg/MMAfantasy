"""Parse UFCStats.com fight detail page into FightDetailPage model.

Uses BeautifulSoup4 with lxml backend and CSS selectors per D-04.
Cardinality checks raise ValueError on structural changes per D-10/D-11.
"""

from __future__ import annotations

import logging

from bs4 import BeautifulSoup, Tag

from ufc_prediction.scraper.models import (
    FightDetailPage,
    RoundStatsRaw,
    SigStrikesRaw,
)

logger = logging.getLogger(__name__)


# ── "X of Y" arithmetic helpers ────────────────────────────────────────────


def _parse_x_of_y_raw(val: str) -> tuple[int, int]:
    """Parse a raw 'X of Y' string into (landed, attempted) integers."""
    parts = val.strip().split(" of ")
    if len(parts) == 2:
        try:
            return int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            return 0, 0
    return 0, 0


def _format_x_of_y(landed: int, attempted: int) -> str:
    """Format (landed, attempted) back to 'X of Y' string."""
    return f"{landed} of {attempted}"


def _format_pct(landed: int, attempted: int) -> str:
    """Compute percentage from landed/attempted and format as 'N%'."""
    if attempted == 0:
        return "0%"
    return f"{round(landed / attempted * 100)}%"


# ── Main parser ────────────────────────────────────────────────────────────


def parse_fight_detail(html: str) -> FightDetailPage:
    """Parse fight detail HTML into a FightDetailPage model.

    Args:
        html: Raw HTML from the fight detail page.

    Returns:
        FightDetailPage with fighter metadata, totals, sig strikes,
        and per-round breakdowns.

    Raises:
        ValueError: If cardinality checks fail (wrong number of persons,
            columns, or stat rows).
    """
    soup = BeautifulSoup(html, "lxml")

    # -- Fighter metadata --
    persons = soup.select("div.b-fight-details__person")
    if len(persons) != 2:
        msg = f"Fight detail cardinality check failed: expected 2 person divs, got {len(persons)}"
        raise ValueError(msg)

    fighter_a_status = _extract_status(persons[0])
    fighter_a_name, fighter_a_url = _extract_person(persons[0])
    fighter_b_status = _extract_status(persons[1])
    fighter_b_name, fighter_b_url = _extract_person(persons[1])

    # -- Bout type --
    bout_el = soup.select_one("i.b-fight-details__fight-title")
    bout_type = bout_el.get_text(strip=True) if bout_el else ""

    # -- Method, round, time, time_format, referee, method_detail --
    method, method_detail, round_finished, time_finished, time_format, referee = (
        _extract_fight_metadata(soup)
    )

    # -- Stat tables --
    # Live UFCStats pages put header labels ("Totals", "Significant Strikes")
    # and data tables in separate <section> elements. We find sections that
    # actually contain a <table> with data rows instead of relying on indices.
    sections = soup.select("section.b-fight-details__section")
    data_sections = [
        s for s in sections if s.select_one("tbody tr.b-fight-details__table-row") is not None
    ]

    if len(data_sections) < 1:
        msg = (
            f"Fight detail cardinality check failed: "
            f"expected >=1 data sections, got {len(data_sections)}"
        )
        raise ValueError(msg)

    # Classify data sections by column count:
    # - 10 columns = totals/per-round totals (Fighter, KD, Sig.str, Sig.str%,
    #   Total str, Td, Td%, Sub.att, Rev, Ctrl)
    # - 9 columns = sig strikes (Fighter, Sig.str, Sig.str%, Head, Body, Leg,
    #   Distance, Clinch, Ground)
    totals_sections = []
    sig_sections = []
    for sec in data_sections:
        row = sec.select_one("tbody tr.b-fight-details__table-row")
        if row is None:
            continue
        cols = row.select("td.b-fight-details__table-col")
        if len(cols) >= 10:
            totals_sections.append(sec)
        else:
            sig_sections.append(sec)

    if not totals_sections:
        msg = "Fight detail cardinality check failed: no totals data sections found"
        raise ValueError(msg)

    # First totals section = fight-level totals (single row)
    totals = _parse_totals_row(totals_sections[0])

    # Per-round totals: remaining totals sections, extracting ALL rows
    per_round_totals: list[tuple[RoundStatsRaw, RoundStatsRaw]] = []
    for sec in totals_sections[1:]:
        per_round_totals.extend(_parse_all_totals_rows(sec))

    # Per-round sig strikes: ALL sig sections, extracting ALL rows
    per_round_sig_strikes: list[tuple[SigStrikesRaw, SigStrikesRaw]] = []
    for sec in sig_sections:
        per_round_sig_strikes.extend(_parse_all_sig_strikes_rows(sec))

    logger.debug(
        "Parsed %d per-round totals, %d per-round sig strikes",
        len(per_round_totals),
        len(per_round_sig_strikes),
    )
    if len(per_round_totals) != len(per_round_sig_strikes):
        logger.warning(
            "per_round_totals (%d) and per_round_sig_strikes (%d) lengths differ",
            len(per_round_totals),
            len(per_round_sig_strikes),
        )

    # Fight-level sig strikes: derive from per-round data (sum all rounds).
    # In live HTML there is NO separate fight-level sig strikes section --
    # the only 9-col section contains per-round breakdowns.
    if per_round_sig_strikes:
        sig_strikes = _sum_sig_strikes(per_round_sig_strikes)
    else:
        _empty = SigStrikesRaw(
            fighter_name="",
            sig_str="0 of 0",
            sig_str_pct="0%",
            head="0 of 0",
            body="0 of 0",
            leg="0 of 0",
            distance="0 of 0",
            clinch="0 of 0",
            ground="0 of 0",
        )
        sig_strikes = (_empty, _empty)

    return FightDetailPage(
        fighter_a_name=fighter_a_name,
        fighter_a_url=fighter_a_url,
        fighter_b_name=fighter_b_name,
        fighter_b_url=fighter_b_url,
        fighter_a_status=fighter_a_status,
        fighter_b_status=fighter_b_status,
        bout_type=bout_type,
        method=method,
        method_detail=method_detail,
        round_finished=round_finished,
        time_finished=time_finished,
        time_format=time_format,
        referee=referee,
        totals=totals,
        sig_strikes=sig_strikes,
        per_round_totals=per_round_totals,
        per_round_sig_strikes=per_round_sig_strikes,
    )


# ── Metadata extraction ───────────────────────────────────────────────────


def _extract_status(person_div: Tag) -> str:
    """Extract W/L/D/NC status from a person div."""
    status_el = person_div.select_one("i.b-fight-details__person-status")
    return status_el.get_text(strip=True) if status_el else ""


def _extract_person(person_div: Tag) -> tuple[str, str]:
    """Extract fighter name and URL from a person div."""
    link = person_div.select_one("a.b-fight-details__person-link")
    if link is None:
        return ("", "")
    name = link.get_text(strip=True)
    url_attr = link.get("href", "")
    url = url_attr if isinstance(url_attr, str) else ""
    return (name, url)


def _extract_fight_metadata(
    soup: BeautifulSoup,
) -> tuple[str | None, str | None, int | None, str | None, str | None, str | None]:
    """Extract method, method_detail, round, time, time_format, referee.

    Returns:
        Tuple of (method, method_detail, round_finished, time_finished,
        time_format, referee).
    """
    method: str | None = None
    method_detail: str | None = None
    round_finished: int | None = None
    time_finished: str | None = None
    time_format: str | None = None
    referee: str | None = None

    # Method from first text item
    method_el = soup.select_one("i.b-fight-details__text-item_first")
    if method_el:
        style_el = method_el.select_one("i[style]")
        if style_el:
            method = style_el.get_text(strip=True)

    # Iterate text items for round, time, time format, referee
    text_items = soup.select("i.b-fight-details__text-item")
    for item in text_items:
        label_el = item.select_one("i.b-fight-details__label")
        if label_el is None:
            continue
        label_text = label_el.get_text(strip=True)

        if label_text == "Round:":
            value = item.get_text(strip=True).replace("Round:", "").strip()
            if value.isdigit():
                round_finished = int(value)
        elif label_text == "Time format:":
            time_format = item.get_text(strip=True).replace("Time format:", "").strip()
        elif label_text == "Time:":
            time_finished = item.get_text(strip=True).replace("Time:", "").strip()
        elif label_text == "Referee:":
            span = item.select_one("span")
            if span:
                referee = span.get_text(strip=True)

    # Method detail from second p.b-fight-details__text (Details section)
    detail_paragraphs = soup.select("p.b-fight-details__text")
    if len(detail_paragraphs) >= 2:
        detail_item = detail_paragraphs[1].select_one("i.b-fight-details__text-item_first")
        if detail_item:
            style_el = detail_item.select_one("i[style]")
            if style_el:
                detail_text = style_el.get_text(strip=True)
                if detail_text:
                    method_detail = detail_text

    return method, method_detail, round_finished, time_finished, time_format, referee


# ── Column helpers ─────────────────────────────────────────────────────────


def _paired_text(col: Tag) -> tuple[str, str]:
    """Extract paired text values from a table column.

    Each stat column contains exactly 2 <p> tags:
    first = fighter A, second = fighter B.
    """
    paragraphs = col.select("p.b-fight-details__table-text")
    if len(paragraphs) < 2:
        return ("", "")
    return (paragraphs[0].get_text(strip=True), paragraphs[1].get_text(strip=True))


def _extract_fighter_names_from_col(col: Tag) -> tuple[str, str]:
    """Extract fighter names from the first (Fighter) column.

    Names are in <a> links within <p> tags, or just <p> text.
    """
    paragraphs = col.select("p.b-fight-details__table-text")
    if len(paragraphs) < 2:
        return ("", "")
    name_a = paragraphs[0].get_text(strip=True)
    name_b = paragraphs[1].get_text(strip=True)
    return (name_a, name_b)


# ── Row-level parsers (single row) ────────────────────────────────────────


def _parse_totals_from_row(
    row: Tag,
) -> tuple[RoundStatsRaw, RoundStatsRaw]:
    """Parse a single <tr> row into a pair of RoundStatsRaw.

    Columns: Fighter | KD | Sig.str | Sig.str% | Total str | Td | Td% |
             Sub.att | Rev | Ctrl
    """
    cols = row.select("td.b-fight-details__table-col")
    if len(cols) < 10:
        msg = (
            f"Fight detail cardinality check failed: expected >=10 totals columns, got {len(cols)}"
        )
        raise ValueError(msg)

    name_a, name_b = _extract_fighter_names_from_col(cols[0])
    kd_a, kd_b = _paired_text(cols[1])
    sig_str_a, sig_str_b = _paired_text(cols[2])
    sig_pct_a, sig_pct_b = _paired_text(cols[3])
    total_a, total_b = _paired_text(cols[4])
    td_a, td_b = _paired_text(cols[5])
    td_pct_a, td_pct_b = _paired_text(cols[6])
    sub_a, sub_b = _paired_text(cols[7])
    rev_a, rev_b = _paired_text(cols[8])
    ctrl_a, ctrl_b = _paired_text(cols[9])

    fighter_a = RoundStatsRaw(
        fighter_name=name_a,
        knockdowns=kd_a,
        sig_str=sig_str_a,
        sig_str_pct=sig_pct_a,
        total_str=total_a,
        td=td_a,
        td_pct=td_pct_a,
        sub_att=sub_a,
        rev=rev_a,
        ctrl=ctrl_a,
    )
    fighter_b = RoundStatsRaw(
        fighter_name=name_b,
        knockdowns=kd_b,
        sig_str=sig_str_b,
        sig_str_pct=sig_pct_b,
        total_str=total_b,
        td=td_b,
        td_pct=td_pct_b,
        sub_att=sub_b,
        rev=rev_b,
        ctrl=ctrl_b,
    )
    return (fighter_a, fighter_b)


def _parse_sig_strikes_from_row(
    row: Tag,
) -> tuple[SigStrikesRaw, SigStrikesRaw]:
    """Parse a single <tr> row into a pair of SigStrikesRaw.

    Columns: Fighter | Sig.str | Sig.str% | Head | Body | Leg |
             Distance | Clinch | Ground
    """
    cols = row.select("td.b-fight-details__table-col")
    if len(cols) < 9:
        msg = (
            f"Fight detail cardinality check failed: "
            f"expected >=9 sig strike columns, got {len(cols)}"
        )
        raise ValueError(msg)

    name_a, name_b = _extract_fighter_names_from_col(cols[0])
    sig_a, sig_b = _paired_text(cols[1])
    pct_a, pct_b = _paired_text(cols[2])
    head_a, head_b = _paired_text(cols[3])
    body_a, body_b = _paired_text(cols[4])
    leg_a, leg_b = _paired_text(cols[5])
    dist_a, dist_b = _paired_text(cols[6])
    clinch_a, clinch_b = _paired_text(cols[7])
    ground_a, ground_b = _paired_text(cols[8])

    fighter_a = SigStrikesRaw(
        fighter_name=name_a,
        sig_str=sig_a,
        sig_str_pct=pct_a,
        head=head_a,
        body=body_a,
        leg=leg_a,
        distance=dist_a,
        clinch=clinch_a,
        ground=ground_a,
    )
    fighter_b = SigStrikesRaw(
        fighter_name=name_b,
        sig_str=sig_b,
        sig_str_pct=pct_b,
        head=head_b,
        body=body_b,
        leg=leg_b,
        distance=dist_b,
        clinch=clinch_b,
        ground=ground_b,
    )
    return (fighter_a, fighter_b)


# ── Section-level parsers (single row, backward compat) ───────────────────


def _parse_totals_row(
    section: Tag,
) -> tuple[RoundStatsRaw, RoundStatsRaw]:
    """Parse a totals table section into a pair of RoundStatsRaw.

    Uses only the FIRST row (for fight-level totals sections with 1 row).
    """
    row = section.select_one("tbody tr.b-fight-details__table-row")
    if row is None:
        msg = "Fight detail cardinality check failed: no data row in totals section"
        raise ValueError(msg)
    return _parse_totals_from_row(row)


def _parse_sig_strikes_row(
    section: Tag,
) -> tuple[SigStrikesRaw, SigStrikesRaw]:
    """Parse a sig strikes table section into a pair of SigStrikesRaw.

    Uses only the FIRST row (for fight-level sig strikes sections with 1 row).
    """
    row = section.select_one("tbody tr.b-fight-details__table-row")
    if row is None:
        msg = "Fight detail cardinality check failed: no data row in sig strikes section"
        raise ValueError(msg)
    return _parse_sig_strikes_from_row(row)


# ── Multi-row section parsers (per-round data) ────────────────────────────


def _parse_all_totals_rows(
    section: Tag,
) -> list[tuple[RoundStatsRaw, RoundStatsRaw]]:
    """Parse ALL rows in a totals table section.

    Used for per-round sections where each row represents one round.
    Returns a list of (fighter_a, fighter_b) tuples, one per row.
    """
    rows = section.select("tbody tr.b-fight-details__table-row")
    results: list[tuple[RoundStatsRaw, RoundStatsRaw]] = []
    for row in rows:
        cols = row.select("td.b-fight-details__table-col")
        if len(cols) < 10:
            continue
        results.append(_parse_totals_from_row(row))
    return results


def _parse_all_sig_strikes_rows(
    section: Tag,
) -> list[tuple[SigStrikesRaw, SigStrikesRaw]]:
    """Parse ALL rows in a sig strikes table section.

    Used for per-round sections where each row represents one round.
    Returns a list of (fighter_a, fighter_b) tuples, one per row.
    """
    rows = section.select("tbody tr.b-fight-details__table-row")
    results: list[tuple[SigStrikesRaw, SigStrikesRaw]] = []
    for row in rows:
        cols = row.select("td.b-fight-details__table-col")
        if len(cols) < 9:
            continue
        results.append(_parse_sig_strikes_from_row(row))
    return results


# ── Sig strikes summation ─────────────────────────────────────────────────


def _sum_sig_strikes_single(
    per_round: list[SigStrikesRaw],
) -> SigStrikesRaw:
    """Sum per-round sig strikes for a single fighter into fight-level totals."""
    if not per_round:
        return SigStrikesRaw(
            fighter_name="",
            sig_str="0 of 0",
            sig_str_pct="0%",
            head="0 of 0",
            body="0 of 0",
            leg="0 of 0",
            distance="0 of 0",
            clinch="0 of 0",
            ground="0 of 0",
        )

    fighter_name = per_round[0].fighter_name

    # Sum each "X of Y" field across rounds
    fields = ["sig_str", "head", "body", "leg", "distance", "clinch", "ground"]
    sums: dict[str, tuple[int, int]] = {}
    for field in fields:
        total_landed = 0
        total_attempted = 0
        for rnd in per_round:
            landed, attempted = _parse_x_of_y_raw(getattr(rnd, field))
            total_landed += landed
            total_attempted += attempted
        sums[field] = (total_landed, total_attempted)

    sig_landed, sig_attempted = sums["sig_str"]

    return SigStrikesRaw(
        fighter_name=fighter_name,
        sig_str=_format_x_of_y(sig_landed, sig_attempted),
        sig_str_pct=_format_pct(sig_landed, sig_attempted),
        head=_format_x_of_y(*sums["head"]),
        body=_format_x_of_y(*sums["body"]),
        leg=_format_x_of_y(*sums["leg"]),
        distance=_format_x_of_y(*sums["distance"]),
        clinch=_format_x_of_y(*sums["clinch"]),
        ground=_format_x_of_y(*sums["ground"]),
    )


def _sum_sig_strikes(
    per_round: list[tuple[SigStrikesRaw, SigStrikesRaw]],
) -> tuple[SigStrikesRaw, SigStrikesRaw]:
    """Sum per-round sig strikes to produce fight-level totals for both fighters."""
    fighter_a_rounds = [pair[0] for pair in per_round]
    fighter_b_rounds = [pair[1] for pair in per_round]
    return (
        _sum_sig_strikes_single(fighter_a_rounds),
        _sum_sig_strikes_single(fighter_b_rounds),
    )

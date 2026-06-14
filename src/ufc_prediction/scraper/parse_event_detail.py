"""Parse UFCStats.com event detail page into EventDetail model.

Uses BeautifulSoup4 with lxml backend and CSS selectors per D-04.
Cardinality checks raise ValueError on structural changes per D-10/D-11.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from ufc_prediction.scraper.models import EventDetail, FightSummary


def parse_event_detail(html: str, event_url: str) -> EventDetail:
    """Parse event detail HTML into an EventDetail model.

    Args:
        html: Raw HTML from the event detail page.
        event_url: The URL of the event page (stored on the model).

    Returns:
        EventDetail with list of FightSummary items.

    Raises:
        ValueError: If fight table or required elements not found.
    """
    soup = BeautifulSoup(html, "lxml")

    # Extract event metadata
    name_el = soup.select_one("span.b-content__title-highlight")
    name = name_el.get_text(strip=True) if name_el else ""

    date_str = ""
    location = ""
    for item in soup.select("li.b-list__box-list-item"):
        text = item.get_text(strip=True)
        if text.startswith("Date:"):
            date_str = text.replace("Date:", "").strip()
        elif text.startswith("Location:"):
            location = text.replace("Location:", "").strip()

    # Find fight rows
    fight_rows = soup.select("tr.js-fight-details-click")
    if not fight_rows:
        msg = "No fight rows found in event detail"
        raise ValueError(msg)

    fights: list[FightSummary] = []
    for row in fight_rows:
        fight = _parse_fight_row(row)
        fights.append(fight)

    return EventDetail(
        name=name,
        date_str=date_str,
        location=location,
        url=event_url,
        fights=fights,
    )


def _parse_fight_row(row: Tag) -> FightSummary:
    """Parse a single fight row from the event detail table.

    Args:
        row: A <tr> element with class js-fight-details-click.

    Returns:
        FightSummary with fight data.
    """
    _fu = row.get("data-link", "")
    fight_url = _fu if isinstance(_fu, str) else ""

    tds = row.select("td.b-fight-details__table-col")

    # Column 1 (index 0): W/L flags
    flag_col = tds[0]
    flag_paragraphs = flag_col.select("p.b-fight-details__table-text")

    # Determine winner/outcome from flag
    winner = None
    outcome = "win"  # default

    # Check first fighter's flag
    fighter_a_flag = flag_paragraphs[0].select_one("a.b-flag") if len(flag_paragraphs) > 0 else None
    fighter_b_flag = flag_paragraphs[1].select_one("a.b-flag") if len(flag_paragraphs) > 1 else None

    if fighter_a_flag:
        flag_text = fighter_a_flag.get_text(strip=True).lower()
        if flag_text == "win":
            winner = "fighter_a"
            outcome = "win"
        elif flag_text == "draw":
            winner = None
            outcome = "draw"
        elif flag_text == "nc":
            winner = None
            outcome = "nc"
    elif fighter_b_flag:
        flag_text = fighter_b_flag.get_text(strip=True).lower()
        if flag_text == "win":
            winner = "fighter_b"
            outcome = "win"
        elif flag_text == "draw":
            winner = None
            outcome = "draw"
        elif flag_text == "nc":
            winner = None
            outcome = "nc"

    # Column 2 (index 1): Fighter names and URLs
    fighter_col = tds[1]
    fighter_links = fighter_col.select("a.b-link")
    fighter_a_name = fighter_links[0].get_text(strip=True) if len(fighter_links) >= 1 else ""
    _au = fighter_links[0].get("href", "") if len(fighter_links) >= 1 else ""
    fighter_a_url = _au if isinstance(_au, str) else ""
    fighter_b_name = fighter_links[1].get_text(strip=True) if len(fighter_links) >= 2 else ""
    _bu = fighter_links[1].get("href", "") if len(fighter_links) >= 2 else ""
    fighter_b_url = _bu if isinstance(_bu, str) else ""

    # Column 7 (index 6): Weight class + title fight
    weight_col = tds[6]
    weight_text_el = weight_col.select_one("p.b-fight-details__table-text")
    weight_class_raw = weight_text_el.get_text(strip=True) if weight_text_el else ""
    is_title_fight = weight_col.select_one("img[src*='belt.png']") is not None

    # Column 8 (index 7): Method
    method_col = tds[7]
    method_paragraphs = method_col.select("p.b-fight-details__table-text")
    method = method_paragraphs[0].get_text(strip=True) if len(method_paragraphs) >= 1 else None
    method_detail = None
    if len(method_paragraphs) >= 2:
        detail_text = method_paragraphs[1].get_text(strip=True)
        method_detail = detail_text if detail_text else None

    # Column 9 (index 8): Round
    round_col = tds[8]
    round_text = round_col.get_text(strip=True)
    round_finished = int(round_text) if round_text.isdigit() else None

    # Column 10 (index 9): Time
    time_col = tds[9]
    time_finished = time_col.get_text(strip=True) or None

    return FightSummary(
        fight_url=fight_url,
        fighter_a_name=fighter_a_name,
        fighter_a_url=fighter_a_url,
        fighter_b_name=fighter_b_name,
        fighter_b_url=fighter_b_url,
        winner=winner,
        outcome=outcome,
        weight_class_raw=weight_class_raw,
        is_title_fight=is_title_fight,
        method=method,
        method_detail=method_detail,
        round_finished=round_finished,
        time_finished=time_finished,
    )

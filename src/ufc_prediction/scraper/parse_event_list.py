"""Parse UFCStats.com event listing page into EventSummary models.

Uses BeautifulSoup4 with lxml backend and CSS selectors per D-04.
Cardinality checks raise ValueError on structural changes per D-10/D-11.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from ufc_prediction.scraper.models import EventSummary


def parse_event_list(html: str, min_events: int = 100) -> list[EventSummary]:
    """Parse event listing HTML into a list of EventSummary models.

    Args:
        html: Raw HTML from the event listing page.
        min_events: Minimum expected events for cardinality check.
            Defaults to 100 for production (UFCStats has 700+ events).
            Set lower for test fixtures.

    Returns:
        List of EventSummary models.

    Raises:
        ValueError: If event table not found or cardinality check fails.
    """
    soup = BeautifulSoup(html, "lxml")

    table = soup.select_one("table.b-statistics__table-events")
    if table is None:
        msg = "Event listing table not found"
        raise ValueError(msg)

    rows = table.select("tr.b-statistics__table-row, tr.b-statistics__table-row_type_first")

    events: list[EventSummary] = []
    for row in rows:
        # Skip spacer rows
        if row.select_one("td.b-statistics__table-col_type_clear"):
            continue

        # Find the content container
        content = row.select_one("i.b-statistics__table-content")
        if content is None:
            continue

        # Extract link
        link = content.select_one("a.b-link")
        if link is None:
            continue

        name = link.get_text(strip=True)
        _url = link.get("href", "")
        url = _url if isinstance(_url, str) else ""

        # Check if upcoming via white link style
        link_classes_raw = link.get("class")
        is_upcoming = (
            isinstance(link_classes_raw, list)
            and "b-link_style_white" in link_classes_raw
        )

        # Extract date
        date_span = content.select_one("span.b-statistics__date")
        date_str = date_span.get_text(strip=True) if date_span else ""

        # Extract location from last td in row
        all_tds = row.select("td")
        location = all_tds[-1].get_text(strip=True) if len(all_tds) > 1 else ""

        events.append(
            EventSummary(
                name=name,
                date_str=date_str,
                location=location,
                url=url,
                is_upcoming=is_upcoming,
            )
        )

    # Cardinality check
    if len(events) < min_events:
        msg = (
            f"Event list cardinality check failed: "
            f"expected >={min_events}, got {len(events)}"
        )
        raise ValueError(msg)

    return events

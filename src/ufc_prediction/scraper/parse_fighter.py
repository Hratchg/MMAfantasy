"""Parse UFCStats.com fighter profile page into FighterProfile model.

Uses BeautifulSoup4 with lxml backend and CSS selectors per D-04.
Cardinality checks raise ValueError on structural changes per D-10/D-11.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from ufc_prediction.scraper.models import FighterProfile


def parse_fighter_profile(html: str) -> FighterProfile:
    """Parse fighter profile HTML into a FighterProfile model.

    Args:
        html: Raw HTML from the fighter profile page.

    Returns:
        FighterProfile with physical attributes and biographical data.

    Raises:
        ValueError: If cardinality checks fail (name not found, or
            fewer than 5 info box items).
    """
    soup = BeautifulSoup(html, "lxml")

    # Name
    name_el = soup.select_one("span.b-content__title-highlight")
    if name_el is None:
        msg = "Fighter profile cardinality check failed: name element not found"
        raise ValueError(msg)
    name = name_el.get_text(strip=True)

    # Nickname
    nickname_el = soup.select_one("p.b-content__Nickname")
    nickname: str | None = None
    if nickname_el:
        nick_text = nickname_el.get_text(strip=True)
        if nick_text:
            nickname = nick_text

    # Info box
    info_box = soup.select_one(
        "div.b-list__info-box_style_small-width"
    )
    if info_box is None:
        msg = "Fighter profile cardinality check failed: info box not found"
        raise ValueError(msg)

    items = info_box.select("li.b-list__box-list-item_type_block")
    if len(items) < 5:
        msg = (
            f"Fighter profile cardinality check failed: "
            f"expected >=5 info box items, got {len(items)}"
        )
        raise ValueError(msg)

    # Extract values by position:
    # 0=height, 1=weight, 2=reach, 3=stance, 4=dob
    height_str = _extract_value(items[0])
    weight_str = _extract_value(items[1])
    reach_str = _extract_value(items[2])
    stance_raw = _extract_value(items[3])
    dob_raw = _extract_value(items[4])

    # Normalize: stance and dob use None for empty/missing,
    # height/weight/reach keep "--" as-is (data/parsers.py handles it)
    stance = stance_raw if stance_raw and stance_raw != "--" else None
    dob_str = dob_raw if dob_raw and dob_raw != "--" else None

    return FighterProfile(
        name=name,
        nickname=nickname,
        height_str=height_str if height_str else None,
        weight_str=weight_str if weight_str else None,
        reach_str=reach_str if reach_str else None,
        stance=stance,
        dob_str=dob_str,
    )


def _extract_value(item: Tag) -> str:
    """Extract the value text from a list item, stripping the label.

    Uses a safe approach that avoids mutating the soup tree:
    gets all text, then removes the label text.
    """
    label_el = item.select_one("i.b-list__box-item-title")
    if label_el is None:
        return item.get_text(strip=True)

    label_text = label_el.get_text(strip=True)
    full_text = item.get_text(strip=True)

    # Remove the label from the full text
    value = full_text.replace(label_text, "", 1).strip()
    return value

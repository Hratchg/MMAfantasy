"""Shared fixtures for data ingestion tests.

Provides sample CSV files mimicking Kaggle dataset formats for integration
and temporal integrity testing.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def rajeevw_fighters_csv(tmp_path: Path) -> Path:
    """Create a temp CSV file mimicking raw_fighter_details.csv format.

    Includes:
    - Jon Jones: full profile with all fields
    - Amanda Nunes: full profile with all fields
    - Test Fighter: all missing/sentinel values to test D-04 nullable handling
    """
    csv_content = (
        "fighter_name,Height,Weight,Reach,Stance,DOB,"
        "SLpM,Str_Acc,SApM,Str_Def,TD_Avg,TD_Acc,TD_Def,Sub_Avg\n"
        "Jon Jones,6' 4\",205 lbs.,84.5\",Orthodox,"
        '"Jul 19, 1987",4.29,57%,2.17,64%,1.83,44%,97%,0.44\n'
        "Amanda Nunes,5' 8\",135 lbs.,69\",Orthodox,"
        '"May 30, 1988",4.71,50%,4.12,54%,2.71,76%,85%,0.49\n'
        'Test Fighter,--,--,--,,,,,,,,,,'
    )
    csv_path = tmp_path / "raw_fighter_details.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    return csv_path


@pytest.fixture()
def rajeevw_fights_csv(tmp_path: Path) -> Path:
    """Create a temp CSV file mimicking raw_total_fight_data.csv format (semicolon-delimited).

    Includes two valid fight rows with different dates for temporal ordering tests.
    """
    header = (
        "R_fighter;B_fighter;R_KD;B_KD;R_SIG_STR.;B_SIG_STR.;R_SIG_STR_pct;B_SIG_STR_pct;"
        "R_TOTAL_STR.;B_TOTAL_STR.;R_TD;B_TD;R_TD_pct;B_TD_pct;R_SUB_ATT;B_SUB_ATT;"
        "R_REV;B_REV;R_CTRL;B_CTRL;"
        "R_HEAD;B_HEAD;R_BODY;B_BODY;R_LEG;B_LEG;"
        "R_DISTANCE;B_DISTANCE;R_CLINCH;B_CLINCH;R_GROUND;B_GROUND;"
        "win_by;last_round;last_round_time;Format;Referee;date;location;Fight_type;Winner;event"
    )
    row1 = (
        "Jon Jones;Amanda Nunes;2;0;69 of 101;48 of 89;68%;54%;"
        "85 of 130;60 of 110;2 of 5;0 of 2;40%;0%;0;1;"
        "0;1;3:15;0:00;"
        "40 of 60;25 of 50;10 of 20;8 of 15;19 of 21;15 of 24;"
        "50 of 70;30 of 55;10 of 15;8 of 12;9 of 15;10 of 23;"
        "Decision - Unanimous;3;5:00;3 Rnd (5-5-5);Herb Dean;March 28, 2020;Las Vegas, Nevada, USA;"
        "Middleweight Bout;Red;UFC 248"
    )
    row2 = (
        "Amanda Nunes;Jon Jones;1;0;55 of 90;40 of 80;61%;50%;"
        "70 of 100;50 of 90;1 of 3;0 of 1;33%;0%;1;0;"
        "0;0;2:30;0:00;"
        "30 of 50;20 of 40;10 of 15;5 of 10;15 of 25;15 of 30;"
        "40 of 60;25 of 50;10 of 12;5 of 8;5 of 8;10 of 22;"
        "KO/TKO;2;3:45;3 Rnd (5-5-5);Marc Goddard;June 15, 2021;Jacksonville, Florida, USA;"
        "Middleweight Bout;Red;UFC 263"
    )
    csv_content = f"{header}\n{row1}\n{row2}"
    csv_path = tmp_path / "raw_total_fight_data.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    return csv_path


@pytest.fixture()
def rajeevw_fights_csv_with_bad_row(tmp_path: Path) -> Path:
    """Create a temp CSV with one valid row and one malformed row (bad date).

    The malformed row has an unparseable date to test D-06 skip-and-log.
    """
    header = (
        "R_fighter;B_fighter;R_KD;B_KD;R_SIG_STR.;B_SIG_STR.;R_SIG_STR_pct;B_SIG_STR_pct;"
        "R_TOTAL_STR.;B_TOTAL_STR.;R_TD;B_TD;R_TD_pct;B_TD_pct;R_SUB_ATT;B_SUB_ATT;"
        "R_REV;B_REV;R_CTRL;B_CTRL;"
        "R_HEAD;B_HEAD;R_BODY;B_BODY;R_LEG;B_LEG;"
        "R_DISTANCE;B_DISTANCE;R_CLINCH;B_CLINCH;R_GROUND;B_GROUND;"
        "win_by;last_round;last_round_time;Format;Referee;date;location;Fight_type;Winner;event"
    )
    valid_row = (
        "Jon Jones;Amanda Nunes;2;0;69 of 101;48 of 89;68%;54%;"
        "85 of 130;60 of 110;2 of 5;0 of 2;40%;0%;0;1;"
        "0;1;3:15;0:00;"
        "40 of 60;25 of 50;10 of 20;8 of 15;19 of 21;15 of 24;"
        "50 of 70;30 of 55;10 of 15;8 of 12;9 of 15;10 of 23;"
        "Decision - Unanimous;3;5:00;3 Rnd (5-5-5);Herb Dean;March 28, 2020;Las Vegas, Nevada, USA;"
        "Middleweight Bout;Red;UFC 248"
    )
    # Malformed row: unparseable date "NOT_A_DATE"
    bad_row = (
        "Bad Fighter A;Bad Fighter B;0;0;0 of 0;0 of 0;0%;0%;"
        "0 of 0;0 of 0;0 of 0;0 of 0;0%;0%;0;0;"
        "0;0;0:00;0:00;"
        "0 of 0;0 of 0;0 of 0;0 of 0;0 of 0;0 of 0;"
        "0 of 0;0 of 0;0 of 0;0 of 0;0 of 0;0 of 0;"
        "Decision - Unanimous;3;5:00;3 Rnd (5-5-5);Herb Dean;NOT_A_DATE;Nowhere;"
        "Middleweight Bout;Red;Bad Event"
    )
    csv_content = f"{header}\n{valid_row}\n{bad_row}"
    csv_path = tmp_path / "raw_total_fight_data.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    return csv_path


@pytest.fixture()
def mdabbert_csv(tmp_path: Path) -> Path:
    """Create a temp CSV mimicking mdabbert ufc-master.csv format (comma-delimited).

    Includes:
    - Row 1: Fight between Jon Jones and Amanda Nunes on 2020-03-28
      (should be detected as duplicate if rajeevw was ingested first)
    - Row 2: New fight from 2023 with a new fighter "New Fighter"
      (should create new Fighter and Fight records)
    """
    header = (
        "R_fighter,B_fighter,date,Winner,weight_class,title_bout,no_of_rounds,"
        "finish,finish_details,finish_round,finish_round_time,location,"
        "R_Height_cms,R_Reach_cms,R_Stance,R_age,"
        "B_Height_cms,B_Reach_cms,B_Stance,B_age"
    )
    # Row 1: duplicate fight (same fighters, same date as rajeevw fixture)
    row1 = (
        "Jon Jones,Amanda Nunes,2020-03-28,Red,Middleweight,False,3,"
        "Decision,Unanimous,3,5:00,Las Vegas Nevada USA,"
        "193.04,214.63,Orthodox,32,"
        "172.72,175.26,Orthodox,31"
    )
    # Row 2: new fight with new fighter
    row2 = (
        "New Fighter,Amanda Nunes,2023-06-10,Red,Bantamweight,False,3,"
        "KO/TKO,Punches,2,3:15,Miami Florida USA,"
        "180.34,182.88,Southpaw,28,"
        "172.72,175.26,Orthodox,35"
    )
    csv_content = f"{header}\n{row1}\n{row2}"
    csv_path = tmp_path / "ufc-master.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    return csv_path

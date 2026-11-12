import json

import pytest

from clip_annotator import auto_cuts
from clip_annotator.wsgi import _auto_cuts

HEADER = "output_folder,source_display,start,end,duration,event_type,side,status\n"


def ledger(tmp_path, *rows):
    path = tmp_path / "sources.csv"
    path.write_text("﻿" + HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("source, expected", [
    ("SenseCV-02-06-2026-IFCE/SenseCV-02-06-2026-IFCE-Gimbal/07", "SenseCV-06-02-2026-IFCE__07"),
    ("SenseCV-24-07-2026-UFC/01", "SenseCV-07-24-2026-UFC__01"),
    (r"SenseCV-05-07-2026-ASSAI\SenseCV-05-07-2026-ASSAI\03", "SenseCV-07-05-2026-ASSAI__03"),
])
def test_ledger_sources_map_to_month_first_clip_ids(source, expected):
    assert auto_cuts.clip_id_for(source) == expected


@pytest.mark.parametrize("source", ["SenseCV-02-06-2026-IFCE", "SenseCV-MAGNETOMETRO/01", "x/01"])
def test_unmappable_sources_are_rejected(source):
    with pytest.raises(ValueError):
        auto_cuts.clip_id_for(source)


def test_build_maps_kinds_sides_and_skips_bad_rows(tmp_path):
    path = ledger(
        tmp_path,
        "a,SenseCV-02-06-2026-IFCE/G/01,1.4714,2.471,1.0,desvio,LEFT,ok",
        "b,SenseCV-02-06-2026-IFCE/G/02,0.5,1.5,1.0,desvio,RIGHT,ok",
        "c,SenseCV-02-06-2026-IFCE/G/03,3.0,4.0,1.0,parada,NONE,ok",
        "d,SenseCV-02-06-2026-IFCE/G/04,2.0,3.0,1.0,livre,NONE,ok",
        "e,SenseCV-02-06-2026-IFCE/G/05,2.0,3.0,1.0,desvio,,ok",       # no side
        "f,SenseCV-02-06-2026-IFCE/G/06,2.0,3.0,1.0,desvio,LEFT,failed",
        "g,SenseCV-02-06-2026-IFCE/G/07,3.0,2.0,1.0,parada,NONE,ok",   # end before start
        "h,SenseCV-02-06-2026-IFCE/H/01,0.1,1.1,1.0,livre,NONE,ok",    # collides with G/01
    )
    cuts, skipped = auto_cuts.build(path)
    assert cuts == {
        "SenseCV-06-02-2026-IFCE__01": {"kind": "desvio", "side": "esquerda", "start_s": 1.471, "end_s": 2.471},
        "SenseCV-06-02-2026-IFCE__02": {"kind": "desvio", "side": "direita", "start_s": 0.5, "end_s": 1.5},
        "SenseCV-06-02-2026-IFCE__03": {"kind": "parada", "side": None, "start_s": 3.0, "end_s": 4.0},
        "SenseCV-06-02-2026-IFCE__04": {"kind": "livre", "side": None, "start_s": 2.0, "end_s": 3.0},
    }
    assert len(skipped) == 4


def test_build_checks_ids_and_durations_against_the_index(tmp_path):
    path = ledger(tmp_path,
                  "a,SenseCV-02-06-2026-IFCE/G/01,1.0,2.0,1.0,desvio,LEFT,ok",
                  "b,SenseCV-02-06-2026-IFCE/G/02,5.0,6.0,1.0,desvio,LEFT,ok",
                  "c,SenseCV-05-07-2026-ASSAI/A/01,1.0,2.0,1.0,parada,NONE,ok")
    durations = {"SenseCV-06-02-2026-IFCE__01": 6.9, "SenseCV-06-02-2026-IFCE__02": 5.5}
    cuts, skipped = auto_cuts.build(path, durations)
    assert list(cuts) == ["SenseCV-06-02-2026-IFCE__01"]
    assert any("after the clip" in m for m in skipped)
    assert any("not in the index" in m for m in skipped)


def test_cli_writes_a_file_that_load_accepts(tmp_path):
    path = ledger(tmp_path, "a,SenseCV-02-06-2026-IFCE/G/01,1.0,2.0,1.0,desvio,RIGHT,ok")
    out = tmp_path / "out" / "auto_cuts.json"
    assert auto_cuts.main(["--ledger", str(path), "--out", str(out)]) == 0
    assert auto_cuts.load(out) == {
        "SenseCV-06-02-2026-IFCE__01": {"kind": "desvio", "side": "direita", "start_s": 1.0, "end_s": 2.0}}


def test_missing_file_means_no_suggestions(tmp_path):
    assert auto_cuts.load(tmp_path / "absent.json") == {}


@pytest.mark.parametrize("raw", [
    [],
    {"../x": {"kind": "livre", "side": None, "start_s": 0, "end_s": 1}},
    {"c": {"kind": "desvio", "side": None, "start_s": 0, "end_s": 1}},
    {"c": {"kind": "parada", "side": "esquerda", "start_s": 0, "end_s": 1}},
    {"c": {"kind": "livre", "side": None, "start_s": 2, "end_s": 1}},
    {"c": {"kind": "livre", "side": None, "start_s": True, "end_s": 1}},
    {"c": {"kind": "outro", "side": None, "start_s": 0, "end_s": 1}},
])
def test_corrupt_files_fail_loudly(tmp_path, raw):
    path = tmp_path / "auto_cuts.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        auto_cuts.load(path)


def test_bundled_file_is_valid_and_mostly_deviations():
    cuts = auto_cuts.load()
    assert len(cuts) > 500
    assert sum(c["kind"] == "desvio" for c in cuts.values()) > 300


def test_server_wiring_honours_the_path_override(tmp_path):
    path = tmp_path / "cuts.json"
    path.write_text(json.dumps({"c": {"kind": "livre", "side": None, "start_s": 0, "end_s": 1}}),
                    encoding="utf-8")
    assert list(_auto_cuts({"AUTO_CUTS_JSON": str(path)})) == ["c"]
    assert len(_auto_cuts({})) == len(auto_cuts.load())

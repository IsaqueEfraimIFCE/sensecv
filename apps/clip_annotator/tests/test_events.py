import pytest

from clip_annotator.events import ClipEvents, frame_at, is_safe_id
from conftest import desvio


@pytest.fixture
def empty():
    return ClipEvents.empty(10.0, 30.0)


def parada(start=7.0, end=None, obstacle=None):
    return {"kind": "parada", "side": None, "obstacle": obstacle, "start_s": start, "end_s": end}


def test_new_clip_is_pending(empty):
    assert empty.status == "pendente"
    assert empty.to_json() == {"events": [], "verdict": None, "done": False, "note": "",
                               "status": "pendente"}


def test_events_are_sorted_and_set_the_verdict(empty):
    doc = empty.replace_with({"events": [parada(), desvio(2.0, 3.5)]})
    assert [e.kind for e in doc.events] == ["desvio", "parada"]
    assert doc.verdict == "eventos"
    assert doc.status == "andamento"


def test_event_json_carries_frames_shown_at_each_time(empty):
    event = empty.replace_with({"events": [desvio(2.0, 3.5)]}).to_json()["events"][0]
    assert event == {"kind": "desvio", "side": "esquerda", "obstacle": "centro",
                     "start_s": 2.0, "end_s": 3.5, "start_frame": 60, "end_frame": 105}


def test_frame_at_uses_floor():
    assert frame_at(0.0, 30) == 0
    assert frame_at(1 / 30 - 0.001, 30) == 0
    assert frame_at(2.0, 30) == 60  # exact boundaries are not lost to float error


def test_times_are_rounded_to_milliseconds(empty):
    doc = empty.replace_with({"events": [desvio(2.00049, 3.123456)]})
    assert (doc.events[0].start_s, doc.events[0].end_s) == (2.0, 3.123)


@pytest.mark.parametrize("event, message", [
    ({**desvio(), "side": None}, "side"),
    ({**desvio(), "end_s": None}, "end time"),
    (desvio(3.0, 3.0), "after"),
    (desvio(-1.0, 1.0), "outside"),
    (desvio(2.0, 11.0), "outside"),
    ({**desvio(), "start_s": True}, "number"),
    ({**desvio(), "start_s": float("nan")}, "number"),
    ({**desvio(), "kind": "giro"}, "kind"),
    ({**desvio(), "obstacle": "longe"}, "obstacle"),
    ({**parada(), "side": "direita"}, "no side"),
    ("desvio", "object"),
])
def test_invalid_events_are_rejected(empty, event, message):
    with pytest.raises(ValueError, match=message):
        empty.replace_with({"events": [event]})


def test_parada_may_have_no_end(empty):
    doc = empty.replace_with({"events": [parada(9.9)]})
    assert doc.events[0].end_s is None
    assert doc.to_json()["events"][0]["end_frame"] is None


def test_duration_slack_allows_the_browser_duration(empty):
    assert empty.replace_with({"events": [desvio(9.0, 10.4)]}).events[0].end_s == 10.4


def test_overlapping_events_are_rejected(empty):
    with pytest.raises(ValueError, match="overlap"):
        empty.replace_with({"events": [desvio(2.0, 4.0), desvio(3.9, 5.0, side="direita")]})
    with pytest.raises(ValueError, match="overlap"):
        empty.replace_with({"events": [parada(3.0), desvio(2.0, 3.0)]})


def test_too_many_events_are_rejected(empty):
    events = [desvio(i * 0.4, i * 0.4 + 0.2) for i in range(21)]
    with pytest.raises(ValueError, match="at most"):
        empty.replace_with({"events": events})


def test_no_event_verdicts_exclude_events(empty):
    assert empty.replace_with({"verdict": "sem_evento", "done": True}).status == "concluido"
    with pytest.raises(ValueError, match="cannot be"):
        empty.replace_with({"events": [desvio()], "verdict": "inutilizavel"})


def test_deleting_every_event_clears_the_eventos_verdict(empty):
    assert empty.replace_with({"events": [], "verdict": "eventos"}).verdict is None


def test_done_needs_a_verdict(empty):
    with pytest.raises(ValueError, match="at least one event"):
        empty.replace_with({"done": True})
    assert empty.replace_with({"events": [desvio()], "done": True}).done


@pytest.mark.parametrize("payload", [
    [], {"done": "yes"}, {"note": 5}, {"note": "x" * 501}, {"events": {}}, {"verdict": "talvez"},
])
def test_malformed_payloads_are_rejected(empty, payload):
    with pytest.raises(ValueError):
        empty.replace_with(payload)


def test_note_is_trimmed_and_counts_as_started(empty):
    doc = empty.replace_with({"note": "  câmera tremida  "})
    assert doc.note == "câmera tremida"
    assert doc.status == "andamento"


def test_round_trip_through_json(empty):
    doc = empty.replace_with({"events": [desvio(2.0, 3.0), parada(8.0, 9.0, "centro")],
                              "done": True, "note": "ok"})
    assert ClipEvents.from_json(doc.to_json(), 10.0, 30.0) == doc


def test_stored_times_beyond_a_shorter_republished_duration_still_load(empty):
    stored = empty.replace_with({"events": [desvio(8.0, 9.9)]}).to_json()
    doc = ClipEvents.from_json(stored, 8.6, 30.0)  # index rebuilt with a shorter video
    assert doc.events[0].end_s == 9.9
    with pytest.raises(ValueError, match="outside"):  # saving asks for a fix
        doc.replace_with(stored)
    assert doc.replace_with({"events": [desvio(8.0, 8.5)]}).events[0].end_s == 8.5


def test_from_json_rejects_garbage():
    with pytest.raises(ValueError):
        ClipEvents.from_json({"events": ["x"]}, 10.0, 30.0)
    with pytest.raises(ValueError):
        ClipEvents.from_json([], 10.0, 30.0)


def test_empty_needs_positive_timing():
    with pytest.raises(ValueError):
        ClipEvents.empty(0, 30)


@pytest.mark.parametrize("value, ok", [
    ("SenseCV-10-09-2026-UNIFOR-ISAQUE__41", True), ("annotator01", True),
    ("../x", False), ("", False), ("a b", False), (None, False), ("x" * 101, False),
    ("abc\n", False),
])
def test_is_safe_id(value, ok):
    assert is_safe_id(value) is ok

"""Per-frame label logic: individual frames are labelled, never whole clips."""
import pytest

from frame_annotator.labels import ClipAnnotation, FrameLabel


def test_new_clip_has_every_frame_unlabelled():
    clip = ClipAnnotation.empty(10)

    assert clip.frame_count == 10
    assert clip.labelled_count == 0
    assert not clip.is_complete


def test_free_path_forces_deviation_none():
    clip = ClipAnnotation.empty(5).apply(0, 4, obstacle="free_path")

    assert clip.label_at(2) == FrameLabel("free_path", "none", False)
    assert clip.is_complete


def test_obstacle_alone_is_partial_until_deviation_is_set():
    clip = ClipAnnotation.empty(3).apply(0, 2, obstacle="obstacle")
    assert clip.label_at(0) == FrameLabel("obstacle", None, False)
    assert clip.labelled_count == 0

    clip = clip.apply(0, 2, deviation="left")
    assert clip.label_at(0) == FrameLabel("obstacle", "left", False)
    assert clip.is_complete


def test_labels_apply_to_a_frame_range_only():
    clip = ClipAnnotation.empty(10).apply(3, 5, obstacle="obstacle", deviation="right")

    assert clip.labelled_count == 3
    assert clip.label_at(2) is None
    assert clip.label_at(3) == FrameLabel("obstacle", "right", False)
    assert clip.label_at(5) == FrameLabel("obstacle", "right", False)
    assert clip.label_at(6) is None


def test_unusable_clears_labels_and_counts_as_done():
    clip = (
        ClipAnnotation.empty(4)
        .apply(0, 3, obstacle="obstacle", deviation="left")
        .apply(1, 2, unusable=True)
    )

    assert clip.label_at(1) == FrameLabel(None, None, True)
    assert clip.label_at(0) == FrameLabel("obstacle", "left", False)
    assert clip.is_complete


def test_labelling_an_unusable_frame_makes_it_usable_again():
    clip = (
        ClipAnnotation.empty(2)
        .apply(0, 1, unusable=True)
        .apply(0, 0, obstacle="free_path")
    )

    assert clip.label_at(0) == FrameLabel("free_path", "none", False)
    assert clip.label_at(1) == FrameLabel(None, None, True)


def test_switching_to_free_path_overrides_an_earlier_side():
    clip = (
        ClipAnnotation.empty(2)
        .apply(0, 1, obstacle="obstacle", deviation="left")
        .apply(0, 1, obstacle="free_path")
    )

    assert clip.label_at(0) == FrameLabel("free_path", "none", False)


def test_leaving_free_path_resets_the_forced_side_so_it_must_be_chosen():
    clip = (
        ClipAnnotation.empty(2)
        .apply(0, 1, obstacle="free_path")
        .apply(0, 1, obstacle="obstacle")
    )

    assert clip.label_at(0) == FrameLabel("obstacle", None, False)
    assert not clip.is_complete


def test_unflagging_an_unusable_frame_returns_it_to_unlabelled():
    clip = ClipAnnotation.empty(2).apply(0, 1, unusable=True).apply(0, 0, unusable=False)

    assert clip.label_at(0) is None
    assert clip.label_at(1) == FrameLabel(None, None, True)


def test_side_with_free_path_in_one_call_is_rejected():
    with pytest.raises(ValueError):
        ClipAnnotation.empty(2).apply(0, 1, obstacle="free_path", deviation="left")


def test_side_on_a_frame_that_is_free_path_is_rejected():
    clip = ClipAnnotation.empty(2).apply(0, 1, obstacle="free_path")

    with pytest.raises(ValueError):
        clip.apply(0, 1, deviation="right")


@pytest.mark.parametrize(
    "start,end", [(-1, 2), (0, 5), (3, 2), (5, 5)], ids=["neg", "past-end", "reversed", "oob"]
)
def test_invalid_ranges_are_rejected(start, end):
    with pytest.raises(ValueError):
        ClipAnnotation.empty(5).apply(start, end, obstacle="obstacle")


@pytest.mark.parametrize(
    "kwargs",
    [{"obstacle": "wall"}, {"deviation": "up"}, {"unusable": "yes"}],
)
def test_unknown_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ClipAnnotation.empty(3).apply(0, 2, **kwargs)


def test_apply_returns_a_new_clip_and_leaves_the_original_untouched():
    original = ClipAnnotation.empty(3)
    changed = original.apply(0, 2, obstacle="free_path")

    assert original.labelled_count == 0
    assert changed.labelled_count == 3


def test_runs_roundtrip_and_stay_compact():
    clip = (
        ClipAnnotation.empty(100)
        .apply(0, 59, obstacle="free_path")
        .apply(60, 79, obstacle="obstacle", deviation="left")
        .apply(80, 84, unusable=True)
    )

    runs = clip.to_runs()

    assert runs == [
        [0, 59, "free_path", "none", False],
        [60, 79, "obstacle", "left", False],
        [80, 84, None, None, True],
    ]
    assert ClipAnnotation.from_runs(100, runs) == clip


def test_runs_keep_partial_labels():
    clip = ClipAnnotation.empty(4).apply(1, 2, obstacle="obstacle")

    assert clip.to_runs() == [[1, 2, "obstacle", None, False]]
    assert ClipAnnotation.from_runs(4, clip.to_runs()) == clip


@pytest.mark.parametrize(
    "runs",
    [
        [[0, 9, "obstacle", "left", False]],  # beyond frame_count=5
        [[0, 2, "wall", "none", False]],
        [[0, 3, "obstacle", "left", False], [2, 4, "free_path", "none", False]],  # overlap
        [[0, 1, "free_path", "left", False]],  # side with free path
        [[0, 1]],  # malformed
    ],
    ids=["oob", "bad-value", "overlap", "free-with-side", "malformed"],
)
def test_from_runs_rejects_corrupt_data(runs):
    with pytest.raises(ValueError):
        ClipAnnotation.from_runs(5, runs)

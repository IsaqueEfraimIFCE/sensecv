"""Sprite sheets are a transport format only: frames stay individually addressable."""
import io

from PIL import Image

from frame_annotator.sprites import build_sheets

COLORS = [(250, 0, 0), (0, 250, 0), (0, 0, 250), (250, 250, 0), (0, 250, 250),
          (250, 0, 250), (120, 120, 120)]


def _frames(tmp_path, colors, size=(50, 40)):
    paths = []
    for i, color in enumerate(colors):
        path = tmp_path / f"f{i:03d}.jpg"
        Image.new("RGB", size, color).save(path, "JPEG", quality=95)
        paths.append(path)
    return paths


def _close(a, b, tolerance=25):
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


def test_frames_are_packed_in_order_across_sheets(tmp_path):
    sheets = build_sheets(_frames(tmp_path, COLORS), tile=32, cols=3, rows=2)

    assert [(s.sheet_index, s.first_frame, s.frame_count) for s in sheets] == [(0, 0, 6), (1, 6, 1)]
    assert all(s.cols == 3 and s.tile_w == 32 and s.tile_h == 32 for s in sheets)
    first = Image.open(io.BytesIO(sheets[0].jpeg))
    assert first.size == (96, 64)
    assert _close(first.getpixel((32 + 16, 32 + 16)), COLORS[4])  # frame 4: col 1, row 1
    assert _close(first.getpixel((16, 16)), COLORS[0])


def test_every_sheet_is_cols_wide_so_css_percentages_hold(tmp_path):
    sheets = build_sheets(_frames(tmp_path, COLORS), tile=32, cols=3, rows=2)

    last = Image.open(io.BytesIO(sheets[1].jpeg))
    assert last.size == (96, 32)
    assert _close(last.getpixel((16, 16)), COLORS[6])


def test_non_square_frames_fill_the_tile(tmp_path):
    sheets = build_sheets(_frames(tmp_path, [(10, 200, 10)], size=(80, 20)), tile=32, cols=1, rows=1)

    image = Image.open(io.BytesIO(sheets[0].jpeg))
    assert image.size == (32, 32)
    assert _close(image.getpixel((2, 2)), (10, 200, 10))
    assert _close(image.getpixel((29, 29)), (10, 200, 10))


def test_empty_frame_list_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        build_sheets([])

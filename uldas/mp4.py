#file: uldas/mp4.py

"""In-place language tagging for MP4 / M4V (ISO base media) files.

This is the MP4 counterpart of what ``mkvpropedit`` does for Matroska:
each track carries its language in the 2-byte ``language`` field of its
``moov/trak/mdia/mdhd`` box, packed as three 5-bit ISO 639-2 letters.
We locate that field and overwrite exactly those two bytes.

Only the language can be set this way.  Track names live in the
variable-length ``hdlr`` box and MP4 has no standard forced flag, so
neither is supported for MP4.
"""

import io
import logging
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Optional

from uldas.constants import MP4_EXTENSIONS

logger = logging.getLogger(__name__)

# Handler types → track kind (mirrors ffmpeg's mov demuxer classification)
AUDIO_HANDLERS = {b"soun", b"m1a "}
SUBTITLE_HANDLERS = {b"sbtl", b"subt", b"subp", b"clcp", b"text"}

# Top-level box types we expect to see first in a real ISO-BMFF file.
_KNOWN_TOP_LEVEL = {
    b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"moof",
    b"mfra", b"sidx", b"styp", b"uuid", b"meta", b"pdin", b"junk",
}

# Refuse to slurp absurdly large moov boxes into memory.
_MAX_MOOV_BYTES = 256 * 1024 * 1024

_LANG_RE = re.compile(r"[a-z]{3}")


@dataclass
class Box:
    type: bytes
    offset: int      # absolute offset of the box header
    size: int        # total box size including header
    header: int      # header length: 8, or 16 for 64-bit sizes

    @property
    def body(self) -> int:
        return self.offset + self.header

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass
class TrackInfo:
    ordinal: int             # 0-based index among tracks of the same kind
    kind: str                # "audio" | "subtitle" | "video" | "other"
    handler: bytes
    mdhd_offset: int         # absolute offset of the mdhd box header
    mdhd_version: int
    lang_offset: int         # absolute offset of the 2-byte language field
    current_lang: Optional[str]
    has_elng: bool = False


def is_mp4(path: Path) -> bool:
    return path.suffix.lower() in MP4_EXTENSIONS


# ── Language packing ─────────────────────────────────────────────────────
def pack_language(lang3: str) -> int:
    """Pack a 3-letter lowercase ISO 639-2 code into the mdhd 15-bit form."""
    if not _LANG_RE.fullmatch(lang3):
        raise ValueError(f"Not a 3-letter ISO 639-2 code: {lang3!r}")
    c1, c2, c3 = (ord(c) - 0x60 for c in lang3)
    return (c1 << 10) | (c2 << 5) | c3


def unpack_language(code: int) -> Optional[str]:
    """Inverse of :func:`pack_language`.

    Returns ``None`` for legacy Macintosh language codes (``< 0x400``),
    which ffmpeg maps through its own table (e.g. ``0`` → ``eng``).
    """
    code &= 0x7FFF
    if code < 0x400:
        return None
    chars = ((code >> 10) & 0x1F, (code >> 5) & 0x1F, code & 0x1F)
    if any(c < 1 or c > 26 for c in chars):
        return None
    return "".join(chr(c + 0x60) for c in chars)


# ── Box parsing ──────────────────────────────────────────────────────────
def _read_box_header(fp: BinaryIO, offset: int, end: int) -> Optional[Box]:
    """Read one box header at *offset*; ``end`` bounds the parent."""
    if offset + 8 > end:
        return None
    fp.seek(offset)
    hdr = fp.read(8)
    if len(hdr) < 8:
        return None
    size, btype = struct.unpack(">I4s", hdr)
    header = 8
    if size == 1:
        large = fp.read(8)
        if len(large) < 8:
            return None
        size = struct.unpack(">Q", large)[0]
        header = 16
    elif size == 0:
        size = end - offset       # extends to end of parent / file
    if size < header or offset + size > end:
        raise ValueError(
            f"Malformed box {btype!r} at {offset}: size {size} exceeds bounds"
        )
    return Box(btype, offset, size, header)


def _iter_boxes(fp: BinaryIO, start: int, end: int):
    offset = start
    while offset < end:
        box = _read_box_header(fp, offset, end)
        if box is None:
            return
        yield box
        offset = box.end


def _find_child(fp: BinaryIO, parent: Box, btype: bytes) -> Optional[Box]:
    for child in _iter_boxes(fp, parent.body, parent.end):
        if child.type == btype:
            return child
    return None


def _file_size(fp: BinaryIO) -> int:
    cur = fp.tell()
    fp.seek(0, io.SEEK_END)
    size = fp.tell()
    fp.seek(cur)
    return size


def _find_moov(fp: BinaryIO) -> Box:
    total = _file_size(fp)
    moov: Optional[Box] = None
    first = True
    for box in _iter_boxes(fp, 0, total):
        if first:
            if box.type not in _KNOWN_TOP_LEVEL:
                raise ValueError(
                    f"Not an ISO base media file (first box is {box.type!r})"
                )
            first = False
        if box.type == b"moov":
            if moov is not None:
                raise ValueError("File contains more than one moov box")
            moov = box
    if moov is None:
        raise ValueError("No moov box found")
    if moov.size > _MAX_MOOV_BYTES:
        raise ValueError(f"moov box too large ({moov.size} bytes)")
    return moov


def _parse_trak(fp: BinaryIO, trak: Box) -> Optional[TrackInfo]:
    mdia = _find_child(fp, trak, b"mdia")
    if mdia is None:
        raise ValueError(f"trak at {trak.offset} has no mdia box")
    hdlr = _find_child(fp, mdia, b"hdlr")
    mdhd = _find_child(fp, mdia, b"mdhd")
    if hdlr is None or mdhd is None:
        raise ValueError(f"trak at {trak.offset} is missing hdlr/mdhd")
    has_elng = _find_child(fp, mdia, b"elng") is not None

    # hdlr: FullBox(4) + pre_defined(4) + handler_type(4)
    fp.seek(hdlr.body + 8)
    handler = fp.read(4)
    if len(handler) < 4:
        raise ValueError(f"Truncated hdlr box at {hdlr.offset}")

    fp.seek(mdhd.body)
    version = fp.read(1)
    if not version:
        raise ValueError(f"Truncated mdhd box at {mdhd.offset}")
    version = version[0]
    if version > 1:
        raise ValueError(f"Unsupported mdhd version {version} at {mdhd.offset}")
    # FullBox(4) + creation/modification/timescale/duration
    lang_offset = mdhd.body + 4 + (16 if version == 0 else 28)
    if lang_offset + 4 > mdhd.end:
        raise ValueError(f"mdhd box at {mdhd.offset} is too short")

    fp.seek(lang_offset)
    packed = struct.unpack(">H", fp.read(2))[0]

    if handler in AUDIO_HANDLERS:
        kind = "audio"
    elif handler in SUBTITLE_HANDLERS:
        kind = "subtitle"
    elif handler == b"vide":
        kind = "video"
    else:
        kind = "other"

    return TrackInfo(
        ordinal=-1, kind=kind, handler=handler,
        mdhd_offset=mdhd.offset, mdhd_version=version,
        lang_offset=lang_offset, current_lang=unpack_language(packed),
        has_elng=has_elng,
    )


def list_tracks(fp: BinaryIO) -> List[TrackInfo]:
    """Enumerate every ``trak`` in file order (== ffprobe stream order),
    with per-kind ordinals assigned."""
    moov = _find_moov(fp)
    if _find_child(fp, moov, b"cmov") is not None:
        raise ValueError("Compressed moov (cmov) is not supported")

    tracks: List[TrackInfo] = []
    counts = {"audio": 0, "subtitle": 0, "video": 0, "other": 0}
    for child in _iter_boxes(fp, moov.body, moov.end):
        if child.type != b"trak":
            continue
        info = _parse_trak(fp, child)
        info.ordinal = counts[info.kind]
        counts[info.kind] += 1
        tracks.append(info)
    return tracks


# ── Public writer ────────────────────────────────────────────────────────
def set_track_language(
    file_path: Path,
    kind: str,
    ordinal: int,
    lang3: str,
    expected_count: Optional[int] = None,
) -> bool:
    """Set the language of the *ordinal*-th track of *kind* in *file_path*.

    ``expected_count`` is the number of tracks of that kind the caller saw
    via ffprobe; if our own count differs we refuse rather than risk
    tagging the wrong track.
    """
    if kind not in ("audio", "subtitle"):
        logger.error("Unsupported track kind %r for %s", kind, file_path)
        return False
    if not _LANG_RE.fullmatch(lang3 or ""):
        logger.error("Cannot write %r to %s: MP4 needs a 3-letter ISO 639-2 code",
                     lang3, file_path)
        return False

    try:
        with open(file_path, "r+b") as fp:
            tracks = [t for t in list_tracks(fp) if t.kind == kind]

            if expected_count is not None and len(tracks) != expected_count:
                logger.error(
                    "%s: found %d %s track(s) in moov but ffprobe reported %d; "
                    "refusing to write language", file_path, len(tracks), kind,
                    expected_count,
                )
                return False
            if ordinal < 0 or ordinal >= len(tracks):
                logger.error("%s: %s track %d not found (%d present)",
                             file_path, kind, ordinal, len(tracks))
                return False

            track = tracks[ordinal]
            if track.has_elng:
                logger.warning(
                    "%s: %s track %d also carries an extended language (elng) "
                    "box, which some players prefer over the tag being written",
                    file_path.name, kind, ordinal,
                )

            # Belt and braces: make sure we're really pointing at the mdhd
            # box before touching anything.
            fp.seek(track.mdhd_offset)
            hdr = fp.read(8)
            if len(hdr) < 8 or hdr[4:8] != b"mdhd":
                logger.error("%s: mdhd box moved unexpectedly; aborting", file_path)
                return False

            packed = struct.pack(">H", pack_language(lang3))
            fp.seek(track.lang_offset)
            fp.write(packed)
            fp.flush()
            fp.seek(track.lang_offset)
            if fp.read(2) != packed:
                logger.error("%s: language write did not verify", file_path)
                return False
        return True
    except (OSError, ValueError, struct.error) as exc:
        logger.error("Error updating MP4 language for %s: %s", file_path, exc)
        return False

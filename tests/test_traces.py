"""S0 trace writer: offsets must be one unit, and it must be CHARACTERS.

`supervision_spans.char_start/char_end` are character ranges into the decoded
content of the day file — `span_text` slices `read_text()` output, which is a
Python `str`, i.e. code points. An external review (2026-10-16) found the writer
computed `start` as the file's BYTE size (`st_size`) and then added a CHAR count
for the end: mixed units that coincide for pure-ASCII days and silently corrupt
every span on any day containing Tamil, Hindi, emoji, or any other multibyte
text — which is precisely the user this assistant is built for.

The acceptance test is the round trip: for English, Tamil and emoji lines, the
recorded span must resolve back to exactly the line that was written.
"""
from __future__ import annotations

import json

from friday.memory.traces import Trace, TraceWriter


def test_offsets_are_characters_not_bytes(root):
    tamil = "வாடகை மாதம் பதினெட்டாயிரம் ரூபாய்"          # rent is 18000 rupees
    emoji = "lease renewal moved to next week ✅🎉"

    w = TraceWriter(root / "memory" / "traces")
    # A multibyte line FIRST, so any byte-based offset for the second line is
    # strictly larger than the correct char-based one.
    w.append(Trace(kind="capture", content=tamil))
    _, s2, e2 = w.append(Trace(kind="capture", content="rent is 18000 INR"))
    _, s3, e3 = w.append(Trace(kind="capture", content=emoji))

    day = w.path_for()
    lines = day.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(lines) == 3
    # exact round trip: the span IS the line, for every script
    assert w.span_text(day, s2, e2) == lines[1]
    assert w.span_text(day, s3, e3) == lines[2]
    # and the first line's own span still resolves after later appends
    first = json.loads(lines[0])
    assert w.span_text(day, 0, len(lines[0])) == lines[0]
    assert first["content"] == tamil


def test_span_offsets_survive_many_multibyte_appends(root):
    """Not just one: N multibyte lines, then a span on the last one. Byte offsets
    would drift by (bytes - chars) per line; characters do not."""
    w = TraceWriter(root / "memory" / "traces")
    payload = "மாத வாடகை ௧௮௦௦௦ ✅"
    for i in range(8):
        w.append(Trace(kind="capture", content=f"{i} {payload}"))
    tr = Trace(kind="capture", content="இறுதி வரி 🎯")
    _, s, e = w.append(tr)

    day = w.path_for()
    lines = day.read_text(encoding="utf-8").splitlines(keepends=True)
    assert w.span_text(day, s, e) == lines[-1]
    assert json.loads(w.span_text(day, s, e))["content"] == "இறுதி வரி 🎯"

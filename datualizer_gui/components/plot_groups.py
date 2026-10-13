"""Which tracks share a subplot and which colour each track gets (campaign window).

Grouping and colour scheme are personal preferences (QSettings); a manual colour per channel
belongs to the campaign (`ChannelDisplay.color` in campaign.json).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from datualizer_core.campaign import Track

NO_UNIT = "(no unit)"


class GroupBy(str, Enum):
    CHANNEL = "channel"  # one subplot per channel
    UNIT = "unit"        # channels with the same unit share a subplot
    SOURCE = "source"    # all channels of a source share a subplot


class ColorBy(str, Enum):
    CHANNEL = "channel"  # a colour per track, in order
    SOURCE = "source"    # all channels of a source in its colour (as in the coverage strip)
    MANUAL = "manual"    # the colour set per channel, else as CHANNEL


def track_key(tr: Track) -> str:
    return f"{tr.source_id}/{tr.channel}"


@dataclass
class PlotGroup:
    """Tracks drawn into one subplot."""

    key: str
    tracks: list[Track] = field(default_factory=list)
    title: str = ""
    axis_label: str = ""
    unit: str = ""

    @property
    def legend(self) -> bool:
        return len(self.tracks) > 1


def group_tracks(tracks: Sequence[Track], group_by: GroupBy, source_labels: dict[str, str] | None = None) -> list[PlotGroup]:
    """Subplots in order of first appearance. With one subplot per channel the group key is the
    track key ('source/channel'), so a single-track group is addressed like the track."""
    group_by = GroupBy(group_by)
    labels = source_labels or {}
    groups: dict[str, PlotGroup] = {}
    for tr in tracks:
        if group_by is GroupBy.CHANNEL:
            key = track_key(tr)
        elif group_by is GroupBy.UNIT:
            key = f"unit:{tr.unit}"
        else:
            key = f"source:{tr.source_id}"
        groups.setdefault(key, PlotGroup(key)).tracks.append(tr)
    for g in groups.values():
        first = g.tracks[0]
        units = {t.unit for t in g.tracks}
        g.unit = first.unit if len(units) == 1 else ""
        if group_by is GroupBy.CHANNEL:
            g.title, g.axis_label = first.label, first.name
        elif group_by is GroupBy.UNIT:
            g.title = f"Unit: {first.unit or NO_UNIT}"
            g.axis_label = first.name if len(g.tracks) == 1 else ""
        else:
            g.title = labels.get(first.source_id, first.source_id)
            g.axis_label = first.name if len(g.tracks) == 1 else ""
        if len(units) > 1:  # mixed units: name them in the axis label
            g.axis_label = ", ".join(sorted(u or NO_UNIT for u in units))
    return list(groups.values())


def track_colors(
    tracks: Sequence[Track], color_by: ColorBy, palette: Sequence[str], source_order: Sequence[str] = ()
) -> dict[str, str]:
    """Colour per track key. By source, the colour follows the source's position in
    `source_order` (the campaign's source list, as in the coverage strip)."""
    color_by = ColorBy(color_by)
    order = list(source_order)
    out: dict[str, str] = {}
    for i, tr in enumerate(tracks):
        by_channel = palette[i % len(palette)]
        if color_by is ColorBy.SOURCE:
            if tr.source_id not in order:
                order.append(tr.source_id)
            out[track_key(tr)] = palette[order.index(tr.source_id) % len(palette)]
        elif color_by is ColorBy.MANUAL and tr.color:
            out[track_key(tr)] = tr.color
        else:
            out[track_key(tr)] = by_channel
    return out

from typing import override

from picard.plugin3.api import Track, t_

from ..common.data import ReplaygainablePair
from .shared_action import BaseReplayGainAction


class ScanTracks(BaseReplayGainAction[list[Track]]):
    TITLE = t_("action.tracks", "Calculate Replay&Gain…")

    @override
    def filter_objects(self, objs) -> list[list[Track]]:
        return [list(filter(lambda o: isinstance(o, Track), objs))]

    @override
    def get_item_name(self, item: list[Track]):
        return item[0].files[0].filename

    @property
    @override
    def unit(self):
        return "track"

    @override
    def replaygainablepairs_of_item(
        self, item: list[Track]
    ) -> list[ReplaygainablePair]:
        return ReplaygainablePair.from_tracks(item)

    @override
    def update_item(self, item: list[Track]):
        for track in item:
            for file in track.files:
                file.update()
            track.update()

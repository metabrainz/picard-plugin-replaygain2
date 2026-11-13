from picard.plugin3.api import Track, t_

from ..common.data import ReplaygainablePair
from .shared_action import BaseReplayGainAction


class ScanTracks(BaseReplayGainAction[list[Track]]):
    TITLE = t_("action.tracks", "Calculate Replay&Gain…")

    def filter_objects(self, objs) -> list[list[Track]]:
        return [list(filter(lambda o: isinstance(o, Track), objs))]

    def get_item_name(self, item: list[Track]):
        return item[0].files[0].filename

    @property
    def unit(self):
        return "track"

    def replaygainablepairs_of_item(
        self, item: list[Track]
    ) -> list[ReplaygainablePair]:
        return ReplaygainablePair.from_tracks(item)

    def update_item(self, item: list[Track]):
        for track in item:
            for file in track.files:
                file.update()
            track.update()

from typing import final, override

from picard.plugin3.api import Album, t_

from ..common.data import ReplaygainablePair
from .shared_action import BaseReplayGainAction


@final
class ScanAlbums(BaseReplayGainAction[Album]):
    TITLE = t_("action.albums", "Calculate Replay&Gain…")

    @override
    def filter_objects(self, objs) -> list[Album]:
        return list(filter(lambda o: isinstance(o, Album), objs))

    @override
    def get_item_name(self, item: Album):
        return item.metadata["album"]

    @property
    @override
    def unit(self):
        return "album"

    @override
    def replaygainablepairs_of_item(self, item: Album) -> list[ReplaygainablePair]:
        return ReplaygainablePair.from_album(item)

    @override
    def update_item(self, item: Album):
        for track in item.tracks:
            for file in track.files:
                file.update()
            track.update()
        item.update()

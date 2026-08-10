from typing import override

from picard.plugin3.api import Cluster, t_

from ..common.data import ReplaygainablePair
from .shared_action import BaseReplayGainAction


class ScanCluster(BaseReplayGainAction[Cluster]):
    TITLE = t_("action.cluster", "Calculate Cluster Replay&Gain as Album…")

    @override
    def filter_objects(self, objs) -> list[Cluster]:
        return list(filter(lambda o: isinstance(o, Cluster), objs))

    @override
    def get_item_name(self, item: Cluster):
        return item.metadata["album"]

    @property
    @override
    def unit(self):
        return "cluster"

    @override
    def replaygainablepairs_of_item(self, item: Cluster) -> list[ReplaygainablePair]:
        return ReplaygainablePair.from_cluster(item)

    @override
    def update_item(self, item: Cluster):
        for file in item.files:
            file.update()

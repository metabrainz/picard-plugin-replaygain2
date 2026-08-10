from abc import abstractmethod
from functools import partial

from picard.plugin3.api import BaseAction
from picard.util import thread

from ..common.data import ReplaygainablePair
from ..common.rsgain import build_rsgain_options, calculate_replaygain
from ..common.statusbar import StatusbarMessages
from ..common.util import does_rsgain_path_still_exist
from ..options.config import PluginConfig


class BaseReplayGainAction[T](BaseAction):
    def __init__(self):
        super().__init__()
        self.num_items = 0
        self.current = 0

    @abstractmethod
    def filter_objects(self, objs) -> list[T]:
        pass

    @abstractmethod
    def get_item_name(self, item: T) -> str:
        pass

    @property
    @abstractmethod
    def unit(self) -> str:
        pass

    @abstractmethod
    def replaygainablepairs_of_item(self, item: T) -> list[ReplaygainablePair]:
        pass

    @abstractmethod
    def update_item(self, item: T):
        pass

    def callback(self, objs):
        config = PluginConfig(self.api)
        if not does_rsgain_path_still_exist(self.api, config.rsgain_path):
            return
        self.options = build_rsgain_options(config)
        items = self.filter_objects(objs)

        self.num_items = len(items)
        self.current = 0
        StatusbarMessages.inprogress(
            self.api, self.get_item_name(items[0]), self.num_items, self.unit
        )
        for item in items:
            thread.run_task(
                partial(
                    calculate_replaygain,
                    self.api,
                    self.replaygainablepairs_of_item(item),
                    self.options,
                ),
                partial(self._result_callback, item),
            )

    def _format_progress(self):
        if self.num_items == 1:
            return ""
        else:
            self.current += 1
            return f" ({self.current}/{self.num_items})"

    def _result_callback(self, item: T, result=None, error=None):
        progress = self._format_progress()
        if error is None:
            self.update_item(item)
            StatusbarMessages.success(
                self.api, self.get_item_name(item), progress, self.unit
            )
        else:
            StatusbarMessages.failure(
                self.api, self.get_item_name(item), progress, self.unit
            )

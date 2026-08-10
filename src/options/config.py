from typing import final

from picard.plugin3.api import PluginApi

from ..common.data import (
    PLUGIN_CONFIG_ALBUM_AES77,
    PLUGIN_CONFIG_ALBUM_TAGS,
    PLUGIN_CONFIG_CLIP_MODE,
    PLUGIN_CONFIG_MAX_PEAK,
    PLUGIN_CONFIG_OPUS_M23,
    PLUGIN_CONFIG_OPUS_MODE,
    PLUGIN_CONFIG_REFERENCE_LOUDNESS,
    PLUGIN_CONFIG_RSGAIN_COMMAND,
    PLUGIN_CONFIG_TARGET_LOUDNESS,
    PLUGIN_CONFIG_TRUE_PEAK,
    ClipMode,
    OpusMode,
)


@final
class PluginConfig:
    def __init__(self, api: PluginApi) -> None:
        self.api = api

    @classmethod
    def register_with(cls, api: PluginApi):
        api.plugin_config.register_option(PLUGIN_CONFIG_RSGAIN_COMMAND, "rsgain")
        api.plugin_config.register_option(PLUGIN_CONFIG_ALBUM_TAGS, True)
        api.plugin_config.register_option(PLUGIN_CONFIG_ALBUM_AES77, False)
        api.plugin_config.register_option(PLUGIN_CONFIG_TRUE_PEAK, False)
        api.plugin_config.register_option(PLUGIN_CONFIG_REFERENCE_LOUDNESS, False)
        api.plugin_config.register_option(PLUGIN_CONFIG_TARGET_LOUDNESS, -18)
        api.plugin_config.register_option(PLUGIN_CONFIG_CLIP_MODE, ClipMode.POSITIVE)
        api.plugin_config.register_option(PLUGIN_CONFIG_MAX_PEAK, 0)
        api.plugin_config.register_option(PLUGIN_CONFIG_OPUS_MODE, OpusMode.STANDARD)
        api.plugin_config.register_option(PLUGIN_CONFIG_OPUS_M23, False)

    @property
    def rsgain_path(self) -> str:
        return self.api.plugin_config[PLUGIN_CONFIG_RSGAIN_COMMAND]

    @rsgain_path.setter
    def rsgain_path(self, value: str):
        self.api.plugin_config[PLUGIN_CONFIG_RSGAIN_COMMAND] = value

    @property
    def should_calculate_album_gain(self) -> bool:
        return self.api.plugin_config[PLUGIN_CONFIG_ALBUM_TAGS]

    @should_calculate_album_gain.setter
    def should_calculate_album_gain(self, value: bool):
        self.api.plugin_config[PLUGIN_CONFIG_ALBUM_TAGS] = value

    @property
    def should_use_loudest_track_as_album_loudness(self) -> bool:
        return self.api.plugin_config[PLUGIN_CONFIG_ALBUM_AES77]

    @should_use_loudest_track_as_album_loudness.setter
    def should_use_loudest_track_as_album_loudness(self, value: bool):
        self.api.plugin_config[PLUGIN_CONFIG_ALBUM_AES77] = value

    @property
    def should_use_true_peak(self) -> bool:
        return self.api.plugin_config[PLUGIN_CONFIG_TRUE_PEAK]

    @should_use_true_peak.setter
    def should_use_true_peak(self, value: bool):
        self.api.plugin_config[PLUGIN_CONFIG_TRUE_PEAK] = value

    @property
    def should_write_reference_loudness_tags(self) -> bool:
        return self.api.plugin_config[PLUGIN_CONFIG_REFERENCE_LOUDNESS]

    @should_write_reference_loudness_tags.setter
    def should_write_reference_loudness_tags(self, value: bool):
        self.api.plugin_config[PLUGIN_CONFIG_REFERENCE_LOUDNESS] = value

    @property
    def target_loudness(self) -> int:
        return self.api.plugin_config[PLUGIN_CONFIG_TARGET_LOUDNESS]

    @target_loudness.setter
    def target_loudness(self, value: int):
        self.api.plugin_config[PLUGIN_CONFIG_TARGET_LOUDNESS] = value

    @property
    def clipping_protection_mode(self) -> ClipMode:
        return self.api.plugin_config[PLUGIN_CONFIG_CLIP_MODE]

    @clipping_protection_mode.setter
    def clipping_protection_mode(self, value: ClipMode):
        self.api.plugin_config[PLUGIN_CONFIG_CLIP_MODE] = value

    @property
    def max_peak_db(self) -> int:
        return self.api.plugin_config[PLUGIN_CONFIG_MAX_PEAK]

    @max_peak_db.setter
    def max_peak_db(self, value: int):
        self.api.plugin_config[PLUGIN_CONFIG_MAX_PEAK] = value

    @property
    def opus_mode(self) -> OpusMode:
        return self.api.plugin_config[PLUGIN_CONFIG_OPUS_MODE]

    @opus_mode.setter
    def opus_mode(self, value: OpusMode):
        self.api.plugin_config[PLUGIN_CONFIG_OPUS_MODE] = value

    @property
    def should_opus_r128_to_m23(self) -> bool:
        return self.api.plugin_config[PLUGIN_CONFIG_OPUS_M23]

    @should_opus_r128_to_m23.setter
    def should_opus_r128_to_m23(self, value: bool):
        self.api.plugin_config[PLUGIN_CONFIG_OPUS_M23] = value

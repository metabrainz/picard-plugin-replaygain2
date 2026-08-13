import os
from typing import ClassVar, override

from picard.plugin3.api import OptionsPage, PageOptionConfigs
from PyQt6.QtWidgets import QFileDialog

from ..common.data import (
    PLUGIN_CONFIG_ALBUM_AES77,
    PLUGIN_CONFIG_ALBUM_LOAD,
    PLUGIN_CONFIG_ALBUM_TAGS,
    PLUGIN_CONFIG_CLIP_MODE,
    PLUGIN_CONFIG_MAX_PEAK,
    PLUGIN_CONFIG_OPUS_M23,
    PLUGIN_CONFIG_OPUS_MODE,
    PLUGIN_CONFIG_REFERENCE_LOUDNESS,
    PLUGIN_CONFIG_TARGET_LOUDNESS,
    PLUGIN_CONFIG_TRUE_PEAK,
    ClipMode,
    OpusMode,
)
from ..options.config import PluginConfig
from .ui_options import Ui_ReplayGain2OptionsPage


class ReplayGain2OptionsPage(OptionsPage):
    OPTIONS: ClassVar[PageOptionConfigs] = {
        PLUGIN_CONFIG_ALBUM_TAGS: {"widgets": ["album_tags"]},
        PLUGIN_CONFIG_ALBUM_AES77: {"widgets": ["album_aes77"]},
        PLUGIN_CONFIG_TRUE_PEAK: {"widgets": ["true_peak"]},
        PLUGIN_CONFIG_REFERENCE_LOUDNESS: {"widgets": ["reference_loudness"]},
        PLUGIN_CONFIG_TARGET_LOUDNESS: {
            "widgets": ["target_loudness", "target_loudness_label"]
        },
        PLUGIN_CONFIG_CLIP_MODE: {"widgets": ["clip_mode", "clip_mode_label"]},
        PLUGIN_CONFIG_MAX_PEAK: {"widgets": ["max_peak", "max_peak_label"]},
        PLUGIN_CONFIG_OPUS_MODE: {"widgets": ["opus_mode", "opus_mode_label"]},
        PLUGIN_CONFIG_OPUS_M23: {"widgets": ["opus_m23"]},
        PLUGIN_CONFIG_ALBUM_LOAD: {"widgets": ["album_load"]},
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_ReplayGain2OptionsPage()
        self.ui.setupUi(self)
        self.plugin_config = PluginConfig(self.api)

        for mode in ClipMode:
            label = self._get_clip_mode_label(mode)
            self.ui.clip_mode.addItem(label, mode)

        for mode in OpusMode:
            label = self._get_opus_mode_label(mode)
            self.ui.opus_mode.addItem(label, mode)

        self.ui.rsgain_command_browse.clicked.connect(self.rsgain_command_browse)

    def _get_clip_mode_label(self, clip_mode: ClipMode):
        option_clip_mode = "option.clip_mode"
        match clip_mode:
            case ClipMode.DISABLED:
                return self.api.tr(f"{option_clip_mode}.disabled", "Disabled")
            case ClipMode.POSITIVE:
                return self.api.tr(
                    f"{option_clip_mode}.enabled_positive_gain",
                    "Enabled for positive gain values only",
                )
            case ClipMode.ALWAYS:
                return self.api.tr(
                    f"{option_clip_mode}.enabled_always", "Always enabled"
                )

    def _get_opus_mode_label(self, opus_mode: OpusMode):
        option_opus = "option.opus"
        match opus_mode:
            case OpusMode.STANDARD:
                return self.api.tr(
                    f"{option_opus}.standard", "Write standard ReplayGain tags"
                )
            case OpusMode.R128:
                return self.api.tr(f"{option_opus}.r128", "Write R128_*_GAIN tags")
            case OpusMode.BOTH:
                return self.api.tr(
                    f"{option_opus}.both", "Write both standard and R128 tags"
                )

    @override
    def load(self):
        self.ui.rsgain_command.setText(self.plugin_config.rsgain_path)
        self.ui.album_tags.setChecked(self.plugin_config.should_calculate_album_gain)
        self.ui.album_aes77.setChecked(
            self.plugin_config.should_use_loudest_track_as_album_loudness
        )
        self.ui.true_peak.setChecked(self.plugin_config.should_use_true_peak)
        self.ui.reference_loudness.setChecked(
            self.plugin_config.should_write_reference_loudness_tags
        )
        self.ui.target_loudness.setValue(self.plugin_config.target_loudness)
        self.ui.clip_mode.setCurrentText(
            self._get_clip_mode_label(self.plugin_config.clipping_protection_mode)
        )
        self.ui.max_peak.setValue(self.plugin_config.max_peak_db)
        self.ui.opus_mode.setCurrentText(
            self._get_opus_mode_label(self.plugin_config.opus_mode)
        )
        self.ui.opus_m23.setChecked(self.plugin_config.should_opus_r128_to_m23)
        self.ui.album_load.setChecked(self.plugin_config.should_calculate_on_album_load)

    @override
    def save(self):
        self.plugin_config.rsgain_path = self.ui.rsgain_command.text()
        self.plugin_config.should_calculate_album_gain = self.ui.album_tags.isChecked()
        self.plugin_config.should_use_loudest_track_as_album_loudness = (
            self.ui.album_aes77.isChecked()
        )
        self.plugin_config.should_use_true_peak = self.ui.true_peak.isChecked()
        self.plugin_config.should_write_reference_loudness_tags = (
            self.ui.reference_loudness.isChecked()
        )
        self.plugin_config.target_loudness = self.ui.target_loudness.value()
        self.plugin_config.clipping_protection_mode = self.ui.clip_mode.currentData()
        self.plugin_config.max_peak_db = self.ui.max_peak.value()
        self.plugin_config.opus_mode = self.ui.opus_mode.currentData()
        self.plugin_config.should_opus_r128_to_m23 = self.ui.opus_m23.isChecked()
        self.plugin_config.should_calculate_on_album_load = (
            self.ui.album_load.isChecked()
        )

    def rsgain_command_browse(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "", self.ui.rsgain_command.text()
        )
        if path:
            path = os.path.normpath(path)
            self.ui.rsgain_command.setText(path)

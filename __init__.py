# -*- coding: utf-8 -*-

from dataclasses import dataclass
import os
import shutil
import subprocess  # nosec: B404
from enum import Enum, IntEnum
from functools import partial
from typing import Any, final, override

from picard.formats import (
    AiffFile,
    ASFFile,
    FLACFile,
    MonkeysAudioFile,
    MP3File,
    MP4File,
    MusepackFile,
    OggFLACFile,
    OggOpusFile,
    OggSpeexFile,
    OggTheoraFile,
    OggVorbisFile,
    TAKFile,
    WAVFile,
    WavPackFile,
)
from picard.item import ListOfMetadataItems
from picard.metadata import Metadata
from picard.plugin3.api import (
    Album,
    BaseAction,
    Cluster,
    File,
    OptionsPage,
    PluginApi,
    Track,
    t_,
)
from picard.track import NonAlbumTrack
from picard.util import thread
from PyQt6.QtWidgets import QFileDialog

from .ui_options import Ui_ReplayGain2OptionsPage

SUPPORTED_FORMATS = (
    AiffFile,
    ASFFile,
    FLACFile,
    MonkeysAudioFile,
    MP3File,
    MP4File,
    MusepackFile,
    OggFLACFile,
    OggOpusFile,
    OggSpeexFile,
    OggTheoraFile,
    OggVorbisFile,
    TAKFile,
    WAVFile,
    WavPackFile,
)

TABLE_HEADER = (
    "filename",
    "loudness",
    "gain",
    "peak",
    "peak_db",
    "peak_type",
    "clipping_adjustment",
)

REPLACEABLE_REPLAYGAIN_TAGS = (
    "replaygain_album_gain",
    "replaygain_album_peak",
    "replaygain_album_range",
    "replaygain_reference_loudness",
    "replaygain_track_gain",
    "replaygain_track_peak",
    "replaygain_track_range",
    "r128_album_gain",
    "r128_track_gain",
)

RSGAIN_TABLE_HEADER_LENGTH = 1

# Plugin option keys
PLUGIN_CONFIG_RSGAIN_COMMAND = "rsgain_command"
PLUGIN_CONFIG_ALBUM_TAGS = "album_tags"
PLUGIN_CONFIG_ALBUM_AES77 = "album_aes77"
PLUGIN_CONFIG_TRUE_PEAK = "true_peak"
PLUGIN_CONFIG_REFERENCE_LOUDNESS = "reference_loudness"
PLUGIN_CONFIG_TARGET_LOUDNESS = "target_loudness"
PLUGIN_CONFIG_CLIP_MODE = "clip_mode"
PLUGIN_CONFIG_MAX_PEAK = "max_peak"
PLUGIN_CONFIG_OPUS_MODE = "opus_mode"
PLUGIN_CONFIG_OPUS_M23 = "opus_m23"


class ClipMode(Enum):
    DISABLED = "n"
    POSITIVE = "p"
    ALWAYS = "a"


class OpusMode(IntEnum):
    STANDARD = 0
    R128 = 1
    BOTH = 2


class ReplayGain2Error(Exception):
    pass


@final
class PluginConfig:
    def __init__(self, api: PluginApi | None = None) -> None:
        self.api = PluginApi.get_api() if not api else api

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


class WindowStatusbarReplaygainCalculationMessages:
    @classmethod
    def inprogress(cls, name: str, count: int, unit: str):
        api = PluginApi.get_api()
        api.tagger.window.set_statusbar_message(
            api.trn(
                f"statusbar.calculating.{unit}s",
                "Calculating ReplayGain for {name}…",
                "Calculating ReplayGain for {count} {unit}s…",
                count,
                name=name,
                count=count,
                unit=unit,
            )
        )

    @classmethod
    def success(cls, name: str, progress: str, unit: str):
        api = PluginApi.get_api()
        api.tagger.window.set_statusbar_message(
            api.tr(
                f"statusbar.success.{unit}s",
                'Successfully calculated ReplayGain for "{name}"{progress}.',
                name=name,
                progress=progress,
            )
        )

    @classmethod
    def failure(cls, name: str, progress: str, unit: str):
        api = PluginApi.get_api()
        api.tagger.window.set_statusbar_message(
            api.tr(
                f"statusbar.failure.{unit}s",
                'Failed to calculate ReplayGain for "{name}"{progress}.',
                name=name,
                progress=progress,
            )
        )

    @classmethod
    def rsgain_not_found(cls):
        api = PluginApi.get_api()
        api.tagger.window.set_statusbar_message(
            api.tr(
                "statusbar.rsgain_not_found",
                "Failed to locate rsgain. Enter the path in the plugin settings.",
            )
        )


@dataclass
class ReplaygainablePair:
    file: File
    is_non_album_track: bool

    @classmethod
    def from_files(
        cls, files: ListOfMetadataItems, is_non_album_track: bool
    ) -> list["ReplaygainablePair"]:
        return [ReplaygainablePair(file, is_non_album_track) for file in files]

    @classmethod
    def from_tracks(cls, tracks: list[Track]) -> list["ReplaygainablePair"]:
        track_files_forest = [
            cls.from_files(track.files, isinstance(track, NonAlbumTrack))
            for track in tracks
        ]
        return [pair for track in track_files_forest for pair in track]

    @classmethod
    def from_cluster(cls, cluster: Cluster) -> list["ReplaygainablePair"]:
        return cls.from_files(cluster.files, False)

    @classmethod
    def from_album(cls, album: Album) -> list["ReplaygainablePair"]:
        return cls.from_tracks(album.tracks)


# Make sure the rsgain executable exists
def does_rsgain_path_still_exist(rsgain_command: str) -> bool:
    if os.path.exists(rsgain_command) or shutil.which(rsgain_command) is not None:
        return True
    # INVARIANT: rsgain no longer exists ):

    WindowStatusbarReplaygainCalculationMessages.rsgain_not_found()
    return False


# Convert Picard settings dict to rsgain command line options
def build_rsgain_options(config: PluginConfig):
    options = ["custom", "-O", "-s", "s"]
    if config.should_calculate_album_gain:
        options.append("-a")
    if config.should_use_loudest_track_as_album_loudness:
        options.append("-e")
    if config.should_use_true_peak:
        options.append("-t")
    options += ["-l", str(config.target_loudness)]
    options += ["-c", str(config.clipping_protection_mode.value)]
    options += ["-m", str(config.max_peak_db)]
    return options


# Convert table row to result dict
def parse_result(line: str):
    result = dict()
    columns = line.split("\t")

    if len(columns) != len(TABLE_HEADER):
        return None
    for i, column in enumerate(columns):
        result[TABLE_HEADER[i]] = column
    return result


# Format the gain as a Q7.8 fixed point number per RFC 7845
# see: https://datatracker.ietf.org/doc/html/rfc7845
def format_r128(result, config: PluginConfig):
    gain = float(result["gain"])
    if config.should_opus_r128_to_m23:
        gain += float(-23 - config.target_loudness)
    return str(int(round(gain * 256.0)))


def update_metadata(
    config: PluginConfig,
    metadata: Metadata,
    track_result,
    album_result,
    is_nat: bool,
    opus_mode: OpusMode,
):
    for tag in REPLACEABLE_REPLAYGAIN_TAGS:
        metadata.delete(tag)

    # Opus R128 tags
    if opus_mode in {OpusMode.R128, OpusMode.BOTH}:
        metadata.set("r128_track_gain", format_r128(track_result, config))
        if album_result is not None:
            metadata.set("r128_album_gain", format_r128(album_result, config))

    # Standard ReplayGain tags
    if opus_mode in {OpusMode.STANDARD, OpusMode.BOTH}:
        metadata.set("replaygain_track_gain", track_result["gain"] + " dB")
        metadata.set("replaygain_track_peak", track_result["peak"])
        if config.should_calculate_album_gain:
            if is_nat:
                metadata.set("replaygain_album_gain", track_result["gain"] + " dB")
                metadata.set("replaygain_album_peak", track_result["peak"])
            elif album_result is not None:
                metadata.set("replaygain_album_gain", album_result["gain"] + " dB")
                metadata.set("replaygain_album_peak", album_result["peak"])
        if config.should_write_reference_loudness_tags:
            metadata.set(
                "replaygain_reference_loudness",
                f"{float(config.target_loudness):.2f} LUFS",
            )


def calculate_replaygain_v2(pairs: list[ReplaygainablePair], options):
    api = PluginApi.get_api()
    config = PluginConfig(api)

    # Validate file formats
    files = [pair.file for pair in pairs]
    for file in files:
        if not isinstanceany(file, SUPPORTED_FORMATS):
            raise ReplayGain2Error(f"File '{file.filename}' is of unsupported format")
    # INVARIANT: All files are a valid format

    filenames = [file.filename for file in files]
    call: Any = [config.rsgain_path] + options + filenames
    for item in call:
        item.encode("utf-8")

    # Prevent an unwanted console spawn in Windows
    si = None
    if os.name == "nt":
        si = subprocess.STARTUPINFO()
        si.dwFlags = subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE

    # Execute the scan with rsgain
    lines = list()
    api.logger.debug(f"Running rsgain with options: {' '.join(options)}")
    with subprocess.Popen(  # nosec: B603
        call,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        startupinfo=si,
        encoding="utf-8",
        text=True,
    ) as process:
        (output, _unused) = process.communicate()
        rc = process.poll()
        if rc:
            api.logger.debug(process.stderr)
            raise ReplayGain2Error(f"rsgain returned non-zero code ({rc})")
        api.logger.debug(output)
        lines = output.splitlines()

    rsgain_album_row_length = 1 if config.should_calculate_album_gain else 0
    valid_rsgain_output_length = (
        len(pairs) + RSGAIN_TABLE_HEADER_LENGTH + rsgain_album_row_length
    )
    if len(lines) != valid_rsgain_output_length:
        raise ReplayGain2Error(f"Unexpected output from rsgain: {lines}")

    lines.pop(0)  # Don't care about the table header

    album_result = None
    if config.should_calculate_album_gain:
        album_result = parse_result(lines[-1])
        lines.pop(-1)

    results = list()
    for line in lines:
        result = parse_result(line)
        if result is None:
            raise ReplayGain2Error("Failed to parse result")
        results.append(result)

    for i, pair in enumerate(pairs):
        opus_mode = (
            config.opus_mode
            if isinstance(pair.file, OggOpusFile)
            else OpusMode.STANDARD
        )

        update_metadata(
            config,
            pair.file.metadata,
            results[i],
            album_result,
            pair.is_non_album_track,
            opus_mode,
        )


def isinstanceany(obj: object, types):
    return any(isinstance(obj, t) for t in types)


class ScanCluster(BaseAction):
    TITLE = t_("action.cluster", "Calculate Cluster Replay&Gain as Album…")

    @override
    def callback(self, objs):
        config = PluginConfig()

        if not does_rsgain_path_still_exist(config.rsgain_path):
            return
        clusters: list[Cluster] = list(filter(lambda o: isinstance(o, Cluster), objs))

        self.options = build_rsgain_options(config)
        num_clusters = len(clusters)
        WindowStatusbarReplaygainCalculationMessages.inprogress(
            clusters[0].metadata["album"], num_clusters, "cluster"
        )
        for cluster in clusters:
            thread.run_task(
                partial(
                    calculate_replaygain_v2,
                    ReplaygainablePair.from_cluster(cluster),
                    self.options,
                ),
                partial(self._replaygain_callback, cluster.files),
            )

    def _replaygain_callback(self, files: ListOfMetadataItems, result=None, error=None):
        if error is None:
            for file in files:
                file.update()
            WindowStatusbarReplaygainCalculationMessages.success(
                files[0].filename, "", "cluster"
            )
        else:
            WindowStatusbarReplaygainCalculationMessages.failure(
                files[0].filename, "", "cluster"
            )


class ScanTracks(BaseAction):
    TITLE = t_("action.tracks", "Calculate Replay&Gain…")

    @override
    def callback(self, objs):
        config = self.api.plugin_config
        window = self.api.tagger.window

        if not does_rsgain_path_still_exist(config["rsgain_command"], window):
            return
        tracks: list[Track] = list(filter(lambda o: isinstance(o, Track), objs))
        self.options = build_rsgain_options(config)
        num_tracks = len(tracks)

        WindowStatusbarReplaygainCalculationMessages.inprogress(
            tracks[0].files[0].filename, num_tracks, "track"
        )
        thread.run_task(
            partial(
                calculate_replaygain_v2,
                ReplaygainablePair.from_tracks(tracks),
                self.options,
            ),
            partial(self._replaygain_callback, tracks),
        )

    def _replaygain_callback(self, tracks, result=None, error=None):
        window = self.api.tagger.window
        if error is None:
            for track in tracks:
                for file in track.files:
                    file.update()
                track.update()
            WindowStatusbarReplaygainCalculationMessages.success(
                tracks[0].files[0].filename, "", "track"
            )
        else:
            WindowStatusbarReplaygainCalculationMessages.failure(
                tracks[0].files[0].filename, "", "track"
            )


def albumgain_callback(progress: str, album: Album, result=None, error=None):
    if error is None:
        for track in album.tracks:
            for file in track.files:
                file.update()
            track.update()
        album.update()
        WindowStatusbarReplaygainCalculationMessages.success(
            album.metadata["album"], progress, "album"
        )
    else:
        WindowStatusbarReplaygainCalculationMessages.failure(
            album.metadata["album"], progress, "album"
        )


@final
class ScanAlbums(BaseAction):
    TITLE = t_("action.albums", "Calculate Replay&Gain…")

    def __init__(self):
        super().__init__()
        self.options = []
        self.num_albums = 0
        self.current = 0

    @override
    def callback(self, objs):
        config = PluginConfig()
        if not does_rsgain_path_still_exist(config.rsgain_path):
            return
        self.options = build_rsgain_options(config)
        albums: list[Album] = list(filter(lambda o: isinstance(o, Album), objs))

        self.num_albums = len(albums)
        self.current = 0
        WindowStatusbarReplaygainCalculationMessages.inprogress(
            albums[0].metadata["album"], self.num_albums, "album"
        )
        for album in albums:
            thread.run_task(
                partial(
                    calculate_replaygain_v2,
                    ReplaygainablePair.from_album(album),
                    self.options,
                ),
                partial(self._albumgain_callback, album),
            )

    def _format_progress(self):
        if self.num_albums == 1:
            return ""
        else:
            self.current += 1
            return f" ({self.current}/{self.num_albums})"

    def _albumgain_callback(self, album: Album, result=None, error=None):
        progress = self._format_progress()
        albumgain_callback(progress, album, result, error)


class ReplayGain2OptionsPage(OptionsPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_ReplayGain2OptionsPage()
        self.ui.setupUi(self)
        self.plugin_config = PluginConfig()

        for mode in ClipMode:
            label = self._load_clip_mode(mode)
            self.ui.opus_mode.addItem(label, mode)

        for mode in OpusMode:
            label = self._load_opus_mode(mode)
            self.ui.opus_mode.addItem(label, mode)

        self.ui.rsgain_command_browse.clicked.connect(self.rsgain_command_browse)

    def _load_clip_mode(self, clip_mode: ClipMode):
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

    def _load_opus_mode(self, opus_mode: OpusMode):
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
            self._load_clip_mode(self.plugin_config.clipping_protection_mode)
        )
        self.ui.max_peak.setValue(self.plugin_config.max_peak_db)
        self.ui.opus_mode.setCurrentText(
            self._load_opus_mode(self.plugin_config.opus_mode)
        )
        self.ui.opus_m23.setChecked(self.plugin_config.should_opus_r128_to_m23)

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

    def rsgain_command_browse(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "", self.ui.rsgain_command.text()
        )
        if path:
            path = os.path.normpath(path)
            self.ui.rsgain_command.setText(path)


def album_metadata_processor_callback(
    api: PluginApi, album: Album, metadata: Metadata, options
):
    WindowStatusbarReplaygainCalculationMessages.inprogress(
        album.metadata["album"], 1, "album"
    )
    config = PluginConfig(api)
    thread.run_task(
        partial(
            calculate_replaygain_v2,
            ReplaygainablePair.from_album(album),
            build_rsgain_options(config),
        ),
        partial(albumgain_callback, "", album),
    )


def enable(api: PluginApi):
    """Called when plugin is enabled."""
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
    api.register_track_action(ScanTracks)
    api.register_album_action(ScanAlbums)
    api.register_cluster_action(ScanCluster)
    api.register_options_page(ReplayGain2OptionsPage)
    api.register_album_metadata_processor(album_metadata_processor_callback)

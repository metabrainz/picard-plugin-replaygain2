# -*- coding: utf-8 -*-

import os
import shutil
import subprocess  # nosec: B404
from enum import Enum, IntEnum
from functools import partial

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

TAGS = (
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


# Make sure the rsgain executable exists
def rsgain_found(rsgain_command, window):
    if not os.path.exists(rsgain_command) and shutil.which(rsgain_command) is None:
        api = PluginApi.get_api()
        window.set_statusbar_message(
            api.tr(
                "statusbar.rsgain_not_found",
                "Failed to locate rsgain. Enter the path in the plugin settings.",
            )
        )
        return False
    return True


# Convert Picard settings dict to rsgain command line options
def build_options(config):
    options = ["custom", "-O", "-s", "s"]
    if config["album_tags"]:
        options.append("-a")
    if config["true_peak"]:
        options.append("-t")
    options += ["-l", str(config["target_loudness"])]
    options += ["-c", str(config["clip_mode"].value)]
    options += ["-m", str(config["max_peak"])]
    return options


# Convert table row to result dict
def parse_result(line):
    result = dict()
    columns = line.split("\t")

    if len(columns) != len(TABLE_HEADER):
        return None
    for i, column in enumerate(columns):
        result[TABLE_HEADER[i]] = column
    return result


# Format the gain as a Q7.8 fixed point number per RFC 7845
# see: https://datatracker.ietf.org/doc/html/rfc7845
def format_r128(result, config):
    gain = float(result["gain"])
    if config["opus_m23"]:
        gain += float(-23 - config["target_loudness"])
    return str(int(round(gain * 256.0)))


def update_metadata(config, metadata, track_result, album_result, is_nat, opus_mode):
    for tag in TAGS:
        metadata.delete(tag)

    # Opus R128 tags
    if opus_mode in (OpusMode.R128, OpusMode.BOTH):
        metadata.set("r128_track_gain", format_r128(track_result, config))
        if album_result is not None:
            metadata.set("r128_album_gain", format_r128(album_result, config))

    # Standard ReplayGain tags
    if opus_mode in (OpusMode.STANDARD, OpusMode.BOTH):
        metadata.set("replaygain_track_gain", track_result["gain"] + " dB")
        metadata.set("replaygain_track_peak", track_result["peak"])
        if config["album_tags"]:
            if is_nat:
                metadata.set("replaygain_album_gain", track_result["gain"] + " dB")
                metadata.set("replaygain_album_peak", track_result["peak"])
            elif album_result is not None:
                metadata.set("replaygain_album_gain", album_result["gain"] + " dB")
                metadata.set("replaygain_album_peak", album_result["peak"])
        if config["reference_loudness"]:
            metadata.set(
                "replaygain_reference_loudness",
                f"{float(config['target_loudness']):.2f} LUFS",
            )


def calculate_replaygain(api: PluginApi, input_objs, options):
    # Make sure files are of supported type, build file list
    files = list()
    valid_list = list()
    for obj in input_objs:
        if isinstance(obj, Track):
            if not obj.files:
                continue
            file = obj.files[0]
        elif isinstance(obj, File):
            file = obj
        else:
            raise ReplayGain2Error(f"Object {obj} is not a Track or File")

        if not isinstanceany(file, SUPPORTED_FORMATS):
            raise ReplayGain2Error(f"File '{file.filename}' is of unsupported format")
        files.append(file.filename)
        valid_list.append(obj)

    call = [api.plugin_config["rsgain_command"]] + options + files
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
            raise ReplayGain2Error(f"rsgain returned non-zero code ({rc})")
        api.logger.debug(output)
        lines = output.splitlines()
    album_tags = api.plugin_config["album_tags"]

    # Make sure the number of rows in the output is what we expected
    if (
        len(lines)
        != 1  # Table header
        + len(valid_list)  # 1 row per track
        + 1
        if album_tags
        else 0
    ):  # Album result
        raise ReplayGain2Error(f"Unexpected output from rsgain: {lines}")
    lines.pop(0)  # Don't care about the table header

    # Parse album result
    album_result = None
    if album_tags:
        album_result = parse_result(lines[-1])
        lines.pop(-1)

    # Parse track results
    results = list()
    for line in lines:
        result = parse_result(line)
        if result is None:
            raise ReplayGain2Error("Failed to parse result")
        results.append(result)

    # Update track metadata with results
    for i, item in enumerate(valid_list):
        if isinstance(item, Track):
            filelist = item.files
        else:  # is a file
            filelist = [item]

        for file in filelist:
            if isinstance(file, OggOpusFile):
                opus_mode = api.plugin_config["opus_mode"]
            else:
                opus_mode = OpusMode.STANDARD

            update_metadata(
                api.plugin_config,
                file.metadata,
                results[i],
                album_result,
                isinstance(item, NonAlbumTrack),
                opus_mode,
            )


def isinstanceany(obj, types):
    return any(isinstance(obj, t) for t in types)


class ScanCluster(BaseAction):
    TITLE = t_("action.cluster", "Calculate Cluster Replay&Gain as Album…")

    def callback(self, objs):
        config = self.api.plugin_config
        window = self.api.tagger.window

        if not rsgain_found(config["rsgain_command"], window):
            return
        clusters = list(filter(lambda o: isinstance(o, Cluster), objs))

        self.options = build_options(config)
        num_clusters = len(clusters)
        window.set_statusbar_message(
            self.api.trn(
                "statusbar.calculating.clusters",
                "Calculating ReplayGain for {name}…",
                "Calculating ReplayGain for {count} clusters…",
                num_clusters,
                name=clusters[0].metadata["album"],
                count=num_clusters,
            )
        )
        for cluster in clusters:
            thread.run_task(
                partial(calculate_replaygain, self.api, cluster.files, self.options),
                partial(self._replaygain_callback, cluster.files),
            )

    def _replaygain_callback(self, files, result=None, error=None):
        window = self.api.tagger.window
        if error is None:
            for file in files:
                file.update()
            window.set_statusbar_message(
                self.api.tr("statusbar.success", "ReplayGain successfully calculated.")
            )
        else:
            window.set_statusbar_message(
                self.api.tr("statusbar.failure", "Could not calculate ReplayGain.")
            )


class ScanTracks(BaseAction):
    TITLE = t_("action.tracks", "Calculate Replay&Gain…")

    def callback(self, objs):
        config = self.api.plugin_config
        window = self.api.tagger.window

        if not rsgain_found(config["rsgain_command"], window):
            return
        tracks = list(filter(lambda o: isinstance(o, Track), objs))
        self.options = build_options(config)
        num_tracks = len(tracks)

        window.set_statusbar_message(
            self.api.trn(
                "statusbar.calculating.tracks",
                "Calculating ReplayGain for {name}…",
                "Calculating ReplayGain for {count} tracks…",
                num_tracks,
                name=tracks[0].files[0].filename,
                count=num_tracks,
            )
        )
        thread.run_task(
            partial(calculate_replaygain, self.api, tracks, self.options),
            partial(self._replaygain_callback, tracks),
        )

    def _replaygain_callback(self, tracks, result=None, error=None):
        window = self.api.tagger.window
        if error is None:
            for track in tracks:
                for file in track.files:
                    file.update()
                track.update()
            window.set_statusbar_message(
                self.api.tr("statusbar.success", "ReplayGain successfully calculated.")
            )
        else:
            window.set_statusbar_message(
                self.api.tr("statusbar.failure", "Could not calculate ReplayGain.")
            )


class ScanAlbums(BaseAction):
    TITLE = t_("action.albums", "Calculate Replay&Gain…")

    def callback(self, objs):
        config = self.api.plugin_config
        window = self.api.tagger.window

        if not rsgain_found(config["rsgain_command"], window):
            return
        self.options = build_options(config)
        albums = list(filter(lambda o: isinstance(o, Album), objs))

        self.num_albums = len(albums)
        self.current = 0
        window.set_statusbar_message(
            self.api.trn(
                "statusbar.calculating.albums",
                "Calculating ReplayGain for {name}…",
                "Calculating ReplayGain for {count} albums…",
                self.num_albums,
                name=albums[0].metadata["album"],
                count=self.num_albums,
            )
        )
        for album in albums:
            thread.run_task(
                partial(calculate_replaygain, self.api, album.tracks, self.options),
                partial(self._albumgain_callback, album),
            )

    def _format_progress(self):
        if self.num_albums == 1:
            return ""
        else:
            self.current += 1
            return f" ({self.current}/{self.num_albums})"

    def _albumgain_callback(self, album, result=None, error=None):
        window = self.api.tagger.window
        progress = self._format_progress()
        if error is None:
            for track in album.tracks:
                for file in track.files:
                    file.update()
                track.update()
            album.update()
            window.set_statusbar_message(
                self.api.tr(
                    "statusbar.success.albums",
                    'Successfully calculated ReplayGain for "{album}"{progress}.',
                    album=album.metadata["album"],
                    progress=progress,
                )
            )
        else:
            window.set_statusbar_message(
                self.api.tr(
                    "statusbar.failure.albums",
                    'Failed to calculate ReplayGain for "{album}"{progress}.',
                    album=album.metadata["album"],
                    progress=progress,
                )
            )


class ReplayGain2OptionsPage(OptionsPage):
    def __init__(self, parent=None):
        super(ReplayGain2OptionsPage, self).__init__(parent)
        self.ui = Ui_ReplayGain2OptionsPage()
        self.ui.setupUi(self)
        self.ui.clip_mode.addItem(
            self.api.tr("option.clip_mode.disabled", "Disabled"),
            ClipMode.DISABLED,
        )
        self.ui.clip_mode.addItem(
            self.api.tr(
                "option.clip_mode.enabled_positive_gain",
                "Enabled for positive gain values only",
            ),
            ClipMode.POSITIVE,
        )
        self.ui.clip_mode.addItem(
            self.api.tr("option.clip_mode.enabled_always", "Always enabled"),
            ClipMode.ALWAYS,
        )
        self.ui.opus_mode.addItem(
            self.api.tr("option.opus.standard", "Write standard ReplayGain tags"),
            OpusMode.STANDARD,
        )
        self.ui.opus_mode.addItem(
            self.api.tr("option.opus.r128", "Write R128_*_GAIN tags"),
            OpusMode.R128,
        )
        self.ui.opus_mode.addItem(
            self.api.tr("option.opus.both", "Write both standard and R128 tags"),
            OpusMode.BOTH,
        )
        self.ui.rsgain_command_browse.clicked.connect(self.rsgain_command_browse)

    def load(self):
        self.ui.rsgain_command.setText(self.api.plugin_config["rsgain_command"])
        self.ui.album_tags.setChecked(self.api.plugin_config["album_tags"])
        self.ui.true_peak.setChecked(self.api.plugin_config["true_peak"])
        self.ui.reference_loudness.setChecked(
            self.api.plugin_config["reference_loudness"]
        )
        self.ui.target_loudness.setValue(self.api.plugin_config["target_loudness"])
        self.ui.clip_mode.setCurrentIndex(self.api.plugin_config["clip_mode"])
        self.ui.max_peak.setValue(self.api.plugin_config["max_peak"])
        self.ui.opus_mode.setCurrentIndex(self.api.plugin_config["opus_mode"])
        self.ui.opus_m23.setChecked(self.api.plugin_config["opus_m23"])

    def save(self):
        self.api.plugin_config["rsgain_command"] = self.ui.rsgain_command.text()
        self.api.plugin_config["album_tags"] = self.ui.album_tags.isChecked()
        self.api.plugin_config["true_peak"] = self.ui.true_peak.isChecked()
        self.api.plugin_config["reference_loudness"] = (
            self.ui.reference_loudness.isChecked()
        )
        self.api.plugin_config["target_loudness"] = self.ui.target_loudness.value()
        self.api.plugin_config["clip_mode"] = self.ui.clip_mode.currentData()
        self.api.plugin_config["max_peak"] = self.ui.max_peak.value()
        self.api.plugin_config["opus_mode"] = self.ui.opus_mode.currentData()
        self.api.plugin_config["opus_m23"] = self.ui.opus_m23.isChecked()

    def rsgain_command_browse(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "", self.ui.rsgain_command.text()
        )
        if path:
            path = os.path.normpath(path)
            self.ui.rsgain_command.setText(path)


def enable(api: PluginApi):
    """Called when plugin is enabled."""
    api.plugin_config.register_option("rsgain_command", "rsgain")
    api.plugin_config.register_option("album_tags", True)
    api.plugin_config.register_option("true_peak", False)
    api.plugin_config.register_option("reference_loudness", False)
    api.plugin_config.register_option("target_loudness", -18)
    api.plugin_config.register_option("clip_mode", ClipMode.POSITIVE)
    api.plugin_config.register_option("max_peak", 0)
    api.plugin_config.register_option("opus_mode", OpusMode.STANDARD)
    api.plugin_config.register_option("opus_m23", False)
    api.register_track_action(ScanTracks)
    api.register_album_action(ScanAlbums)
    api.register_cluster_action(ScanCluster)
    api.register_options_page(ReplayGain2OptionsPage)

import os
import subprocess
from typing import Any

from picard.formats import OggOpusFile
from picard.plugin3.api import Metadata, PluginApi

from ..options.config import PluginConfig
from .data import (
    REPLACEABLE_REPLAYGAIN_TAGS,
    RSGAIN_TABLE_HEADER_LENGTH,
    SUPPORTED_FORMATS,
    TABLE_HEADER,
    OpusMode,
    ReplayGain2Error,
    ReplaygainablePair,
)
from .util import isinstanceany


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
def parse_rsgain_result(line: str):
    columns = line.split("\t")
    if len(columns) != len(TABLE_HEADER):
        return None

    return {TABLE_HEADER[i]: column for i, column in enumerate(columns)}


def calculate_replaygain(api: PluginApi, pairs: list[ReplaygainablePair], options):
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
    lines = []
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
        album_result = parse_rsgain_result(lines[-1])
        lines.pop(-1)

    results = []
    for line in lines:
        result = parse_rsgain_result(line)
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


# Format the gain as a Q7.8 fixed point number per RFC 7845
# see: https://datatracker.ietf.org/doc/html/rfc7845
def format_r128(result, config: PluginConfig):
    gain = float(result["gain"])
    if config.should_opus_r128_to_m23:
        gain += float(-23 - config.target_loudness)
    return str(round(gain * 256.0))

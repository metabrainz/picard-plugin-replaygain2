from dataclasses import dataclass
from enum import Enum, IntEnum

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
from picard.plugin3.api import Album, Cluster, File, Track
from picard.track import NonAlbumTrack

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

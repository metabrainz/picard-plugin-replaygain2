from picard.plugin3.api import PluginApi

from .src.actions.scan_albums import ScanAlbums
from .src.actions.scan_cluster import ScanCluster
from .src.actions.scan_tracks import ScanTracks
from .src.callbacks.album_load import album_metadata_processor_callback
from .src.options.config import PluginConfig
from .src.options.options_page import ReplayGain2OptionsPage


def enable(api: PluginApi):
    """Called when plugin is enabled."""

    PluginConfig.register_with(api)
    api.register_track_action(ScanTracks)
    api.register_album_action(ScanAlbums)
    api.register_cluster_action(ScanCluster)
    api.register_options_page(ReplayGain2OptionsPage)
    api.register_album_metadata_processor(album_metadata_processor_callback)

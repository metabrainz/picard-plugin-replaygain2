from picard.plugin3.api import Album, Metadata, PluginApi

from ..actions.scan_albums import ScanAlbums
from ..options.config import PluginConfig


def album_metadata_processor_callback(
    api: PluginApi, album: Album, metadata: Metadata, options
):
    if not PluginConfig(api).should_calculate_on_album_load:
        return

    # Must run_when_loaded, else tracks will not be present on the Album object
    album.run_when_loaded(lambda: ScanAlbums().callback([album]))

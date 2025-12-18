# MusicBrainz Picard Replaygain 2.0

MusicBrainz Picard 3 plugin to calculate ReplayGain information for tracks and albums according to the
[ReplayGain 2.0 specification](https://wiki.hydrogenaud.io/index.php?title=ReplayGain_2.0_specification).
This plugin depends on the ReplayGain utility [rsgain](https://github.com/complexlogic/rsgain). Users
are required to install rsgain and set its path in the plugin settings before use.

## Usage

Select one or more tracks, albums, or clusters then right click and select Plugin->Calculate ReplayGain.
The plugin will calculate ReplayGain information for the selected items and display the results in the
metadata window. Click the save button to write the tags to file.

The following file formats are supported:

- MP3 (.mp3)
- FLAC (.flac)
- Ogg (.ogg, .oga, spx)
- Opus (.opus)
- MPEG-4 Audio (.m4a, .mp4)
- Wavpack (.wv)
- Monkey's Audio (.ape)
- WMA (.wma)
- MP2 (.mp2)
- WAV (.wav)
- AIFF (.aiff)
- TAK (.tak)
- MusePack _(Stream Version 8 only)_ (.mpc)

This plugin is based on the original ReplayGain plugin by Philipp Wolfer and Sophist.

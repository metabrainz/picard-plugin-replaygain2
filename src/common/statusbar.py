from picard.plugin3.api import PluginApi


class StatusbarMessages:
    @classmethod
    def inprogress(cls, api: PluginApi, name: str, count: int, unit: str):
        api.tagger.window.set_statusbar_message(
            api.trn(
                "statusbar.calculating.{unit}s",
                "Calculating ReplayGain for {name}…",
                "Calculating ReplayGain for {count} {unit}s…",
                count,
                name=name,
                count=count,
                unit=unit,
            )
        )

    @classmethod
    def success(cls, api: PluginApi, name: str, progress: str, unit: str):
        api.tagger.window.set_statusbar_message(
            api.tr(
                "statusbar.success.{unit}s",
                'Successfully calculated ReplayGain for "{name}"{progress}.',
                name=name,
                progress=progress,
                unit=unit,
            )
        )

    @classmethod
    def failure(cls, api: PluginApi, name: str, progress: str, unit: str):
        api.tagger.window.set_statusbar_message(
            api.tr(
                "statusbar.failure.{unit}s",
                'Failed to calculate ReplayGain for "{name}"{progress}.',
                name=name,
                progress=progress,
                unit=unit,
            )
        )

    @classmethod
    def rsgain_not_found(
        cls,
        api: PluginApi,
    ):
        api.tagger.window.set_statusbar_message(
            api.tr(
                "statusbar.rsgain_not_found",
                "Failed to locate rsgain. Enter the path in the plugin settings.",
            )
        )

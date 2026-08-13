# Contributing to MusicBrainz Picard Replaygain 2.0

Picard and associated plugins and documentation has been a collaborative effort by volunteer contributors from the very start, and contributions continue to be welcome from anyone in the community.

You can help with the development of this plugin by:

- Providing code improvements or bug fixes (as pull requests on the [GitHub repository](https://github.com/metabrainz/picard-plugin-replaygain2)).
- Reporting issues or feature requests on the [Picard issue tracker](https://tickets.metabrainz.org/projects/PICARD).
- Help translating the plugin into other languages on [Weblate](https://translations.metabrainz.org/projects/picard-plugins/replaygain2/).

Please also read the [MetaBrainz Contribution Guidelines](https://github.com/metabrainz/guidelines/blob/master/README.md), specifically the "AI use policy" section, and the [MetaBrainz Code of Conduct](https://metabrainz.org/code-of-conduct) before contributing.

## Technical setup

It is recommended to set up a virtual Python environment for the development of this plugin, e.g. by using [venv](https://docs.python.org/3/library/venv.html).

Create the virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

To install the plugin for development inside Picard, you can install the plugin without git support. This allows testing any code changes directly without the need to commit and push changes. Disabling and enabling the plugin Picard is enough to have Picard load the changed code again.

Installing the plugin can be done using the `picard-cli` command line tool:

```bash
picard-cli plugins install --no-git .
```

You can also install it from the Picard GUI by navigating to Options → Plugins → Install Plugin… → Local, selecting the plugin directory, activating the "Load in-place (ignore git)" option, and clicking "Install…".

## Code style and formatting

The plugin uses ruff for linting and formatting, and the project provides pre-commit hooks to run these tools automatically before committing changes. To set up the pre-commit hooks, run the following command inside the active virtual environment:

```bash
pip install pre-commit
pre-commit install
```

## Plugin API auto-completion and type checking

The plugin uses the MusicBrainz Picard v3 plugin API. Most code editors offer auto-completion and type checking for Python. In order to make use of this for the development of this plugin, install the `picard` package in your virtual environment:

```bash
pip install --pre picard
```

import pytest
from click.testing import CliRunner

from cogito.cli import cli


@pytest.mark.parametrize(
    "args",
    [
        ["init", "--help"],
        ["scaffold", "--help"],
        ["run", "--help"],
        ["predict", "--help"],
        ["train", "--help"],
        ["config", "--help"],
        ["config", "version", "--help"],
        ["config", "upgrade", "--help"],
    ],
)
def test_config_dependent_command_help_includes_root_options(args):
    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0
    assert "Global Options:" in result.output
    assert "-c, --config-path TEXT" in result.output
    assert "The path to the configuration file" in result.output


def test_root_help_does_not_repeat_root_options():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Global Options:" not in result.output
    assert result.output.count("--config-path") == 1


def test_version_help_omits_unused_config_option():
    result = CliRunner().invoke(cli, ["version", "--help"])

    assert result.exit_code == 0
    assert "Global Options:" not in result.output
    assert "--config-path" not in result.output

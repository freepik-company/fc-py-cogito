import json

import click

from cogito.core.exceptions import ConfigFileNotFoundError
from cogito.lib.training import run, setup


@click.command()
@click.option("--payload", type=str, required=True, help="The payload for the training")
@click.pass_obj
def train(ctx: click.Context, payload: str) -> None:
    """
    Run a cogito training with the specified payload, printing the result to stdout.

    Example: python -m cogito.cli train --payload '{"key": "value"}'
    """

    try:
        config_path = ctx.get("config_path")
        payload_data = json.loads(payload)

        setup(config_path)
        result = run(config_path, payload_data)

        click.echo(result)
    except ConfigFileNotFoundError:
        click.echo("No configuration file found. Please initialize the project first.")
        exit(1)
    except Exception as e:
        # print stack trace
        # traceback.print_exc()
        click.echo(f"Error: {e}", err=True, color=True)
        exit(1)

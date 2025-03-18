import json

import click

from cogito.core.exceptions import ConfigFileNotFoundError, NoSetupMethodError
from cogito.lib.training import Trainer


@click.command()
@click.option("--payload", type=str, required=True, help="The payload for the training")
@click.pass_obj
def train(ctx: click.Context, payload: str) -> None:
    """
    Run a cogito training with the specified payload, printing the result to stdout.

    Example: python -m cogito.cli train --payload '{"key": "value"}'
    """

    # Load config and payload
    try:
        config_path = ctx.get("config_path")
        payload_data = json.loads(payload)
    except ConfigFileNotFoundError as e:
        click.echo(f"Config file not found: {e}", err=True, color=True)
        exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True, color=True)
        exit(1)

    # Initialize trainer
    try:
        trainer = Trainer(config_path)
    except Exception as e:
        click.echo(f"Error: {e}", err=True, color=True)
        exit(1)

    # Setup trainer
    try:
        trainer.setup()
    except NoSetupMethodError as e:
        click.echo(f"Warning: {e}", err=True, color=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True, color=True)
        exit(1)

    # Run trainer
    try:
        result = trainer.run(payload_data, run_setup=True)
        click.echo(result)
    except Exception as e:
        # print stack trace
        # traceback.print_exc()
        click.echo(f"Error: {e}", err=True, color=True)
        exit(1)

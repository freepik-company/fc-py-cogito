import json

import click

from cogito.core.exceptions import ConfigFileNotFoundError, NoSetupMethodError
from cogito.lib.prediction import Predict


@click.command()
@click.option(
    "--payload", type=str, required=True, help="The payload for the prediction"
)
@click.pass_obj
def predict(ctx: click.Context, payload: str) -> None:
    """
    Run a cogito prediction with the specified payload, printing the result to stdout.

    Example: python -m cogito.cli predict --payload '{"key": "value"}'
    """

    # Load config and initialize predictor
    try:
        config_path = ctx.get("config_path")
        payload_data = json.loads(payload)
        predictor = Predict(config_path)
    except Exception as e:
        click.echo(f"Error initializing the predictor: {e}", err=True, color=True)
        exit(1)

    # Setup predictor
    try:
        predictor.setup()
    except NoSetupMethodError as e:
        click.echo(f"Warning: {e}", err=True, color=True)
    except Exception as e:
        click.echo(f"Error setting up the predictor: {e}", err=True, color=True)
        exit(1)

    # Run predictor
    try:
        result = predictor.run(payload_data)

        click.echo(result.model_dump_json(indent=4))
    except ConfigFileNotFoundError as e:
        click.echo(f"Config file not found: {e}", err=True, color=True)
        exit(1)
    except Exception as e:
        # print stack trace
        # traceback.print_exc()
        click.echo(f"Error: {e}", err=True, color=True)
        exit(1)

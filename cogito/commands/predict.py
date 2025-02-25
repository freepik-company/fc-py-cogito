import json
import os
import sys
import time

import click

from cogito.api.responses import ResultResponse
from cogito.core.config import ConfigFile
from cogito.core.exceptions import ConfigFileNotFoundError
from cogito.core.utils import load_predictor


@click.command()
@click.option(
    "--payload", type=str, required=True, help="The payload for the prediction"
)
@click.pass_obj
def predict(ctx, payload):
    """
    Run a cogito prediction with the specified payload, printing the result to stdout.

    Example: python -m cogito.cli predict --payload '{"key": "value"}'
    """
    config_path = ctx.get("config_path")
    app_dir = os.path.dirname(os.path.abspath(config_path))
    try:
        config = ConfigFile.load_from_file(f"{config_path}")
    except ConfigFileNotFoundError:
        click.echo("No configuration file found. Please initialize the project first.")
        exit(1)

    try:
        sys.path.insert(0, app_dir)
        predictor = config.cogito.server.route.predictor
        predictor_instance = load_predictor(predictor)

        payload_data = json.loads(payload)

        start_time = time.time()
        result = predictor_instance.predict(**payload_data)
        end_time = time.time()

        try:
            response = ResultResponse(
                inference_time_seconds=round(float(end_time - start_time), 2),
                input=payload_data,
                result=result,
            )
            click.echo(response.model_dump_json(indent=4))
        except TypeError:
            click.echo("Error: Unable to serialize the response to JSON.", err=True)
            exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True, color=True)
        exit(1)

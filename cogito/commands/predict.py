import json
import os
import sys

import click

from cogito.core.config import ConfigFile
from cogito.core.exceptions import ConfigFileNotFoundError
from cogito.core.utils import (
    create_request_model,
    load_predictor,
    wrap_handler,
    get_predictor_handler_return_type,
)


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
        _, input_mode_class = create_request_model(
            predictor, predictor_instance.predict
        )
        input_model = input_mode_class(**payload_data)

        response_model = get_predictor_handler_return_type(predictor_instance)

        handler = wrap_handler(
            descriptor=predictor,
            original_handler=predictor_instance.predict,
            response_model=response_model,
        )
        response = handler(input_model)
        click.echo(response.model_dump_json(indent=4))
    except Exception as e:
        # print stack trace
        # traceback.print_exc()
        click.echo(f"Error: {e}", err=True, color=True)
        exit(1)

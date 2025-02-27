import asyncio
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
        # Load predictor instance using the path to the cogito.yaml file
        sys.path.insert(0, app_dir)
        predictor = config.cogito.server.route.predictor
        predictor_instance = load_predictor(predictor)

        # Run setup method asynchronously
        asyncio.run(predictor_instance.setup())

        # Create input model from payload
        payload_data = json.loads(payload)
        _, input_model_class = create_request_model(
            predictor, predictor_instance.predict
        )
        input_model = input_model_class(**payload_data)

        # Get response model type
        response_model = get_predictor_handler_return_type(predictor_instance)

        # Wrap handler with response model
        handler = wrap_handler(
            descriptor=predictor,
            original_handler=predictor_instance.predict,
            response_model=response_model,
        )

        # Call handler with input model
        response = handler(input_model)

        # Print response in JSON format
        click.echo(response.model_dump_json(indent=4))
    except Exception as e:
        # print stack trace
        # traceback.print_exc()
        click.echo(f"Error: {e}", err=True, color=True)
        exit(1)

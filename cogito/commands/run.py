import os
import sys

import click

from cogito import Application


@click.command()
@click.pass_obj
def run(ctx: click.Context) -> None:
    """Run cogito app"""
    config_path = ctx.get("config_path")
    absolute_path = os.path.abspath(config_path)
    click.echo(f"Running '{absolute_path}' cogito application...")
    if not os.path.exists(absolute_path):
        click.echo(
            f"Error: Path '{absolute_path}' does not exist.",
            err=True,
            color=True,
        )
        exit(1)

    try:
        app_dir = os.path.dirname(os.path.abspath(config_path))
        sys.path.insert(0, app_dir)
        app = Application(config_file_path=absolute_path)
        app.run()
    except Exception as e:
        import traceback

        traceback.print_exc()
        click.echo(f"Error: {e}", err=True, color=True)
        exit(1)

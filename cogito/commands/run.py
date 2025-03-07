import os
import sys

import click

from cogito import Application


@click.command()
@click.pass_obj
def run(ctx):
    """Run cogito app"""
    config_path = ctx.get("config_path")
    config_absolute_path = os.path.abspath(config_path)
    click.echo(f"Running '{config_absolute_path}' cogito application...")

    config_dir_absolute_path = os.path.dirname(config_absolute_path)
    os.chdir(config_dir_absolute_path)

    if not os.path.exists(config_absolute_path):
        click.echo(
            f"Error: Path '{config_absolute_path}' does not exist.",
            err=True,
            color=True,
        )
        exit(1)

    try:
        sys.path.insert(0, config_dir_absolute_path)
        app = Application(config_file_path=config_absolute_path)
        app.run()
    except Exception as e:
        click.echo(f"Error: {e}", err=True, color=True)
        exit(1)

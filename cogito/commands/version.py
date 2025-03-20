import click

from cogito._version import __version__


@click.command()
def version():
    """Show the current version of Cogito."""
    click.echo(f"Version: {__version__}")

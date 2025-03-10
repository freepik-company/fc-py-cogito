import click

from cogito.commands.initialize import init
from cogito.commands.scaffold import scaffold
from cogito.commands.run import run
from cogito.commands.predict import predict
from cogito.commands.version import version
from cogito.commands.train import train
from cogito.commands.config import config


@click.group()
@click.option(
    "-c",
    "--config-path",
    type=str,
    default="./cogito.yaml",
    help="The path to the configuration file",
)
@click.pass_context
def cli(ctx, config_path: str = ".") -> None:
    """
    Cogito CLI
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


cli.add_command(init)
cli.add_command(scaffold)
cli.add_command(run)
cli.add_command(predict)
cli.add_command(train)
cli.add_command(version)
cli.add_command(config)


def main():
    cli(obj={})


if __name__ == "__main__":
    main()

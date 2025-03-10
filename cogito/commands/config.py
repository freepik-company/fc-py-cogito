import click

from cogito.core.exceptions import ConfigFileNotFoundError
from cogito.core.config.file import ConfigFile


@click.group()
@click.pass_obj
def config(ctx: click.Context) -> None:
    """Configuration management commands."""
    pass


@config.command()
@click.pass_obj
def version(ctx: click.Context) -> None:
    """Show the current configuration version."""
    try:
        # Load the configuration file using the context's configuration path
        config_file = ConfigFile.load_from_file(ctx["config_path"])
        click.echo(f"Configuration version: {config_file.config_version}")

        # If the configuration has a version field in cogito.server, display it as well
        if (
            config_file.cogito
            and hasattr(config_file.cogito, "server")
            and hasattr(config_file.cogito.server, "version")
        ):
            click.echo(f"Server version: {config_file.cogito.server.version}")

        # Display the latest available config version
        latest_version = ConfigFile.latest_config_version()
        if latest_version > config_file.config_version:
            click.echo(
                f"Latest available config version: {latest_version} (upgrade available)"
            )
        else:
            click.echo("Your configuration is up to date.")

    except ConfigFileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo(f"Configuration file not found at: {ctx['config_path']}", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@config.command()
@click.option(
    "--backup/--no-backup", default=True, help="Create a backup before upgrading"
)
@click.pass_obj
def upgrade(ctx: click.Context, backup: bool = True) -> None:
    """Upgrade the configuration file to the latest version if an upgrade is available."""
    try:
        # Load the configuration file
        config_file = ConfigFile.load_from_file(ctx["config_path"])
        current_version = config_file.config_version
        latest_version = ConfigFile.latest_config_version()

        # Check if upgrade is needed
        if current_version >= latest_version:
            click.echo("Configuration is already at the latest version.")
            return

        click.echo(
            f"Upgrading configuration from version {current_version} to {latest_version}"
        )

        # Create backup if requested
        if backup:
            import shutil
            from pathlib import Path
            import datetime

            backup_path = f"{ctx['config_path']}.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
            shutil.copy2(ctx["config_path"], backup_path)
            click.echo(f"Backup created at: {backup_path}")

        # Perform the upgrade
        config_file.upgrade()

        # Save the upgraded configuration
        config_file.save_to_file(ctx["config_path"])

        click.echo(f"Configuration successfully upgraded to version {latest_version}")

    except ConfigFileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo(f"Configuration file not found at: {ctx['config_path']}", err=True)
    except Exception as e:
        click.echo(f"Error upgrading configuration: {e}", err=True)

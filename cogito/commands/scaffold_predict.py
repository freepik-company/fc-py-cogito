import os

import click
from jinja2 import Environment, FileSystemLoader

from cogito.core.config import ConfigFile
from cogito.core.exceptions import ConfigFileNotFoundError


def scaffold_predict_classes(config: ConfigFile, force: bool = False) -> None:
    template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("predict_class_template.jinja2")

    # TODO: validation of the config file?

    # Group routes by file
    files = {}
    route = config.cogito.server.route

    file_name = f'{route.predictor.split(":")[0]}.py'
    class_name = route.predictor.split(":")[1]
    class_data = route

    if file_name not in files:
        files[file_name] = []

    files[file_name].append({"class_name": class_name, "class_data": class_data})

    # Create the files
    for file, routes in files.items():
        class_names = ", ".join([route["class_name"] for route in routes])

        click.echo(f"Creating a scaffold predict classes ({class_names}) in {file}...")

        if os.path.exists(file) and not force:
            click.echo(f"File {file} already exists. Use --force to overwrite.")
            continue

        rendered_content = template.render(file=files, routes=routes)
        with open(file, "w") as f:
            f.write(rendered_content)

    pass


@click.command()
@click.option(
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Force overwrite of existing files",
)
@click.pass_context
def scaffold(ctx, force: bool = False) -> None:
    """Generate predict classes"""

    config_path = ctx.obj.get("config_path", ".") if ctx.obj else "."

    click.echo("Generating predict classes...")

    try:
        config = ConfigFile.load_from_file(f"{config_path}")
    except ConfigFileNotFoundError:
        click.echo("No configuration file found. Please initialize the project first.")
        return

    scaffold_predict_classes(config, force)

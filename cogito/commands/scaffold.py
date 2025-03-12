import os

import click
from jinja2 import Environment, FileSystemLoader

from cogito.core.config.file import ConfigFile
from cogito.core.exceptions import ConfigFileNotFoundError


def scaffold_predict_classes(config: ConfigFile, force: bool = False) -> None:
    template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("predict_class_template.jinja2")

    # TODO: validation of the config file?

    # Group routes by file
    files = {}
    predictor = config.cogito.predictor
    route = config.cogito.server.route

    file_name = f'{predictor.split(":")[0]}.py'
    class_name = predictor.split(":")[1]
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


def scaffold_train_classes(config: ConfigFile, force: bool = False) -> bool:
    template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("train_class_template.jinja2")

    # TODO: validation of the config file?

    # Group routes by file
    files = {}
    # Check if trainer is defined in config, default to 'train:Trainer' if not found
    trainer = getattr(config.cogito, "trainer", "train:Trainer")
    route = config.cogito.server.route

    if not trainer:
        click.echo(
            "No trainer defined in config. Please define a trainer in the config file."
        )
        return False

    file_name = f'{trainer.split(":")[0]}.py'
    class_name = trainer.split(":")[1]
    class_data = route

    if file_name not in files:
        files[file_name] = []

    files[file_name].append({"class_name": class_name, "class_data": class_data})

    # Create the files
    for file, routes in files.items():
        class_names = ", ".join([route["class_name"] for route in routes])

        click.echo(f"Creating a scaffold train classes ({class_names}) in {file}...")

        if os.path.exists(file) and not force:
            click.echo(f"File {file} already exists. Use --force to overwrite.")
            continue

        rendered_content = template.render(file=files, routes=routes)
        with open(file, "w") as f:
            f.write(rendered_content)

    return True


@click.command()
@click.option(
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Force overwrite of existing files",
)
@click.option(
    "--predict/--no-predict",
    is_flag=True,
    default=True,
    help="Generate predict classes",
)
@click.option(
    "--train/--no-train",
    is_flag=True,
    default=False,
    help="Generate train classes",
)
@click.pass_context
def scaffold(
    ctx: click.Context, force: bool = False, predict: bool = True, train: bool = False
) -> None:
    """Generate predict and/or train classes"""

    config_path = ctx.obj.get("config_path", ".") if ctx.obj else "."

    if not predict and not train:
        click.echo("Error: You must specify at least one of --predict or --train")
        exit(1)

    try:
        config = ConfigFile.load_from_file(f"{config_path}")
    except ConfigFileNotFoundError:
        click.echo("No configuration file found. Please initialize the project first.")
        exit(1)

    if predict:
        click.echo("Generating predict classes...")
        scaffold_predict_classes(config, force)

    if train:
        click.echo("Generating train classes...")
        if not scaffold_train_classes(config, force):
            exit(1)

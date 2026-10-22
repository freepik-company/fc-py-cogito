import click

from cogito.commands.scaffold import scaffold_predict_classes
from cogito.core.config.file import CONFIG_VERSION_MAP, ConfigFile
from cogito.core.config.v2 import PredictorConfig


def _latest_config_version() -> int:
    return max(CONFIG_VERSION_MAP.keys())


def _init_with_default() -> ConfigFile:
    version = _latest_config_version()
    config_class = ConfigFile._get_config_class(version)
    return ConfigFile(cogito=config_class.default(), config_version=version)


def _validate_class_path(value: str) -> str:
    module, _, class_name = value.partition(":")
    if not module or not class_name:
        raise click.BadParameter(
            "Use the 'module:Class' format, e.g. predict:Predictor"
        )
    return value


def _prompt_unique(text: str, default: str, used: set) -> str:
    while True:
        value = click.prompt(text, type=str, default=default, show_default=True)
        if value not in used:
            return value
        click.echo(f"'{value}' is already used by another predictor.")


def _prompt_predictors(route_config_class) -> list:
    predictors = []
    names = set()
    paths = set()

    while True:
        first = not predictors
        name = _prompt_unique("Predictor name", "predictor" if first else None, names)
        class_path = click.prompt(
            "Predictor class (module:Class)",
            type=str,
            default="predict:Predictor" if first else f"predict_{name}:Predictor",
            show_default=True,
            value_proc=_validate_class_path,
        )
        path = _prompt_unique(
            "Route path", "/v1/predict" if first else f"/v1/{name}", paths
        )
        threads = click.prompt(
            "Number of threads",
            type=click.IntRange(min=1),
            default=1,
            show_default=True,
        )

        if first:
            route = route_config_class.default()
            route.path = path
        else:
            route = route_config_class(
                name=name,
                description="Make a single prediction",
                path=path,
                tags=[name],
            )

        predictors.append(
            PredictorConfig(
                name=name, class_path=class_path, route=route, threads=threads
            )
        )
        names.add(name)
        paths.add(path)

        if not click.confirm(
            "Would you like to add another predictor?",
            default=False,
            show_default=True,
        ):
            return predictors


def _init_prompted() -> ConfigFile:
    config_version = _latest_config_version()
    config_class = ConfigFile._get_config_class(config_version)

    # Create instances of the necessary configuration classes
    # Instead of calling static methods, we should instantiate the classes directly
    # or use the appropriate factory methods provided by the config class

    click.echo(
        "Please provide the following information to initialize the project configuration:"
    )
    name = click.prompt(
        "Project name", type=str, default="Cogito ergo sum", show_default=True
    )
    description = click.prompt(
        "Project description", type=str, default=None, show_default=True
    )
    version = click.prompt(
        "Project version", type=str, default="1.0.0", show_default=True
    )

    click.echo("Nice! Now let's configure the FastAPI settings:")

    host = click.prompt("Host", type=str, default="0.0.0.0", show_default=True)
    port = click.prompt("Port", type=int, default=8000, show_default=True)
    debug = click.confirm(
        "Would you like to run the API server in DEBUG mode?",
        default=False,
        show_default=True,
    )
    access_log = click.confirm(
        "Would you like to enable access logs?", default=False, show_default=True
    )

    # Create FastAPI config using the appropriate constructor or factory method
    fastapi_config_class = config_class.get_fastapi_config_class()
    fastapi = fastapi_config_class(
        host=host,
        port=port,
        debug=debug,
        access_log=access_log,
    )

    click.echo("This starts to look like an amazing inference service!")

    click.echo("Now let's declare the predictors served by this instance:")

    predictors = _prompt_predictors(config_class.get_route_config_class())

    cache_dir = click.prompt(
        "Cache directory for model weights and artifacts",
        type=str,
        default="/tmp",
        show_default=True,
    )

    readiness_file = click.prompt(
        "Readiness file for health check",
        type=str,
        default="$HOME/readiness.lock",
        show_default=True,
    )

    # Create Server config
    server_config_class = config_class.get_server_config_class()
    server = server_config_class(
        name=name,
        description=description,
        version=version,
        fastapi=fastapi,
        cache_dir=cache_dir,
        threads=1,
        readiness_file=readiness_file,
    )

    click.echo("Almost there! Let's configure the training settings.")

    # todo add training settings, when defined

    click.echo("Great! We're all set.")

    # Create Cogito config
    cogito_config_class = config_class.get_cogito_config_class()
    cogito = cogito_config_class(
        server=server,
        trainer="train:Trainer",
        predictors=predictors,
    )

    return ConfigFile(cogito=cogito, config_version=config_version)


@click.command()
@click.option(
    "-s",
    "--scaffold",
    is_flag=True,
    default=False,
    help="Create a scaffold predict class for each predictor",
)
@click.option(
    "-d",
    "--default",
    is_flag=True,
    default=False,
    help="Initialize with default values",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Force initialization, even if already initialized",
)
@click.pass_context
def init(
    ctx: click.Context,
    scaffold: bool = False,
    default: bool = False,
    force: bool = False,
) -> None:
    """Initialize the project configuration"""
    config_path = (
        ctx.obj.get("config_path", "./cogito.yaml") if ctx.obj else "./cogito.yaml"
    )
    click.echo("Initializing...")

    if ConfigFile.exists(f"{config_path}") and not force:
        click.echo("Already initialized.")
        return

    if default:
        config = _init_with_default()
    else:
        config = _init_prompted()

    if scaffold:
        scaffold_predict_classes(config, force)

    config.save_to_file(f"{config_path}")
    click.echo("Initialized successfully.")

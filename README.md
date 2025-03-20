# Freepik Company cogito

[![Test](https://github.com/freepik-company/fc-py-cogito/actions/workflows/pr-tests.yaml/badge.svg)](https://github.com/freepik-company/fc-py-cogito/actions/)
[![Publish](https://github.com/freepik-company/fc-py-cogito/actions/workflows/publish-to-pypi-and-test-pypi.yml/badge.svg)](https://github.com/freepik-company/fc-py-cogito/actions/)
[![PyPI version](https://img.shields.io/pypi/v/cogito.svg)](https://pypi.org/project/cogito/)
[![Downloads](https://img.shields.io/pypi/dm/cogito.svg)](https://pypi.org/project/cogito/)
[![License](https://img.shields.io/github/license/freepik-company/fc-py-cogito)](https://github.com/freepik-company/fc-py-cogito/blob/main/LICENSE)
[![Contribute](https://img.shields.io/badge/contribute-guidelines-blue)](https://github.com/freepik-company/fc-py-cogito/blob/main/CONTRIBUTING.md)


Cogito is a versatile Python framework and SDK aimed at simplifying the development and deployment of inference services. 
It allows users to wrap machine learning models or any computational logic into APIs effortlessly, while also providing 
a comprehensive library for programmatic access and a command-line interface for both inference and training operations.

With cogito, you can focus on your core algorithmic functionality while the framework takes care of the heavy lifting, 
including API structure, request handling, error management, and scalability. Cogito provides multiple ways to interact 
with your models:

1. **RESTful HTTP API** - Deploy your models as scalable web services
2. **Python SDK** - Integrate directly into your Python applications
3. **Command-line Interface** - Run predictions and training from the terminal

Key features include:
- **Ease of Use**: Simplifies the process of converting your models into production-ready APIs with minimal boilerplate code.
- **Customizable API**: Provides flexibility to define endpoints, input/output formats, and pre- / post-processing logic.
- **Scalability**: Optimized to handle high-throughput scenarios with support for modern server frameworks.
- **Extensibility**: Easily integrates with third-party libraries, monitoring tools, or cloud services.
- **Error Handling**: Built-in mechanisms to catch and handle runtime issues gracefully.
- **Training Integration**: Run and manage training processes directly through the command line.
- **Unified Workflow**: Consistent patterns for both development and production environments.


## Installation

### Using pip
You can install the package:
```sh
pip install cogito
```
---

## Usage Guide: Cogito CLI

The **Cogito CLI** provides several commands to initialize, scaffold, and run your inference-based projects.

# CLI Reference

- [Global Options](#global-options)
- [Commands](#commands)
  - [Initialize](#initialize)
  - [Scaffold](#scaffold)
  - [Run](#run)
  - [Config](#config)
  - [Version](#version)
  - [Train](#train)
  - [Predict](#predict)
  - [Help](#help)
---

## Global Options

These options can be used with any command:

- `-c, --config-path TEXT`: Path to the configuration file (default: ./cogito.yaml)
- `--help`: Show help message and exit

**Examples:**

1. Specify a custom configuration file:
   ```bash
   cogito-cli -c ./configs/my-config.yaml run
   ```

2. Get help for any command:
   ```bash
   cogito-cli init --help
   ```

---

### Initialize

Command: `init`

**Description:** Initialize the project configuration with default or custom settings.

#### Options:

- `-s, --scaffold`: Generate a scaffold prediction class during initialization.
- `-d, --default`: Initialize with default values without prompts.
- `-f, --force`: Force initialization even if a configuration file already exists.

#### Usage:

```bash
cogito-cli [-c config_path] init [OPTIONS]
```

**Examples:**

1. Initialize with prompts in the current directory:
   ```bash
   cogito-cli init
   ```

2. Initialize with default values:
   ```bash
   cogito-cli init --default
   ```

3. Initialize and scaffold prediction classes:
   ```bash
   cogito-cli init --scaffold
   ```

4. Force initialization over existing configuration:
   ```bash
   cogito-cli init --force
   ```

5. Initialize project in a specific directory:
   ```bash
   cogito-cli -c ./my-project/cogito.yaml init
   ```

**Note:** The init command supports the global `-c, --config-path` option to specify where the configuration file should be created.

---

### Scaffold

Command: `scaffold`

**Description:** Generate prediction and/or training class files based on the routes defined in the configuration file (`cogito.yaml`).

#### Options:

- `-f, --force`: Overwrite existing files if they already exist.
- `--predict/--no-predict`: Generate prediction classes (enabled by default).
- `--train/--no-train`: Generate training classes (disabled by default).

#### Usage:

```bash
cogito-cli [-c config_path] scaffold [OPTIONS]
```

**Examples:**

1. Scaffold prediction classes only (default behavior) using configuration in current directory:
   ```bash
   cogito-cli scaffold
   ```

2. Scaffold and overwrite existing files:
   ```bash
   cogito-cli scaffold --force
   ```

3. Scaffold training classes only:
   ```bash
   cogito-cli scaffold --no-predict --train
   ```

4. Scaffold both prediction and training classes:
   ```bash
   cogito-cli scaffold --predict --train
   ```

5. Scaffold using a configuration file in a specific directory:
   ```bash
   cogito-cli -c ./my-project/cogito.yaml scaffold
   ```

**Note:** The scaffold command uses the global `-c, --config-path` option to locate the configuration file that defines the classes to be generated.

---

### Run

Command: `run`

**Description:** Run the cogito application based on the configuration file.

#### Usage:

```bash
cogito-cli [-c config_path] run
```

**Examples:**

1. Run the cogito application using the default configuration file in the current directory:
   ```bash
   cogito-cli run
   ```

2. Run the cogito application using a specific configuration file:
   ```bash
   cogito-cli -c ./examples/cogito.yaml run
   ```

3. Run the cogito application from a specific directory:
   ```bash
   cogito-cli -c ./my-project/cogito.yaml run
   ```

**Behavior:**
- The command will look for the specified configuration file (defaults to ./cogito.yaml if not provided)
- The directory of the configuration file will be added to the Python path
- Errors during initialization or execution will be printed to stderr with a traceback

**Note:** The run command uses the global `-c, --config-path` option to specify the configuration file location.

---

### Config

Command: `config`

**Description:** Manage configuration settings for Cogito projects.

#### Subcommands:

- `version`: Display the current configuration version and check for updates.
- `upgrade`: Upgrade the configuration file to the latest version.

#### Usage:

```bash
cogito-cli [-c config_path] config [SUBCOMMAND] [OPTIONS]
```

#### Subcommand: `version`

**Description:** Show the current configuration version and check if updates are available.

**Usage:**
```bash
cogito-cli [-c config_path] config version
```

**Example:**
```bash
$ cogito-cli config version
Configuration version: 1.0
Server version: 2.1
Latest available config version: 1.2 (upgrade available)
```

#### Subcommand: `upgrade`

**Description:** Upgrade the configuration file to the latest available version.

**Options:**
- `--backup/--no-backup`: Create a backup before upgrading (default: --backup)

**Usage:**
```bash
cogito-cli [-c config_path] config upgrade [OPTIONS]
```

**Examples:**

1. Upgrade configuration with automatic backup:
   ```bash
   cogito-cli config upgrade
   ```

2. Upgrade without creating a backup:
   ```bash
   cogito-cli config upgrade --no-backup
   ```

3. Upgrade a specific configuration file:
   ```bash
   cogito-cli -c ./my-project/cogito.yaml config upgrade
   ```

**Behavior:**
- Checks if an upgrade is available
- Creates a timestamped backup of the original configuration file (unless --no-backup is specified)
- Performs the upgrade
- Saves the upgraded configuration to the original file

**Note:** The config command uses the global `-c, --config-path` option to specify which configuration file to manage.

---

### Version

Command: `version`

**Description:** Show the current version of the Cogito package.

#### Usage:

```bash
cogito-cli version
```

**Example:**

```bash
$ cogito-cli version
Version: 1.2.3
```

**Behavior:**
- Displays the current version of the Cogito package
- This version information comes from the package's `__version__` attribute
- The version command doesn't require or use a configuration file

**Note:** Unlike most other commands, the version command doesn't use the `-c, --config-path` option as it's checking the installed package version, not a specific project.

---

### Train

Command: `train`

**Description:** Run a training process using a Cogito Trainer class defined in your project.

#### Options:

- `--payload TEXT`: Required. JSON payload containing the training data and parameters.

#### Usage:

```bash
cogito-cli [-c config_path] train --payload JSON_STRING
```

**Examples:**

1. Run training with a simple payload:
   ```bash
   cogito-cli train --payload '{"data": [1, 2, 3], "epochs": 10}'
   ```

2. Run training with a specific configuration file:
   ```bash
   cogito-cli -c ./my-project/cogito.yaml train --payload '{"dataset": "images", "batch_size": 32}'
   ```

3. Load payload from a file using command substitution:
   ```bash
   cogito-cli train --payload "$(cat training_config.json)"
   ```

**Behavior:**
- Loads the configuration file specified by `-c` or uses the default path
- Parses the JSON payload provided in the `--payload` option
- Initializes a Trainer instance based on the configuration
- Calls the `setup()` method if available (warns if not implemented)
- Runs the training process via the `run()` method
- Prints the result to stdout
- Handles and reports errors that occur during setup or execution

**Note:** 
- The train command requires a properly configured Cogito project with a trainer class defined
- The payload structure depends on your specific trainer implementation
- Training output will be printed to the console, which you can redirect to a file if needed

---

### Predict

Command: `predict`

**Description:** Run a prediction using a Cogito Predict class defined in your project.

#### Options:

- `--payload TEXT`: Required. JSON payload containing the input data for prediction.

#### Usage:

```bash
cogito-cli [-c config_path] predict --payload JSON_STRING
```

**Examples:**

1. Run prediction with a simple payload:
   ```bash
   cogito-cli predict --payload '{"input": "image.jpg"}'
   ```

2. Run prediction with a specific configuration file:
   ```bash
   cogito-cli -c ./my-project/cogito.yaml predict --payload '{"text": "Analyze this sentence"}'
   ```

3. Load payload from a file using command substitution:
   ```bash
   cogito-cli predict --payload "$(cat input_data.json)"
   ```

**Behavior:**
- Loads the configuration file specified by `-c` or uses the default path
- Parses the JSON payload provided in the `--payload` option
- Initializes a Predict instance based on the configuration
- Calls the `setup()` method if available (warns if not implemented)
- Executes the prediction via the `run()` method with the provided payload
- Prints the prediction result to stdout as formatted JSON (with 4-space indentation)
- Handles and reports errors that occur during initialization, setup, or execution

**Note:** 
- The predict command requires a properly configured Cogito project with a predictor class defined
- The payload structure depends on your specific predictor implementation
- The output is formatted as indented JSON for better readability
- To process the output programmatically, you can pipe the result to tools like `jq`

---

### Help

Command: `help` or `--help`

**Description:** Display help information about Cogito CLI commands and options.

#### Usage:

```bash
cogito-cli help
cogito-cli --help
cogito-cli [COMMAND] --help
```

**Examples:**

1. Display general help information:
   ```bash
   cogito-cli help
   ```

2. Alternative way to display general help:
   ```bash
   cogito-cli --help
   ```

3. Get help for a specific command:
   ```bash
   cogito-cli init --help
   ```

4. Get help for a subcommand:
   ```bash
   cogito-cli config upgrade --help
   ```

**Behavior:**
- Displays a list of available commands when used without arguments
- Shows detailed help information for a specific command when used with a command name
- Includes information about available options, arguments, and basic usage examples
- Can be used with any command or subcommand to get context-specific help

**Note:** 
- The help command is one of the most useful tools for learning how to use Cogito CLI
- Using `--help` with any command is a good way to understand its options and usage
- Help output is automatically generated based on the command documentation

---

## Development

This section covers how to set up and work with the Cogito codebase for development purposes.

### Build the local development environment

```sh
make build
```

### Development Environment Commands

The project includes various make targets to simplify development workflows:

#### Environment Setup

- `make build` - Create a virtual environment and install dependencies
- `make build-dev` - Build the development environment with additional dev dependencies
- `make .venv` - Create only the virtual environment
- `make clean` - Remove build artifacts
- `make mr-proper` - Clean the project completely including the virtual environment

#### Code Quality

- `make code-style-check` - Check code style with Black without making changes
- `make code-style-dirty` - Apply Black code formatting without committing changes
- `make pre-commit-install` - Install pre-commit hooks
- `make pre-commit-tests` - Run pre-commit hooks for tests
- `make pre-commit-black` - Run pre-commit hooks for Black formatting

#### Dependencies Management

- `make dependencies-compile` - Compile dependencies to requirements.txt
- `make dependencies-install` - Install the dependencies
- `make dependencies-dev-install` - Install the development dependencies

#### Testing

- `make run-test` - Run the test suite

#### Installation & Publishing

- `make install` - Install the package in development mode
- `make dist` - Build the distribution package
- `make upload` - Upload the distribution to the repository (default: testpypi)
- `make alpha` - Bump the version to alpha
- `make beta` - Bump the version to beta
- `make patch` - Bump the version to patch and update changelog
- `make minor` - Bump the version to minor and update changelog
- `make major` - Bump the version to major and update changelog

#### Git Workflow

- `make git-prune` - Clean up remote tracking branches that no longer exist

### Variables

The following variables can be customized when running make commands:

- `PYTHON_VERSION` - Python version to use (default: 3.10)
- `REPOSITORY` - Repository to upload the package to (default: testpypi)
- `BUMP_INCREMENT` - Version increment for alpha/beta releases (default: MINOR)

Examples:

```sh
# Build with a specific Python version
make build PYTHON_VERSION=3.9

# Upload to PyPI instead of TestPyPI
make upload REPOSITORY=pypi

# Create a major alpha release
make alpha BUMP_INCREMENT=MAJOR
```

For a complete list of available commands, run:

```sh
make help
```

### Versioning Strategy

We use [semantic versioning](https://semver.org/) and a milestone-based approach:

#### Alpha and Beta Releases

For feature development and testing, use alpha and beta releases:

```bash
# Create an alpha release with MINOR version bump
make alpha BUMP_INCREMENT=MINOR

# Create a beta release with PATCH version bump (default)
make beta

# Create a beta release with MAJOR version bump
make beta BUMP_INCREMENT=MAJOR
```

### Building and Publishing

The CI pipeline handles building and publishing, but you can test locally:

```bash
# Build the distribution
make dist

# Upload to TestPyPI (default)
make upload

# Upload to PyPI
make upload REPOSITORY=pypi
```

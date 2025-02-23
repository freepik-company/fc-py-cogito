# Freepik Company cogito

Cogito is a versatile Python module aimed at simplifying the development and deployment of inference services. 
It allows users to wrap machine learning models or any computational logic into APIs effortlessly. 
With cogito, you can focus on your core algorithmic functionality while the module takes care of the heavy lifting, 
including API structure, request handling, error management, and scalability.

Key features include:
- Ease of Use: Simplifies the process of converting your models into production-ready APIs with minimal boilerplate code.
- Customizable API: Provides flexibility to define endpoints, input/output formats, and pre- / post-processing logic.
- Scalability: Optimized to handle high-throughput scenarios with support for modern server frameworks.
- Extensibility: Easily integrates with third-party libraries, monitoring tools, or cloud services.
- Error Handling: Built-in mechanisms to catch and handle runtime issues gracefully.

## Development

### Build the local development environment

```sh
make build
```

## Installation

### Using pip
Then, you can install the package:
```sh
pip install cogito
```

## Run example application

```sh
cd examples
python app.py
```

## Usage Guide: Cogito CLI

The **Cogito CLI** provides several commands to initialize, scaffold, and run your inference-based projects.

# CLI Reference

- [Commands](#commands)
  - [Initialize](#initialize)
  - [Scaffold](#scaffold)
  - [Run](#run)
  - [Version](#version)
  - [Help](#help)
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
cogito-cli init [OPTIONS]
```

**Examples:**

1. Initialize with prompts:
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

---

### Scaffold

Command: `scaffold`

**Description:** Generate prediction class files based on the routes defined in the configuration file (`cogito.yaml`).

#### Options:

- `-f, --force`: Overwrite existing files if they already exist.

#### Usage:

```bash
cogito-cli scaffold [OPTIONS]
```

**Examples:**

1. Scaffold prediction classes:
   ```bash
   cogito-cli scaffold
   ```

2. Scaffold and overwrite existing files:
   ```bash
   cogito-cli scaffold --force
   ```

---

### Run

Command: `run`

**Description:** Run the cogito application based on the configuration in the specified directory.

#### Usage:

```bash
cogito-cli [-c context] run
```

**Example:**

1. Run the cogito application located in `examples` directory:
   ```bash
   cogito-cli -c examples run
   ```

This will:
- Change the current working directory to the configuration path.
- Load the application based on the `cogito.yaml` file.
- Start the FastAPI server for your inference service.

### Version

Command: `version`

**Description:** Show the current version of Cogito.

#### Usage:

```bash
cogito-cli version
```

**Example:**

```bash
cogito-cli version
# Cogito version X.Y.Z
```

## Help

Command: `help`

**Description:** Show the help message for the cogito-cli.

#### Usage:

```bash
cogito-cli help
# Show the help message for the cogito-cli.
```

# Freepik Company inferia

Inferia is a versatile Python module aimed at simplifying the development and deployment of inference services. 
It allows users to wrap machine learning models or any computational logic into APIs effortlessly. 
With inferia, you can focus on your core algorithmic functionality while the module takes care of the heavy lifting, 
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
pip install inferia
```

## Run example application

```sh
cd examples
python app.py
```

## Usage Guide: Inferia CLI

The **Inferia CLI** provides several commands to initialize, scaffold, and run your inference-based projects. This guide explains the available commands and their options.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Commands](#commands)
   - [Initialize](#initialize)
   - [Scaffold](#scaffold)
   - [Run](#run)
4. [Examples](#examples)

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
python -m inferia init [OPTIONS]
```

**Examples:**

1. Initialize with prompts:
   ```bash
   python -m inferia init
   ```

2. Initialize with default values:
   ```bash
   python -m inferia init --default
   ```

3. Initialize and scaffold prediction classes:
   ```bash
   python -m inferia init --scaffold
   ```

---

### Scaffold

Command: `scaffold`

**Description:** Generate prediction class files based on the routes defined in the configuration file (`inferia.yaml`).

#### Options:

- `-f, --force`: Overwrite existing files if they already exist.

#### Usage:

```bash
python -m inferia scaffold [OPTIONS]
```

**Examples:**

1. Scaffold prediction classes:
   ```bash
   python -m inferia scaffold
   ```

2. Scaffold and overwrite existing files:
   ```bash
   python -m inferia scaffold --force
   ```

---

### Run

Command: `run`

**Description:** Run the Inferia application based on the configuration in the specified directory.

#### Usage:

```bash
python -m inferia run
```

**Example:**

1. Run the application:
   ```bash
   python -m inferia run
   ```

This will:
- Change the current working directory to the configuration path.
- Load the application based on the `inferia.yaml` file.
- Start the FastAPI server for your inference service.

---

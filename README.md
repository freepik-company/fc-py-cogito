# Freepik Company Cogito

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

## Index

- [Installation and Getting Started](#installation-and-getting-started)
- [CLI Reference](#cli-reference)
- [Development](#development)
- [Standard API Endpoints](#standard-api-endpoints)
- [SDK API Reference](#sdk-api-reference)

## Installation and Getting Started

### Index
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Initialize Your Project](#initialize-your-project)
  - [Developing a Prediction Class](#developing-a-prediction-class)
  - [Developing a Training Class](#developing-a-training-class-optional)
- [Using Cogito](#using-cogito)
  - [Command Line Interface](#using-the-command-line-interface)
    - [Running Predictions](#running-predictions)
    - [Running Training](#running-training-optional)
    - [Launching the RESTful API](#launching-the-restful-api)
  - [Example Workflow](#example-workflow)
  - [Containerization](#dockerfile)
  - [SDK Integration](#integrate-directly-into-your-python-applications)

### Installation

#### Using pip
You can install the package:
```sh
pip install cogito
```

## Getting Started

### Initialize Your Project

Start by initializing a new Cogito project in your directory:

```bash
# Interactive initialization with prompts
cogito-cli init

# Initialize with default values
cogito-cli init --default

# Initialize and scaffold prediction classes
cogito-cli init --scaffold
```

This will create a `cogito.yaml` configuration file in your directory with the necessary settings to get started.

### Developing a Prediction Class

Create a prediction class by extending `BasePredictor`. Here's a basic example:

```python
from pydantic import BaseModel, Field
from cogito import BasePredictor

class PredictResponse(BaseModel):
    result: str
    score: float

class Predictor(BasePredictor):
    def setup(self):
        # Initialize your model and resources here
        self.model = None  # Your model initialization
        print("Model loaded and ready!")
    
    def predict(
        self,
        input_text: str,
        threshold: float = Field(0.5, gt=0.0, lt=1.0),
    ) -> PredictResponse:
        # Your prediction logic here
        return PredictResponse(
            result="Prediction result",
            score=0.95
        )
```

Save this file in your project directory according to the path specified in your `cogito.yaml` file.

### Developing a Training Class (Optional)

For model training capabilities, extend the `BaseTrainer` class:

```python
from cogito.core.models import BaseTrainer

class Trainer(BaseTrainer):
    def setup(self):
        # Initialize training resources
        self.training_config = {}
    
    def train(
        self,
        dataset_path: str,
        epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        # Your training logic here
        print(f"Training model with {epochs} epochs")
        
        # Return training results or metrics
        return {
            "status": "success",
            "accuracy": 0.92,
            "loss": 0.08
        }
```

### Using the Command Line Interface

#### Running Predictions

Once your prediction class is set up, you can use the CLI to run predictions:

```bash
# Run a prediction with a JSON payload
cogito-cli predict --payload '{"input_text": "sample text", "threshold": 0.7}'

# Using a payload from a file
cogito-cli predict --payload "$(cat input_data.json)"
```

#### Running Training (Optional)

If you've implemented a training class:

```bash
# Run training with a JSON payload
cogito-cli train --payload '{"dataset_path": "data/training.csv", "epochs": 20, "learning_rate": 0.001}'

# Using a payload from a file
cogito-cli train --payload "$(cat training_config.json)"
```

#### Launching the RESTful API

To deploy your model as a RESTful API service:

```bash
# Start the API server using configuration in the current directory
cogito-cli run

# Start the API server with a specific configuration file
cogito-cli -c ./my-project/cogito.yaml run
```

By default, this will start a server on localhost that exposes your prediction functionality as API endpoints. You can then send HTTP requests to interact with your model.

### Example Workflow

1. Initialize a new project:
   ```bash
   mkdir my-inference-service
   cd my-inference-service
   cogito-cli init --scaffold
   ```

2. Implement your prediction and/or training logic in the scaffolded files

3. Test your implementation using the CLI:
   ```bash
   cogito-cli predict --payload '{"input_text": "test"}'
   ```

4. Run as an API:
   ```bash
   cogito-cli run
   ```

5. Send requests to your API:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"input_text": "test", "threshold": 0.7}'
   ```

### Dockerfile

To deploy your Cogito application in a containerized environment, you can use the following Dockerfile example:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install cogito
RUN pip install --no-cache-dir cogito

# Copy the current directory contents into the container
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variables if needed
ENV MODEL_PATH=/app/models/model.pkl
ENV LOG_LEVEL=INFO

# Run the Cogito application when the container launches
CMD ["cogito-cli", "run"]
```

#### Building and Running the Docker Container

1. Build the Docker image:
   ```bash
   docker build -t my-cogito-app .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 my-cogito-app
   ```

3. For production deployments, you might want to add health checks and configure environment variables:
   ```bash
   docker run -p 8000:8000 \
     -e MODEL_PATH=/app/models/custom_model.pkl \
     -e LOG_LEVEL=WARNING \
     --health-cmd="curl -f http://localhost:8000/health || exit 1" \
     --health-interval=30s \
     my-cogito-app
   ```

#### Docker Compose Example

For more complex setups, you can use Docker Compose:

```yaml
version: '3.8'

services:
  cogito-app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.pkl
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
```

Run with:
```bash
docker-compose up -d
```

This containerization approach ensures your Cogito application is deployable in any environment that supports Docker, from local development to cloud platforms.

### Integrate Directly into Your Python Applications

In addition to using Cogito as a CLI tool or API service, you can integrate it directly into your Python applications using the SDK interface. This allows you to leverage your Cogito models programmatically without the overhead of HTTP requests.

#### Basic SDK Usage

```python
from cogito.lib.prediction import Predict
import json

# Initialize the predictor with your configuration
predictor = Predict("./cogito.yaml")

# Setup the predictor (loads models, etc.)
predictor.setup()

# Create a payload
payload = {
    "input_text": "This is a sample text to analyze",
    "threshold": 0.7
}

# Run prediction
result = predictor.run(payload)

# Process the result
print(json.dumps(result, indent=2))
```

#### Training with the SDK

If you've implemented a training class, you can use it programmatically as well:

```python
from cogito.lib.training import Trainer

# Initialize the trainer with your configuration
trainer = Trainer("./cogito.yaml")

# Setup the trainer
trainer.setup()

# Create training parameters
training_params = {
    "dataset_path": "data/training.csv",
    "epochs": 20,
    "learning_rate": 0.001,
    "batch_size": 32
}

# Run training
result = trainer.run(training_params)

# Process the training result
print(f"Training completed with results: {result}")
```

#### Integration Example: Web Application

Here's how you might integrate Cogito into a Flask application:

```python
from flask import Flask, request, jsonify
from cogito.lib.prediction import Predict

app = Flask(__name__)

# Initialize predictor at startup
predictor = Predict("./cogito.yaml")
predictor.setup()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    
    try:
        # Use the Cogito predictor
        result = predictor.run(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

#### Batch Processing Example

For processing multiple items efficiently:

```python
from cogito.lib.prediction import Predict
import pandas as pd
from tqdm import tqdm

# Initialize predictor
predictor = Predict("./cogito.yaml")
predictor.setup()

# Load data for batch processing
df = pd.read_csv("input_data.csv")

# Process each row
results = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    payload = {
        "input_text": row['text'],
        "threshold": 0.5
    }
    
    # Run prediction and store result
    result = predictor.run(payload)
    results.append(result)

# Create a results dataframe
results_df = pd.DataFrame(results)
results_df.to_csv("prediction_results.csv", index=False)
```

#### Checking Version

You can also check the installed Cogito version programmatically:

```python
from cogito.lib.version import get_version

print(f"Using Cogito version: {get_version()}")
```

By using the SDK interface, you can seamlessly integrate your Cogito models into larger applications, batch processing workflows, or custom environments where a direct API call might not be optimal.

---

## Usage Guide: Cogito CLI

The **Cogito CLI** provides several commands to initialize, scaffold, and run your inference-based projects.

## CLI Reference

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

### Global Options

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

### Standard API Endpoints

When you deploy a Cogito application as a RESTful API, several standard endpoints are automatically included, regardless of your custom prediction endpoints:

#### Index
- [Health Check Endpoint](#health-check-endpoint)
  - [Kubernetes Integration](#kubernetes-integration)
  - [Using the Readiness File](#using-the-readiness-file)
- [Metrics Endpoint](#metrics-endpoint)
- [Version Endpoint](#version-endpoint)

#### Health Check Endpoint

- **URL**: `/health`
- **Method**: `GET`
- **Description**: Provides a simple health check mechanism to verify that the service is up and running.
- **Response**: 
  ```json
  {
    "status": "OK"
  }
  ```
- **Usage**: Commonly used by container orchestration systems (like Kubernetes) for liveness and readiness probes.

##### Kubernetes Integration

The health endpoint can be used to configure Kubernetes probes for proper container lifecycle management:

1. **Liveness Probe**: Determines if the application is running properly. If it fails, Kubernetes will restart the container.

   a) **HTTP Endpoint Method**: Uses the `/health` endpoint which checks the `readiness_file` internally:

   ```yaml
   livenessProbe:
   httpGet:
      path: /health
      port: 8000
   initialDelaySeconds: 30
   periodSeconds: 10
   timeoutSeconds: 5
   failureThreshold: 3
   ```

   b) **File Existence Method**: Checks for the existence of the `readiness_file` directly:

   ```yaml
   livenessProbe:
     exec:
       command:
       - test
       - -f
       - /tmp/cogito-readiness.lock  # Must match readiness_file in cogito.yaml
     initialDelaySeconds: 5
     periodSeconds: 5
   ```

2. **Readiness Probe**: Determines if the container is ready to receive traffic. There are two approaches you can use:

   a) **HTTP Endpoint Method**: Uses the `/health` endpoint which checks the `readiness_file` internally:

   ```yaml
   readinessProbe:
     httpGet:
       path: /health
       port: 8000
     initialDelaySeconds: 5
     periodSeconds: 5
   ```

   b) **File Existence Method**: Checks for the existence of the `readiness_file` directly:

   ```yaml
   readinessProbe:
     exec:
       command:
       - test
       - -f
       - /tmp/cogito-readiness.lock  # Must match readiness_file in cogito.yaml
     initialDelaySeconds: 5
     periodSeconds: 5
   ```

##### Using the Readiness File

The `readiness_file` parameter in your `cogito.yaml` (e.g., `/tmp/cogito-readiness.lock`) provides an additional mechanism to control when your service is considered ready:

**How it works**: 
   - When your Cogito application starts, it checks if this file exists
   - If the file exists, the health endpoint returns a 200 OK response
   - If the file doesn't exist, it returns a 503 Service Unavailable response
   - Kubernetes can check this file directly or through the health endpoint

**Choosing between methods**:
   - **HTTP endpoint method**: Provides more information (status codes, potential error messages) and follows standard HTTP patterns
   - **File existence method**: Slightly more efficient as it doesn't require an HTTP call and works even if the application is temporarily unable to respond to HTTP requests

This approach ensures that traffic is only directed to your service when it's fully ready to handle requests, preventing errors during startup or maintenance periods.

#### Metrics Endpoint

- **URL**: `/metrics`
- **Method**: `GET`
- **Description**: Exposes Prometheus-compatible metrics about the service's performance and usage.
- **Response**: Plain text in Prometheus exposition format
- **Usage**: Can be scraped by Prometheus to monitor application metrics like request count, latency, and resource usage.

#### Version Endpoint

- **URL**: `/version`
- **Method**: `GET`
- **Description**: Returns the current version of the deployed Cogito application.
- **Response**:
  ```json
  {
    "version": "1.2.3"
  }
  ```
- **Usage**: Helpful for verifying which version is currently deployed, especially in multi-environment setups.

## SDK API Reference

This section documents the core classes and methods available in the Cogito SDK for programmatic integration.

### Index
- [Prediction Module](#prediction-module)
  - [Predict Class](#predict-class)
- [Training Module](#training-module)
  - [Trainer Class](#trainer-class)
- [Version Module](#version-module)
  - [get_version Function](#get_version-function)

### Prediction Module

The prediction module (`cogito.lib.prediction`) provides functionality for making predictions using your Cogito models programmatically.

#### Predict Class

The `Predict` class is the main interface for making predictions with your models.

**Import**:
```python
from cogito.lib.prediction import Predict
```

**Constructor**:
```python
Predict(config_path: str)
```
- `config_path`: Path to the cogito.yaml configuration file

**Methods**:

- `setup()` → `None`
  - Initializes the predictor by calling the setup method of your predictor class
  - This method should be called before making predictions
  - Raises: `NoSetupMethodError` if the predictor class doesn't implement a setup method
  - Raises: `Exception` if any error occurs during setup

- `run(payload_data: dict)` → `dict`
  - Executes a prediction using the provided payload data
  - Parameters:
    - `payload_data`: A dictionary containing the input data for prediction, matching the parameters of your `predict()` method
  - Returns: A dictionary containing the prediction results
  - Raises: `Exception` if any error occurs during prediction

**Example Usage**:
```python
from cogito.lib.prediction import Predict
import json

# Initialize predictor with configuration file
predictor = Predict("./cogito.yaml")

# Set up the predictor (load models, etc.)
predictor.setup()

# Prepare input data
data = {
    "input_text": "Sample text for prediction",
    "threshold": 0.7
}

# Run prediction
try:
    result = predictor.run(data)
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Prediction failed: {e}")
```

### Training Module

The training module (`cogito.lib.training`) provides functionality for training models using your Cogito trainers programmatically.

#### Trainer Class

The `Trainer` class is the main interface for training models.

**Import**:
```python
from cogito.lib.training import Trainer
```

**Constructor**:
```python
Trainer(config_path: str)
```
- `config_path`: Path to the cogito.yaml configuration file

**Methods**:

- `setup()` → `None`
  - Initializes the trainer by calling the setup method of your trainer class
  - This method should be called before training
  - Raises: `NoSetupMethodError` if the trainer class doesn't implement a setup method
  - Raises: `Exception` if any error occurs during setup

- `run(payload_data: dict, run_setup: bool = True)` → `dict`
  - Executes the training process using the provided payload data
  - Parameters:
    - `payload_data`: A dictionary containing the training parameters, matching the parameters of your `train()` method
    - `run_setup`: Boolean flag indicating whether to run setup automatically (default: True)
  - Returns: A dictionary containing the training results or metrics
  - Raises: `Exception` if any error occurs during training

**Example Usage**:
```python
from cogito.lib.training import Trainer

# Initialize trainer with configuration file
trainer = Trainer("./cogito.yaml")

# Set up the trainer (initialize resources)
trainer.setup()

# Prepare training parameters
training_params = {
    "dataset_path": "data/training_data.csv",
    "epochs": 20,
    "learning_rate": 0.001,
    "batch_size": 64
}

# Run training
try:
    result = trainer.run(training_params)
    print(f"Training completed with metrics: {result}")
except Exception as e:
    print(f"Training failed: {e}")
```

### Version Module

The version module (`cogito.lib.version`) provides functionality for retrieving the current version of Cogito.

#### get_version Function

The `get_version` function returns the current version of the Cogito package.

**Import**:
```python
from cogito.lib.version import get_version
```

**Signature**:
```python
get_version() -> str
```
- Returns: A string containing the current version of Cogito

**Example Usage**:
```python
from cogito.lib.version import get_version

version = get_version()
print(f"Using Cogito version: {version}")
```

This allows you to programmatically check which version of Cogito is being used in your application, which can be helpful for debugging or ensuring compatibility.

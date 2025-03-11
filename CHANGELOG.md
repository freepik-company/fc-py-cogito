## 0.3.0a0 (2025-03-11)

## 0.2.6a4 (2025-03-10)

### Feat

- **Makefile**: Enhance pre-commit hook targets
- **Makefile**: Enhance pre-commit hook installation

### Refactor

- Clean up code formatting and remove redundant lines

## 0.2.6a3 (2025-03-07)

### Feat

- Implements BadRequestError raising and 400 error code response (#94)

## 0.2.6a1 (2025-03-06)

## 0.2.6a0 (2025-02-27)

## 0.2.6 (2025-03-10)
## 0.2.6a4 (2025-03-10)

### Feat

- **Makefile**: Enhance pre-commit hook targets
- **Makefile**: Enhance pre-commit hook installation

### Refactor

- Clean up code formatting and remove redundant lines

## 0.2.6a3 (2025-03-07)

### Feat

- Implements BadRequestError raising and 400 error code response (#94)

## 0.2.6a1 (2025-02-27)

## 0.2.6a0 (2025-02-27)

## 0.2.5 (2025-02-21)

### Refactor

- hf hub version dependency removed (#85)

## 0.2.4 (2025-02-18)

### Feat

- adds version command (#81)

## 0.2.4a2 (2025-02-14)

### Fix

- **cogito.core.utils**: changes the default type from None to Any (#80)

## 0.2.4a1 (2025-02-14)

### Fix

- fix error with pydantic2 model_dump function (#78)

## 0.2.4a0 (2025-02-13)

### Fix

- **pyproject.toml**: avoid pydantic version requirements in pyproject (#77)
- changes the default value of readiness file (#75)
- Fixes predict function return type in jinja template (#74)

## 0.2.3 (2025-02-10)

### Feat

- Add readiness file support for health checks and application startup (#72)

### Refactor

- **config,template**: Simplify route configuration and update predict template (#70)

## 0.2.2 (2025-02-06)

### Fix

- **config**: remove hard-coded IP address and allow server to listen on any available IP (#67)

### Refactor

- **initialize,config**: Update project initialization and configuration defaults (#69)

## 0.2.1 (2025-02-04)

### Refactor

- **.github/workflows/publish-to-pypi-and-test-pypi.yml**: Update Sigstore Python action to version 3.0.0 (#64)
- move threads config to server node (#61)

## 0.2.0 (2025-02-04)

### BREAKING CHANGE

- Previous version of cogito.yaml is no longer supported

### Feat

- remove support for several paths per project (#55)
- implements thread limit per predict() (#54)
- add model download utils (#52)
- add metrics endpoint (#49)

### Fix

- Fix assert on test for base Model and ResultResponse.  (#59)

## 0.1.2 (2025-01-27)

### Fix

- library template

## 0.1.2b0 (2025-01-27)

### Feat

- **cli**: adds run cli command (#35)

## 0.1.2a0 (2025-01-27)

## 0.1.1 (2025-01-27)

### Fix

- input model name for http request (#36)

## 0.1.0 (2025-01-27)

### Feat

- build and publish cogito as a pip module in test pypi and pypi (#31)
- Implements type inspection of predict methods (#20)

### Fix

- Changes JSONResponse importation (#26)

### Refactor

- add codestyle pre commit (#32)
- models setup on lifespan (#25)

## 0.0.1 (2025-01-24)

### Feat

- generate predictor classes in client implementations using scaffolds (#27)
- handle setup and predict method raised exceptions (#22)
- update init command for both default/prompted values and forced initialization (#21)
- Adds inspection of predict return type  (#18)
- handling predictor initialization (#13)
- **config.py**: Handles NotFileFound if cogito.yaml not found (#15)
- add default healtcheck endpoint (#4)
- add config file management (#5)
- add app class to initialize (#3)
- initial commit

### Fix

- **cogito/core/app.py**:  re-import correct module for get_predictor_handler_return_type (#19)
- **config.py**: Fixes bug in load_from_file exception handling (#16)

### Refactor

- **setup.py, .gitignore**: Update package dependencies and ignore build artifacts
- define common inference server and training config (#17)
- **setup.py**: update project description and remove old one
- Remove unused files and modules for cleaner structure

import os
import pytest
from unittest.mock import patch, mock_open, MagicMock

from cogito.commands.scaffold import scaffold_predict_classes, scaffold_train_classes
from cogito.core.config import ConfigFile
from cogito.core.config.v0 import (
    RouteConfig,
    ServerConfig,
    CogitoConfig,
    FastAPIConfig,
)


@pytest.fixture
def mock_config():
    route = RouteConfig(
        name="Test Prediction",
        path="/v1/predict",
        predictor="predict.py:PredictClass",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
    )
    server = ServerConfig(
        route=route,
        name="Test Server",
        description="Test Server Description",
        version="1.0.0",
        fastapi=FastAPIConfig(host="0.0.0.0", port=8000, debug=False, access_log=False),
    )
    cogito = CogitoConfig(server=server, trainer="trainer.py:TrainerClass")

    # Create a mock version of the CogitoConfig with properties that match what scaffold_predict expects
    mock_cogito = MagicMock()
    mock_cogito.get_predictor = "predict:PredictClass"
    mock_cogito.predictor = "predict:PredictClass"  # Also set the direct attribute
    mock_cogito.get_trainer = "train:TrainerClass"
    mock_cogito.trainer = "train:TrainerClass"  # Add trainer attribute for train tests
    mock_cogito.get_route = route
    mock_cogito.route = route  # Also set the direct attribute

    # Create a mock ConfigFile with our mock cogito
    mock_config = MagicMock(spec=ConfigFile)
    mock_config.cogito = mock_cogito

    return mock_config


@pytest.fixture
def mock_template():
    return """
class {{ route.class_name }}:
    def predict(self, input_data: dict) -> dict:
        return {}
"""


def test_scaffold_predict_classes_creates_new_file(mock_config, tmp_path):
    """Test que verifica la creación de un nuevo archivo de predicción."""
    # Guardar el directorio original
    original_dir = os.getcwd()

    try:
        # Cambiar al directorio temporal
        os.chdir(tmp_path)

        with patch("cogito.commands.scaffold.Environment") as mock_env:
            # Configurar el mock del template
            mock_template = MagicMock()
            mock_template.render.return_value = "class PredictClass:\n    pass\n"
            mock_env.return_value.get_template.return_value = mock_template

            # Ejecutar la función (sin mockear open para que realmente cree el archivo)
            scaffold_predict_classes(mock_config)

            # Verificar que el archivo fue creado (usando la ruta correcta)
            file_path = os.path.join(tmp_path, "predict.py")
            assert os.path.exists(file_path)

            # Verificar el contenido del archivo
            with open(file_path, "r") as f:
                content = f.read()
                assert "class PredictClass:" in content
    finally:
        # Restaurar el directorio original
        os.chdir(original_dir)


def test_scaffold_predict_classes_force_overwrite(mock_config, tmp_path):
    """Test que verifica la sobrescritura de archivos existentes con --force."""
    # Guardar el directorio original
    original_dir = os.getcwd()

    try:
        # Cambiar al directorio temporal
        os.chdir(tmp_path)

        with patch("cogito.commands.scaffold.Environment") as mock_env:
            # Configurar el mock del template
            mock_template = MagicMock()
            mock_template.render.return_value = "class PredictClass:\n    pass\n"
            mock_env.return_value.get_template.return_value = mock_template

            # Crear un archivo existente con el nombre correcto
            with open("predict.py", "w") as f:
                f.write("# Existing content")

            # Ejecutar la función con force=True
            scaffold_predict_classes(mock_config, force=True)

            # Verificar que el archivo fue sobrescrito
            with open("predict.py", "r") as f:
                content = f.read()
                assert "class PredictClass:" in content
                assert "# Existing content" not in content
    finally:
        # Restaurar el directorio original
        os.chdir(original_dir)


def test_scaffold_predict_classes_no_force_existing_file(mock_config, tmp_path, capsys):
    """Test que verifica que no se sobrescriben archivos sin --force."""
    # Guardar el directorio original
    original_dir = os.getcwd()

    try:
        # Cambiar al directorio temporal
        os.chdir(tmp_path)

        with patch("cogito.commands.scaffold.Environment") as mock_env:
            # Configurar el mock del template
            mock_template = MagicMock()
            mock_template.render.return_value = "class PredictClass:\n    pass\n"
            mock_env.return_value.get_template.return_value = mock_template

            # Crear un archivo existente con el nombre correcto (predict.py)
            existing_content = "# Existing content"
            with open("predict.py", "w") as f:
                f.write(existing_content)

            # Ejecutar la función sin force
            scaffold_predict_classes(mock_config, force=False)

            # Verificar que el archivo mantiene su contenido original
            with open("predict.py", "r") as f:
                content = f.read()
                assert content == existing_content

            # Verificar el mensaje de advertencia
            captured = capsys.readouterr()
            assert "File predict.py already exists" in captured.out
    finally:
        # Restaurar el directorio original
        os.chdir(original_dir)


def test_scaffold_train_classes_creates_new_file(mock_config, tmp_path):
    """Test que verifica la creación de un nuevo archivo de entrenamiento."""
    # Guardar el directorio original
    original_dir = os.getcwd()

    try:
        # Cambiar al directorio temporal
        os.chdir(tmp_path)

        with patch("cogito.commands.scaffold.Environment") as mock_env:
            # Configurar el mock del template
            mock_template = MagicMock()
            mock_template.render.return_value = "class TrainerClass:\n    pass\n"
            mock_env.return_value.get_template.return_value = mock_template

            # Ejecutar la función (sin mockear open para que realmente cree el archivo)
            scaffold_train_classes(mock_config)

            # Verificar que el archivo fue creado (usando la ruta correcta)
            file_path = os.path.join(tmp_path, "train.py")
            assert os.path.exists(file_path)

            # Verificar el contenido del archivo
            with open(file_path, "r") as f:
                content = f.read()
                assert "class TrainerClass:" in content
    finally:
        # Restaurar el directorio original
        os.chdir(original_dir)


def test_scaffold_train_classes_force_overwrite(mock_config, tmp_path):
    """Test que verifica la sobrescritura de archivos existentes de entrenamiento con --force."""
    # Guardar el directorio original
    original_dir = os.getcwd()

    try:
        # Cambiar al directorio temporal
        os.chdir(tmp_path)

        with patch("cogito.commands.scaffold.Environment") as mock_env:
            # Configurar el mock del template
            mock_template = MagicMock()
            mock_template.render.return_value = "class TrainerClass:\n    pass\n"
            mock_env.return_value.get_template.return_value = mock_template

            # Crear un archivo existente con el nombre correcto
            with open("train.py", "w") as f:
                f.write("# Existing content")

            # Ejecutar la función con force=True
            scaffold_train_classes(mock_config, force=True)

            # Verificar que el archivo fue sobrescrito
            with open("train.py", "r") as f:
                content = f.read()
                assert "class TrainerClass:" in content
                assert "# Existing content" not in content
    finally:
        # Restaurar el directorio original
        os.chdir(original_dir)


def test_scaffold_train_classes_no_force_existing_file(mock_config, tmp_path, capsys):
    """Test que verifica que no se sobrescriben archivos de entrenamiento sin --force."""
    # Guardar el directorio original
    original_dir = os.getcwd()

    try:
        # Cambiar al directorio temporal
        os.chdir(tmp_path)

        with patch("cogito.commands.scaffold.Environment") as mock_env:
            # Configurar el mock del template
            mock_template = MagicMock()
            mock_template.render.return_value = "class TrainerClass:\n    pass\n"
            mock_env.return_value.get_template.return_value = mock_template

            # Crear un archivo existente con el nombre correcto (train.py)
            existing_content = "# Existing content"
            with open("train.py", "w") as f:
                f.write(existing_content)

            # Ejecutar la función sin force
            scaffold_train_classes(mock_config, force=False)

            # Verificar que el archivo mantiene su contenido original
            with open("train.py", "r") as f:
                content = f.read()
                assert content == existing_content

            # Verificar el mensaje de advertencia
            captured = capsys.readouterr()
            assert "File train.py already exists" in captured.out
    finally:
        # Restaurar el directorio original
        os.chdir(original_dir)

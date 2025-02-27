import unittest
from typing import List, Optional, Any
from pydantic import Field

try:
    # Pydantic v2
    from pydantic.fields import FieldInfo as PydanticField
except ImportError:
    # Pydantic v1
    from pydantic.fields import Field as PydanticField

from cogito.core.utils import create_request_model


class TestCreateRequestModel(unittest.TestCase):
    def test_create_request_model_basic(self):
        """Test basic functionality of create_request_model"""

        def mock_handler(param1: str, param2: int = 42):
            return f"{param1} {param2}"

        descriptor = "my.module:MockHandler"
        class_name, input_model = create_request_model(descriptor, mock_handler)

        # Check class_name extraction
        self.assertEqual(class_name, "MockHandler")

        # Check model name
        self.assertEqual(input_model.__name__, "MockHandlerRequest")

        # Check model fields
        self.assertTrue(
            hasattr(input_model, "model_fields") or hasattr(input_model, "__fields__")
        )

        # Check for Pydantic v1 or v2
        if hasattr(input_model, "model_fields"):  # Pydantic v2
            fields = input_model.model_fields
        else:  # Pydantic v1
            fields = input_model.__fields__

        self.assertIn("param1", fields)
        self.assertIn("param2", fields)

        # Create an instance to test validation
        instance = input_model(param1="test")
        self.assertEqual(instance.param1, "test")
        self.assertEqual(instance.param2, 42)  # Default value

        # Test with wrong type
        with self.assertRaises(Exception):
            input_model(param1=123)  # param1 should be str

    def test_create_request_model_complex_types(self):
        """Test with more complex parameter types"""

        def complex_handler(
            text: str,
            numbers: List[int] = [1, 2, 3],
            optional_param: Optional[str] = None,
            any_param: Any = "anything",
            field_param: str = Field(
                default="field_default", description="A field with metadata"
            ),
        ):
            return "result"

        descriptor = "complex.module:ComplexHandler"
        class_name, input_model = create_request_model(descriptor, complex_handler)

        # Check type handling
        instance = input_model(text="hello")
        self.assertEqual(instance.text, "hello")
        self.assertEqual(instance.numbers, [1, 2, 3])
        self.assertIsNone(instance.optional_param)
        self.assertEqual(instance.any_param, "anything")
        self.assertEqual(instance.field_param, "field_default")

        # Test with custom values
        custom_instance = input_model(
            text="custom",
            numbers=[4, 5, 6],
            optional_param="provided",
            any_param={"key": "value"},
            field_param="custom_field",
        )
        self.assertEqual(custom_instance.text, "custom")
        self.assertEqual(custom_instance.numbers, [4, 5, 6])
        self.assertEqual(custom_instance.optional_param, "provided")
        self.assertEqual(custom_instance.any_param, {"key": "value"})
        self.assertEqual(custom_instance.field_param, "custom_field")

    def test_create_request_model_ellipsis(self):
        """Test handling of required parameters (with Ellipsis)"""

        def required_handler(required_param, default_param=123):
            return "result"

        descriptor = "required.module:RequiredHandler"
        class_name, input_model = create_request_model(descriptor, required_handler)

        # Should fail without required_param
        with self.assertRaises(Exception):
            input_model()

        # Should work with required_param
        instance = input_model(required_param="value")
        self.assertEqual(instance.required_param, "value")
        self.assertEqual(instance.default_param, 123)


if __name__ == "__main__":
    unittest.main()

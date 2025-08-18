"""
Updated models module for the debate-setting using the registry system.

This module demonstrates how to use the new model registry system
instead of the old fragile string-matching approach.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import model_registry
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the new registry-based model classes
from model_registry.base_models import create_model

# For backward compatibility, we can also import the base classes
from model_registry.base_models import BaseModel, RegistryOpenModel, RegistryClosedModel


# Factory function that replaces the old model creation logic
def create_model_instance(model_name, api_key=None, base_url=None):
    """
    Create and return the appropriate model instance using the registry system.

    This replaces the old logic:
    ```python
    if model_name.startswith(("openai/", "anthropic/", "claude")):
        return ClosedModel(model_id=model_name, api_key=api_key, base_url=base_url)
    else:
        return OpenModel(model_name=model_name)
    ```
    """
    return create_model(model_name, api_key=api_key, base_url=base_url)


# Backward compatibility aliases
OpenModel = RegistryOpenModel
ClosedModel = RegistryClosedModel


# Example usage function
def example_usage():
    """Example of how to use the new registry-based models."""

    # Create models using the new system
    models_to_test = [
        "meta-llama/Llama-2-7b-chat-hf",
        "google/gemma-2b-it",
        "openai/gpt-4o"
    ]

    for model_name in models_to_test:
        print(f"\nCreating model: {model_name}")

        # Create model instance
        if model_name.startswith("openai/"):
            model = create_model_instance(model_name, api_key="your-api-key-here")
        else:
            model = create_model_instance(model_name)

        # Show configuration
        print(f"  Family: {model.config.family.value}")
        print(f"  Size: {model.config.size_category}")
        print(f"  Quantization: {model.config.quantization.value}")
        print(f"  API-based: {model.config.api_based}")

        if model.config.known_issues:
            print(f"  Known issues: {model.config.known_issues}")

        # The setup() and generate_responses() methods work the same way
        # but now use registry configuration internally

        # Example of generating chat messages (same interface as before)
        messages = model.get_chat_messages(
            question="What do you think about AI safety?",
            argument="AI safety is crucial for humanity's future",
            prompt_type="individual_thinker"
        )

        print(f"  Generated {len(messages)} chat messages")


if __name__ == "__main__":
    example_usage()

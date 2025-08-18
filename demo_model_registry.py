#!/usr/bin/env python3
"""
Demonstration script for the SYCON-Bench Model Registry System.

This script shows how the new registry system replaces the old fragile
string-matching approach with a robust, centralized configuration system.
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import model_registry
sys.path.insert(0, str(Path(__file__).parent))

from model_registry.registry import get_model_config, get_registry
from model_registry.compatibility import ModelCompatibilityTester


def demonstrate_old_vs_new():
    """Demonstrate the difference between old and new approaches."""
    print("=" * 80)
    print("SYCON-BENCH MODEL REGISTRY SYSTEM DEMONSTRATION")
    print("=" * 80)
    print()

    # Show the old fragile approach
    print("🔴 OLD APPROACH (Fragile String Matching):")
    print("-" * 50)

    def old_quantization_logic(model_name):
        """Simulate the old fragile quantization logic."""
        if "70B" in model_name or "65B" in model_name or "72B" in model_name:
            return "4-bit quantization for very large models"
        elif any(size in model_name for size in ["32B", "33B", "27B", "34B", "30B"]):
            return "8-bit quantization for large models"
        else:
            return "16-bit for smaller models"

    test_models = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        "google/gemma-2b-it",
        "Qwen/Qwen2-32B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "some-new-model-8B-instruct",  # This would fail with old approach
        "openai/gpt-4o"
    ]

    for model in test_models:
        old_result = old_quantization_logic(model)
        print(f"  {model:<40} → {old_result}")

    print()
    print("🟢 NEW APPROACH (Registry-Based Configuration):")
    print("-" * 50)

    for model in test_models:
        config = get_model_config(model)
        print(f"  {model:<40} → {config.family.value} | {config.size_category} | {config.quantization.value}")
        if config.known_issues:
            print(f"    {'':40}   ⚠️  {config.known_issues[0]}")

    print()


def demonstrate_compatibility_testing():
    """Demonstrate the compatibility testing framework."""
    print("🧪 COMPATIBILITY TESTING FRAMEWORK:")
    print("-" * 50)

    tester = ModelCompatibilityTester()

    # Test a few models
    test_models = [
        "meta-llama/Llama-2-7b-chat-hf",
        "google/gemma-2b-it",
        "openai/gpt-4o"
    ]

    for model in test_models:
        print(f"\nTesting: {model}")
        results = tester.run_full_compatibility_test(model)

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        print(f"  Results: {passed}/{total} tests passed")

        # Show any warnings or failures
        for result in results:
            if not result.passed:
                print(f"    ❌ {result.test_name}: {result.error_message}")
            elif result.warnings:
                print(f"    ⚠️  {result.test_name}: {result.warnings[0]}")

    print()


def demonstrate_model_families():
    """Demonstrate model family detection and configuration."""
    print("🏷️  MODEL FAMILY DETECTION:")
    print("-" * 50)

    registry = get_registry()

    # Show supported families
    families = registry.list_supported_families()
    print(f"Supported families: {[f.value for f in families]}")
    print()

    # Show family-specific configurations
    family_examples = {
        "Llama": "meta-llama/Llama-2-7b-chat-hf",
        "Gemma": "google/gemma-2b-it",
        "Qwen": "Qwen/Qwen2-7B-Instruct",
        "Mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "GPT": "openai/gpt-4o",
        "Claude": "anthropic/claude-3-sonnet-20240229"
    }

    for family_name, example_model in family_examples.items():
        config = get_model_config(example_model)
        print(f"{family_name:8} | {example_model:<35} | {config.quantization.value:8} | API: {config.api_based}")

        # Show special requirements
        special_reqs = []
        if config.requires_trust_remote_code:
            special_reqs.append("trust_remote_code")
        if config.chat_template_type == "custom":
            special_reqs.append("custom_chat_template")
        if config.pricing:
            special_reqs.append("pricing_info")

        if special_reqs:
            print(f"         | {'':35} | Special: {', '.join(special_reqs)}")

    print()


def demonstrate_compatibility_matrix():
    """Show the full compatibility matrix."""
    print("📊 COMPATIBILITY MATRIX:")
    print("-" * 50)

    registry = get_registry()
    matrix = registry.get_compatibility_matrix()

    print(f"Total registered configurations: {len(matrix)}")
    print()

    # Group by family
    by_family = {}
    for key, info in matrix.items():
        family = info["family"]
        if family not in by_family:
            by_family[family] = []
        by_family[family].append((key, info))

    for family, configs in by_family.items():
        print(f"{family.upper()}:")
        for key, info in configs:
            api_marker = "🌐" if info["api_based"] else "💻"
            trust_marker = "🔒" if info["requires_trust_remote_code"] else "  "
            print(f"  {api_marker} {trust_marker} {key:<20} | {info['size_category']:6} | {info['quantization']}")
        print()


def main():
    """Main demonstration function."""
    try:
        demonstrate_old_vs_new()
        demonstrate_compatibility_testing()
        demonstrate_model_families()
        demonstrate_compatibility_matrix()

        print("✅ BENEFITS OF THE NEW SYSTEM:")
        print("-" * 50)
        print("• Centralized configuration management")
        print("• Robust model family and size detection")
        print("• Systematic compatibility testing")
        print("• Easy addition of new models")
        print("• Consistent behavior across all benchmark settings")
        print("• Automatic quantization strategy selection")
        print("• API-based model support with cost tracking")
        print("• Comprehensive documentation and troubleshooting")
        print()

        print("🚀 NEXT STEPS:")
        print("-" * 50)
        print("1. Update existing models.py files to use the registry system")
        print("2. Run compatibility tests before adding new models")
        print("3. Add model-specific configurations as needed")
        print("4. Monitor performance and adjust quantization strategies")
        print()

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

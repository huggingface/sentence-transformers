from __future__ import annotations

import json
import time
from pathlib import Path

from huggingface_hub import list_models
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES


def find_model_for_architecture(architecture):
    """
    Find a model from hf-internal-testing or tiny-random for the given architecture.
    If multiple models exist, prefer the one ending with 'Model'.
    """
    for author in [
        "trl-internal-testing",
        "tiny-random",
        "hf-internal-testing",
        "hf-tiny-model-private",
        "peft-internal-testing",
        "onnx-internal-testing",
    ]:
        # Search for models from the author with the architecture tag
        models = list(list_models(filter=architecture, author=author, limit=100))

        if not models:
            continue

        # Filter for models that match the architecture pattern
        matching_models = []
        for model in models:
            model_id = model.id
            # Check if the model_id contains the architecture name
            if architecture.lower().replace("_", "").replace("-", "") in model_id.lower().replace("_", "").replace(
                "-", ""
            ) and (
                "tiny-random" in model_id.lower() or "tiny" in model_id.lower() and author == "trl-internal-testing"
            ):
                matching_models.append(model_id)

        if not matching_models:
            continue

        # Prefer models ending with "Model"
        model_suffix_matches = [m for m in matching_models if m.split("/")[-1].endswith("Model")]
        if model_suffix_matches:
            # Return the shortest one that ends with "Model"
            return min(model_suffix_matches, key=len)

        # Otherwise return the first match
        return matching_models[0]

    return None


def update_tiny_model_mapping(tiny_model_mapping):
    """Update the tiny model mapping using the Hugging Face Hub API"""
    updated_mapping = {}
    total = len(tiny_model_mapping)

    for idx, (architecture, current_value) in enumerate(tiny_model_mapping.items(), 1):
        print(f"[{idx}/{total}] Processing {architecture}...", end=" ")

        # Skip if we already have a value
        if current_value is not None:
            updated_mapping[architecture] = current_value
            print(f"Skipped (already has value): {current_value}")
            continue

        # Try to find a model for this architecture
        found_model = find_model_for_architecture(architecture)

        if found_model:
            updated_mapping[architecture] = found_model
            print(f"Found: {found_model}")
        else:
            updated_mapping[architecture] = None
            print("Not found")

        time.sleep(0.2)

    return updated_mapping


if __name__ == "__main__":
    json_file = Path(__file__).parent.parent / "models" / "transformers_tiny_models.json"
    with open(json_file, encoding="utf8") as f:
        tiny_model_mapping = json.load(f)

    # Add any missing architectures with None value
    for arch in sorted(set(MODEL_MAPPING_NAMES.keys()) - set(tiny_model_mapping.keys())):
        tiny_model_mapping[arch] = None
    tiny_model_mapping = dict(sorted(tiny_model_mapping.items()))

    updated_mapping = update_tiny_model_mapping(tiny_model_mapping)
    with open(json_file, "w", encoding="utf8") as f:
        json.dump(updated_mapping, f, indent=2)

    # Count statistics
    total = len(updated_mapping)
    found = sum(1 for v in updated_mapping.values() if v is not None)
    not_found = total - found
    print("Statistics:")
    print(f"  Total architectures: {total}")
    print(f"  Found models: {found}")
    print(f"  Not found: {not_found}")

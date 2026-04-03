"""Dataset module for vision-language model training with HuggingFace dataset support."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from urllib.parse import urlparse

from datasets import load_dataset, DatasetDict
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm


@dataclass
class VLMDatasetConfig:
    """Configuration for loading and converting HuggingFace vision-language datasets."""
    
    # Dataset identification
    dataset_name: str
    dataset_config: Optional[str] = None
    split: str = "train"
    max_samples: Optional[int] = None  # Limit dataset size for testing
    
    # Split ratios
    train_split_ratio: float = 0.9
    val_test_split_ratio: float = 0.5
    
    # Field mapping
    image_field: str = "image"
    conversations_field: str = "conversations"
    
    # Image handling
    download_images: bool = True  # Whether to save PIL images or download from URLs
    image_output_dir: str = "images"  # Subdirectory for saved/downloaded images
    
    # Processing
    num_proc: int = 8
    load_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Custom converter for non-standard formats
    custom_converter: Optional[Callable] = None


def prepare_vlm_datasets(config: VLMDatasetConfig, dataset_root: str):
    """
    Prepare vision-language datasets by converting HuggingFace dataset to JSONL format.
    
    Args:
        config: VLMDatasetConfig with dataset settings
        dataset_root: Root directory for output JSONL files
        
    Output format:
        {
            "image": "path/to/image.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\\nQuestion"},
                {"from": "gpt", "value": "Answer"}
            ]
        }
    """
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    marker_file = dataset_root / ".data_ready"
    
    if marker_file.exists():
        print("Dataset already prepared.")
        return
    
    print(f"Loading vision-language dataset '{config.dataset_name}'...")
    
    # Load dataset
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        num_proc=config.num_proc,
        trust_remote_code=True,
        **config.load_kwargs
    )
    
    if config.split not in dataset:
        raise ValueError(
            f"Split '{config.split}' not found in dataset. "
            f"Available splits: {list(dataset.keys())}"
        )
    
    initial_data = dataset[config.split]
    
    # Limit dataset size if specified (for testing)
    if config.max_samples is not None and len(initial_data) > config.max_samples:
        print(f"Limiting dataset to {config.max_samples} samples (from {len(initial_data)})")
        initial_data = initial_data.select(range(config.max_samples))
    
    # Split into train and (val+test)
    train_testval = initial_data.train_test_split(
        test_size=1.0 - config.train_split_ratio
    )
    
    # Split (val+test) into val and test
    if config.val_test_split_ratio <= 0.0:
        # All remaining data goes to validation, none to test
        split_dataset = DatasetDict({
            'train': train_testval['train'],
            'val': train_testval['test'],
            'test': train_testval['test'].select([])
        })
    elif config.val_test_split_ratio >= 1.0:
        # All remaining data goes to test, none to validation
        split_dataset = DatasetDict({
            'train': train_testval['train'],
            'val': train_testval['test'].select([]),
            'test': train_testval['test']
        })
    else:
        test_val = train_testval['test'].train_test_split(
            test_size=config.val_test_split_ratio
        )
        split_dataset = DatasetDict({
            'train': train_testval['train'],
            'val': test_val['train'],
            'test': test_val['test']
        })
    
    print(f"Converting to JSONL format...")
    print(f"  Train samples: {len(split_dataset['train'])}")
    print(f"  Val samples: {len(split_dataset['val'])}")
    print(f"  Test samples: {len(split_dataset['test'])}")
    
    # Create image directory if downloading
    if config.download_images:
        image_dir = dataset_root / config.image_output_dir
        image_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSONL
    _convert_vlm_dataset_to_jsonl(
        split_dataset['train'], 
        dataset_root / "training.jsonl", 
        config,
        dataset_root
    )
    _convert_vlm_dataset_to_jsonl(
        split_dataset['val'], 
        dataset_root / "validation.jsonl", 
        config,
        dataset_root
    )
    _convert_vlm_dataset_to_jsonl(
        split_dataset['test'], 
        dataset_root / "test.jsonl", 
        config,
        dataset_root
    )
    
    marker_file.write_text('ready')
    print("Dataset preparation complete!")


def _convert_vlm_dataset_to_jsonl(
    dataset, 
    path: Path, 
    config: VLMDatasetConfig,
    dataset_root: Path
):
    """Convert HuggingFace vision-language dataset to JSONL format."""
    total_samples = len(dataset)
    converted_count = 0
    skipped_count = 0
    
    print(f"Converting {path.name} ({total_samples} samples)...")
    
    # Process in batches for better performance
    batch_size = 1000
    results = []
    
    for batch_start in tqdm(range(0, total_samples, batch_size), desc=f"Processing batches", unit="batch"):
        batch_end = min(batch_start + batch_size, total_samples)
        batch = dataset[batch_start:batch_end]
        
        # Process batch
        for i in range(len(batch[config.image_field])):
            idx = batch_start + i
            try:
                # Reconstruct sample dict from batch
                sample = {key: batch[key][i] for key in batch.keys()}
                
                converted = _convert_vlm_sample(sample, config, dataset_root, idx)
                if converted is not None:
                    results.append(converted)
                    converted_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                if skipped_count < 10:  # Only print first 10 errors
                    print(f"\nWarning: Failed to convert sample {idx}: {e}")
                skipped_count += 1
                continue
        
        # Write batch to file
        if len(results) >= batch_size:
            with open(path, "a", encoding='utf-8') as f:
                for result in results:
                    json_string = json.dumps(result, ensure_ascii=False) + "\n"
                    f.write(json_string)
            results = []
    
    # Write remaining results
    if results:
        with open(path, "a", encoding='utf-8') as f:
            for result in results:
                json_string = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_string)
    
    print(f"  Converted: {converted_count}/{total_samples} samples (skipped {skipped_count})")


def _convert_vlm_sample(
    sample: Dict[str, Any], 
    config: VLMDatasetConfig,
    dataset_root: Path,
    sample_idx: int
) -> Optional[Dict[str, Any]]:
    """
    Convert a single vision-language sample to universal format.
    
    Returns:
        {
            "image": "path/to/image.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\\nQuestion"},
                {"from": "gpt", "value": "Answer"}
            ]
        }
    """
    # Use custom converter if provided
    if config.custom_converter is not None:
        return config.custom_converter(sample, config, dataset_root, sample_idx)
    
    # Extract image
    image_data = sample.get(config.image_field)
    if image_data is None:
        return None
    
    # Handle image (PIL Image, URL, or path)
    image_path = _process_image(image_data, config, dataset_root, sample_idx)
    if image_path is None:
        return None
    
    # Extract conversations
    conversations = sample.get(config.conversations_field)
    if conversations is None or not conversations or len(conversations) == 0:
        return None
    
    # Normalize conversation format
    try:
        normalized_conversations = _normalize_conversations(conversations)
    except Exception as e:
        # Skip samples with malformed conversations
        return None
    
    # Validate that we have at least one valid conversation turn
    if not normalized_conversations or len(normalized_conversations) == 0:
        return None
    
    # Validate that conversations have required fields
    for conv in normalized_conversations:
        if 'from' not in conv or 'value' not in conv:
            return None
        if not conv['value'] or not conv['value'].strip():
            return None
    
    return {
        "image": image_path,
        "conversations": normalized_conversations
    }


def _process_image(
    image_data: Any,
    config: VLMDatasetConfig,
    dataset_root: Path,
    sample_idx: int
) -> Optional[str]:
    """
    Process image data and return path.
    
    Handles:
    - None (text-only samples - skip silently)
    - PIL Image objects (save to disk)
    - URLs (download if download_images=True, otherwise return URL)
    - Local paths (return as-is)
    """
    # Handle None (text-only samples)
    if image_data is None:
        return None
    
    # If it's a PIL Image
    if isinstance(image_data, Image.Image):
        # Always save PIL images to disk
        image_dir = dataset_root / config.image_output_dir
        image_path = image_dir / f"image_{sample_idx:08d}.jpg"
        image_data.convert('RGB').save(image_path)
        return str(image_path.relative_to(dataset_root))
    
    # If it's a string (URL or path)
    if isinstance(image_data, str):
        # Check if it's a URL
        parsed = urlparse(image_data)
        if parsed.scheme in ('http', 'https'):
            if config.download_images:
                # Download image
                try:
                    response = requests.get(image_data, timeout=10)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                    
                    image_dir = dataset_root / config.image_output_dir
                    image_path = image_dir / f"image_{sample_idx:08d}.jpg"
                    img.convert('RGB').save(image_path)
                    return str(image_path.relative_to(dataset_root))
                except Exception as e:
                    print(f"Warning: Failed to download image for sample {sample_idx}: {e}")
                    return None
            else:
                # Return URL as-is
                return image_data
        else:
            # It's a local path
            return image_data
    
    print(f"Warning: Unknown image data type in sample {sample_idx}: {type(image_data)}")
    return None


def _normalize_conversations(conversations: Any) -> List[Dict[str, str]]:
    """
    Normalize conversation format to standard format.
    
    Expected output:
        [
            {"from": "human", "value": "<image>\\nQuestion"},
            {"from": "gpt", "value": "Answer"}
        ]
    """
    if not isinstance(conversations, list):
        raise ValueError(f"Conversations must be a list, got {type(conversations)}")
    
    normalized = []
    for turn in conversations:
        if not isinstance(turn, dict):
            raise ValueError(f"Each conversation turn must be a dict, got {type(turn)}")
        
        # Normalize 'from' field
        from_field = turn.get('from', turn.get('role', turn.get('speaker')))
        if from_field is None:
            raise ValueError(f"Conversation turn missing 'from'/'role'/'speaker' field: {turn}")
        
        # Normalize to 'human' or 'gpt'
        if from_field.lower() in ('human', 'user', 'question'):
            from_normalized = 'human'
        elif from_field.lower() in ('gpt', 'assistant', 'answer', 'bot'):
            from_normalized = 'gpt'
        else:
            from_normalized = from_field  # Keep as-is if unknown
        
        # Normalize 'value' field
        value_field = turn.get('value', turn.get('text', turn.get('content')))
        if value_field is None:
            raise ValueError(f"Conversation turn missing 'value'/'text'/'content' field: {turn}")
        
        normalized.append({
            "from": from_normalized,
            "value": value_field
        })
    
    return normalized


# ============================================================================
# Pre-built converters for common datasets
# ============================================================================

def llava_instruct_converter(
    sample: Dict[str, Any],
    config: VLMDatasetConfig,
    dataset_root: Path,
    sample_idx: int
) -> Optional[Dict[str, Any]]:
    """
    Converter for LLaVA-Instruct-150K dataset.
    
    Format:
        {
            "id": <string or int>,  # Mixed types in dataset
            "image": "path/to/image.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\\nQuestion"},
                {"from": "gpt", "value": "Answer"}
            ]
        }
    """
    # LLaVA-Instruct is already in the correct format
    image_path = _process_image(sample.get('image'), config, dataset_root, sample_idx)
    if image_path is None:
        return None
    
    conversations = sample.get('conversations', [])
    normalized = _normalize_conversations(conversations)
    
    return {
        "image": image_path,
        "conversations": normalized
    }


def sharegpt4v_converter(
    sample: Dict[str, Any],
    config: VLMDatasetConfig,
    dataset_root: Path,
    sample_idx: int
) -> Optional[Dict[str, Any]]:
    """
    Converter for ShareGPT4V dataset.
    
    Adapt based on actual ShareGPT4V format.
    """
    # Placeholder - adjust based on actual dataset structure
    return _convert_vlm_sample(sample, config, dataset_root, sample_idx)


def vqav2_converter(
    sample: Dict[str, Any],
    config: VLMDatasetConfig,
    dataset_root: Path,
    sample_idx: int
) -> Optional[Dict[str, Any]]:
    """
    Converter for VQAv2 dataset.
    
    Format:
        {
            "question": "What color is the cat?",
            "answer": "Orange",
            "image": <PIL Image>
        }
    """
    image_path = _process_image(sample.get('image'), config, dataset_root, sample_idx)
    if image_path is None:
        return None
    
    question = sample.get('question', '')
    answer = sample.get('answer', sample.get('answers', [''])[0])
    
    conversations = [
        {"from": "human", "value": f"<image>\n{question}"},
        {"from": "gpt", "value": str(answer)}
    ]
    
    return {
        "image": image_path,
        "conversations": conversations
    }


# Registry of pre-built converters
CONVERTER_REGISTRY = {
    "lmms-lab/LLaVA-NeXT-Data": llava_instruct_converter,  # Uses LLaVA format
    "HuggingFaceM4/VQAv2": vqav2_converter,
}


def get_converter_for_dataset(dataset_name: str) -> Optional[Callable]:
    """Get pre-built converter for known datasets."""
    return CONVERTER_REGISTRY.get(dataset_name)

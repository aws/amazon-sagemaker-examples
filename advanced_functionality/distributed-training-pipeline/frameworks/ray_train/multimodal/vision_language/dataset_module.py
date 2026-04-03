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
    
    dataset_name: str
    dataset_config: Optional[str] = None
    split: str = "train"
    max_samples: Optional[int] = None
    
    train_split_ratio: float = 0.9
    val_test_split_ratio: float = 0.5
    
    image_field: str = "image"
    conversations_field: str = "conversations"
    
    download_images: bool = True
    image_output_dir: str = "images"
    
    num_proc: int = 8
    load_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    custom_converter: Optional[Callable] = None


def prepare_vlm_datasets(config: VLMDatasetConfig, dataset_root: str):
    """
    Prepare vision-language datasets by converting HuggingFace dataset to JSONL format.
    
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
    
    if config.max_samples is not None and len(initial_data) > config.max_samples:
        print(f"Limiting dataset to {config.max_samples} samples (from {len(initial_data)})")
        initial_data = initial_data.select(range(config.max_samples))
    
    # Split into train and (val+test)
    train_testval = initial_data.train_test_split(
        test_size=1.0 - config.train_split_ratio
    )
    
    # Split (val+test) into val and test — handle boundary cases
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
    
    if config.download_images:
        image_dir = dataset_root / config.image_output_dir
        image_dir.mkdir(parents=True, exist_ok=True)
    
    _convert_vlm_dataset_to_jsonl(
        split_dataset['train'], dataset_root / "training.jsonl", config, dataset_root
    )
    _convert_vlm_dataset_to_jsonl(
        split_dataset['val'], dataset_root / "validation.jsonl", config, dataset_root
    )
    _convert_vlm_dataset_to_jsonl(
        split_dataset['test'], dataset_root / "test.jsonl", config, dataset_root
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
    
    batch_size = 1000
    results = []
    
    for batch_start in tqdm(range(0, total_samples, batch_size), desc=f"Processing batches", unit="batch"):
        batch_end = min(batch_start + batch_size, total_samples)
        batch = dataset[batch_start:batch_end]
        
        for i in range(len(batch[config.image_field])):
            idx = batch_start + i
            try:
                sample = {key: batch[key][i] for key in batch.keys()}
                converted = _convert_vlm_sample(sample, config, dataset_root, idx)
                if converted is not None:
                    results.append(converted)
                    converted_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                if skipped_count < 10:
                    print(f"\nWarning: Failed to convert sample {idx}: {e}")
                skipped_count += 1
                continue
        
        if len(results) >= batch_size:
            with open(path, "a", encoding='utf-8') as f:
                for result in results:
                    json_string = json.dumps(result, ensure_ascii=False) + "\n"
                    f.write(json_string)
            results = []
    
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
    """Convert a single vision-language sample to universal format."""
    if config.custom_converter is not None:
        return config.custom_converter(sample, config, dataset_root, sample_idx)
    
    image_data = sample.get(config.image_field)
    if image_data is None:
        return None
    
    image_path = _process_image(image_data, config, dataset_root, sample_idx)
    if image_path is None:
        return None
    
    conversations = sample.get(config.conversations_field)
    if conversations is None or not conversations or len(conversations) == 0:
        return None
    
    try:
        normalized_conversations = _normalize_conversations(conversations)
    except Exception:
        return None
    
    if not normalized_conversations or len(normalized_conversations) == 0:
        return None
    
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
    """Process image data and return path."""
    if image_data is None:
        return None
    
    if isinstance(image_data, Image.Image):
        image_dir = dataset_root / config.image_output_dir
        image_path = image_dir / f"image_{sample_idx:08d}.jpg"
        image_data.convert('RGB').save(image_path)
        return str(image_path.relative_to(dataset_root))
    
    if isinstance(image_data, str):
        parsed = urlparse(image_data)
        if parsed.scheme in ('http', 'https'):
            if config.download_images:
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
                return image_data
        else:
            return image_data
    
    print(f"Warning: Unknown image data type in sample {sample_idx}: {type(image_data)}")
    return None


def _normalize_conversations(conversations: Any) -> List[Dict[str, str]]:
    """Normalize conversation format to standard format."""
    if not isinstance(conversations, list):
        raise ValueError(f"Conversations must be a list, got {type(conversations)}")
    
    normalized = []
    for turn in conversations:
        if not isinstance(turn, dict):
            raise ValueError(f"Each conversation turn must be a dict, got {type(turn)}")
        
        from_field = turn.get('from', turn.get('role', turn.get('speaker')))
        if from_field is None:
            raise ValueError(f"Conversation turn missing 'from'/'role'/'speaker' field: {turn}")
        
        if from_field.lower() in ('human', 'user', 'question'):
            from_normalized = 'human'
        elif from_field.lower() in ('gpt', 'assistant', 'answer', 'bot'):
            from_normalized = 'gpt'
        else:
            from_normalized = from_field
        
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
    """Converter for LLaVA-Instruct-150K dataset."""
    image_path = _process_image(sample.get('image'), config, dataset_root, sample_idx)
    if image_path is None:
        return None
    conversations = sample.get('conversations', [])
    normalized = _normalize_conversations(conversations)
    return {"image": image_path, "conversations": normalized}


def vqav2_converter(
    sample: Dict[str, Any],
    config: VLMDatasetConfig,
    dataset_root: Path,
    sample_idx: int
) -> Optional[Dict[str, Any]]:
    """Converter for VQAv2 dataset."""
    image_path = _process_image(sample.get('image'), config, dataset_root, sample_idx)
    if image_path is None:
        return None
    question = sample.get('question', '')
    answer = sample.get('answer', sample.get('answers', [''])[0])
    conversations = [
        {"from": "human", "value": f"<image>\n{question}"},
        {"from": "gpt", "value": str(answer)}
    ]
    return {"image": image_path, "conversations": conversations}


CONVERTER_REGISTRY = {
    "lmms-lab/LLaVA-NeXT-Data": llava_instruct_converter,
    "HuggingFaceM4/VQAv2": vqav2_converter,
}


def get_converter_for_dataset(dataset_name: str) -> Optional[Callable]:
    """Get pre-built converter for known datasets."""
    return CONVERTER_REGISTRY.get(dataset_name)

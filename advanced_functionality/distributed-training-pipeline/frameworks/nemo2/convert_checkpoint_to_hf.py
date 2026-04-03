import argparse
from pathlib import Path
import re
from dataclasses import dataclass, fields, MISSING
from typing import Optional
from nemo.collections.llm import api
from nemo.collections.llm import peft


@dataclass
class ExportConfig:
    """Configuration for exporting NeMo 2.0 PEFT checkpoints to HuggingFace format."""
    # Checkpoint
    nemo_logs_dir: str = None
    
    # Export settings
    target: str = "hf"  # Target format (default: HuggingFace)
    overwrite: bool = False  # Whether to overwrite existing files
    
    # LoRA adapter settings
    no_merge: bool = False  # No merge of LoRA adapter with base model
    
    # Optional ModelOpt settings (for quantized models)
    use_modelopt: bool = False
    modelopt_export_kwargs: Optional[dict] = None
    
    @property
    def output_path(self) -> str:
        suffix = ".hf_peft" if self.no_merge else ".hf_model"
        return str(Path(self.nemo_checkpoint_path).with_suffix(suffix))

    @property
    def nemo_checkpoint_path(self) -> str:
        """Find the latest checkpoint in the checkpoints directory."""
        output_path = Path(self.nemo_logs_dir)
        
        # Find all timestamp directories (format: YYYY-MM-DD_HH-MM-SS)
        timestamp_dirs = [d for d in output_path.iterdir() if d.is_dir() and 
                        len(d.name.split('_')) == 2 and 
                        len(d.name.split('_')[0].split('-')) == 3]
        
        if not timestamp_dirs:
            raise FileNotFoundError(f"No timestamp directories found in {output_path}")
        
        # Get the latest timestamp directory by modification time
        latest_timestamp_dir = max(timestamp_dirs, key=lambda p: p.stat().st_mtime)
        
        # Look for checkpoint directories under checkpoints/
        checkpoints_dir = latest_timestamp_dir / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"No checkpoints directory found in {latest_timestamp_dir}")
        
        # Find all checkpoint directories (starting with "nemo_logs--")
        ckpt_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and 
                    d.name.startswith("nemo_logs--") and 
                    not re.search(r'\.(hf_model|hf_peft|merged)$', d.name)]
        
        if not ckpt_dirs:
            raise FileNotFoundError(f"No checkpoint directories found in {checkpoints_dir}")
        
        # Return the latest checkpoint directory by modification time
        latest_ckpt = max(ckpt_dirs, key=lambda p: p.stat().st_mtime)
        return str(latest_ckpt)
    
    def __post_init__(self):
        if self.nemo_logs_dir is None:
            outputs_path =Path.home() / "outputs"
            nemo_logs_dirs = list(outputs_path.rglob("nemo_logs"))
            if nemo_logs_dirs:
                self.nemo_logs_dir = str(max(nemo_logs_dirs, key=lambda p: p.stat().st_mtime))
                print(f"Found nemo_logs directory: {self.nemo_logs_dir}")
            else:
                raise ValueError("No nemo_logs folder found under outputs folder")
            
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ExportConfig':
        """Create ExportConfig from argparse Namespace."""
        config_fields = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in vars(args).items() 
                 if k in config_fields and v is not None}
        return cls(**kwargs)
    
    def validate(self):
        """Validate the configuration."""
        if not self.nemo_checkpoint_path:
            raise ValueError("nemo_checkpoint_path must be provided")
        
        checkpoint_path = Path(self.nemo_checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {self.nemo_checkpoint_path}")
        
        # Check if it's a valid NeMo checkpoint structure
        context_dir = checkpoint_path / "context"
        if not context_dir.exists():
            raise ValueError(
                f"Invalid NeMo checkpoint structure. Expected 'context' directory at: {context_dir}"
            )


def create_parser_from_dataclass(dataclass_type) -> argparse.ArgumentParser:
    """Generate argparse parser from dataclass fields."""
    parser = argparse.ArgumentParser(
        description="Export NeMo 2.0 PEFT checkpoints to HuggingFace format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    for f in fields(dataclass_type):
        # Skip complex types
        if f.name in ['modelopt_export_kwargs']:
            continue
        
        field_type = f.type
        default_value = f.default if f.default is not MISSING else None
        
        # Handle Optional types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is type(Optional):
            field_type = field_type.__args__[0]
        
        # Determine the argument type
        if field_type == bool:
            parser.add_argument(
                f'--{f.name}',
                action='store_true',
                default=default_value,
                help=f"Boolean flag for {f.name}"
            )
        else:
            parser.add_argument(
                f'--{f.name}',
                type=field_type if field_type in [int, float, str] else str,
                default=default_value,
                help=f"Value for {f.name}"
            )
    
    return parser

def export_nemo_to_hf(config: ExportConfig) -> Path:
    """
    Export NeMo 2.0 checkpoint to HuggingFace format.
    
    Args:
        config: ExportConfig object containing export parameters
        
    Returns:
        Path: The path where the HuggingFace checkpoint was saved
    """
    # Validate configuration
    config.validate()
    
    # Convert to Path object
    nemo_path = Path(config.nemo_checkpoint_path)
    output_path = Path(config.output_path)
    
    print("\nStarting export process...")
    print(f"Source checkpoint: {nemo_path}")
    
    try:
        # Handle LoRA adapter export
        if config.no_merge:
            print("Exporting LoRA adapter to HF PEFT format...")
            exported_path = peft.export_lora(
                lora_checkpoint_path=str(nemo_path),
                output_path=str(output_path),
            )
            return Path(exported_path)
        
        # Handle LoRA merge with base model
        print("Merging LoRA adapter with base model...")
        merged_path = output_path.parent / str(output_path.name).replace(".hf_model", ".merged")
        peft.merge_lora(
            lora_checkpoint_path=str(nemo_path),
            output_path=str(merged_path),
        )
        nemo_path = merged_path
        print(f"Merged checkpoint saved to: {merged_path}")
        
        # Export to HuggingFace format
        print(f"Exporting to {config.target} format...")
        export_kwargs = {}
        if config.use_modelopt and config.modelopt_export_kwargs:
            export_kwargs['modelopt_export_kwargs'] = config.modelopt_export_kwargs
        
        exported_path = api.export_ckpt(
            path=nemo_path,
            target=config.target,
            output_path=output_path,
            overwrite=config.overwrite,
            **export_kwargs
        )
        
        print(f"\n✓ Export completed successfully!")
        print(f"HuggingFace checkpoint saved to: {exported_path}")
        
        # Display directory structure
        if exported_path.exists():
            print("\nExported files:")
            for item in sorted(exported_path.iterdir()):
                if item.is_file():
                    size_mb = item.stat().st_size / (1024 * 1024)
                    print(f"  • {item.name} ({size_mb:.2f} MB)")
                elif item.is_dir():
                    print(f"  • {item.name}/ [directory]")
        
        return exported_path
        
    except Exception as e:
        print(f"\n✗ Export failed with error:")
        print(f"{str(e)}")
        raise


def main():
    """Main entry point for the export script."""
    parser = create_parser_from_dataclass(ExportConfig)
    args = parser.parse_args()
    
    # Create config from arguments
    config = ExportConfig.from_args(args)
    
    # Perform export
    try:
        exported_path = export_nemo_to_hf(config)
        print(f"\nExport completed successfully!")
        if config.no_merge:
            print(f"\nYou can now use this LoRA adapter with HuggingFace PEFT:")
            print(f"from peft import PeftModel")
            print(f"from transformers import AutoModelForCausalLM")
            print(f"base_model = AutoModelForCausalLM.from_pretrained('base_model')")
            print(f"model = PeftModel.from_pretrained(base_model, '{exported_path}')")
        else:
            print(f"\nYou can now use this checkpoint with HuggingFace Transformers:")
            print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
            print(f"model = AutoModelForCausalLM.from_pretrained('{exported_path}')")
            print(f"tokenizer = AutoTokenizer.from_pretrained('{exported_path}')")
        
    except Exception as e:
        print(f"\nExport failed!")
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
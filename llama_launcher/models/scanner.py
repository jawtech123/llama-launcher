"""Model Filesystem Scanner."""
import os
from pathlib import Path
from .types import ModelInfo


def scan_models(directory: str) -> list[ModelInfo]:
    """Scan filesystem for model files (.gguf, .bin, .safetensors)."""
    models = []
    dir_path = Path(directory)
    
    if not dir_path.exists() or not dir_path.is_dir():
        return models
    
    # Supported extensions
    extensions = ['.gguf', '.bin', '.safetensors']
    
    for ext in extensions:
        for file_path in dir_path.rglob(f"*{ext}"):
            if file_path.is_file():
                model_info = _parse_model_file(file_path)
                if model_info:
                    models.append(model_info)
                    
    return models


def _parse_model_file(file_path: Path) -> Optional[ModelInfo]:
    """Parse a model file and extract metadata."""
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        file_name = file_path.name
        
        # Determine format
        if file_path.suffix.lower() == '.gguf':
            file_format = 'gguf'
        elif file_path.suffix.lower() == '.bin':
            file_format = 'bin'
        elif file_path.suffix.lower() == '.safetensors':
            file_format = 'safetensors'
        else:
            file_format = 'unknown'
        
        # Infer family and quant type from filename
        family = None
        quant_type = None
        
        name_lower = file_name.lower()
        
        # Detect family
        if 'llama' in name_lower or 'llm' in name_lower:
            family = 'llama'
        elif 'mistral' in name_lower:
            family = 'mistral'
        elif 'qwen' in name_lower or 'qwen2' in name_lower or 'qwq' in name_lower:
            family = 'qwen'
        elif 'codellama' in name_lower:
            family = 'codellama'
        elif 'phi' in name_lower:
            family = 'phi'
        elif 'gemma' in name_lower:
            family = 'gemma'
        else:
            family = 'unknown'
        
        # Detect quant type
        quant_patterns = [
            ('q8_0', 'Q8_0'),
            ('q6_k', 'Q6_K'),
            ('q5_k_m', 'Q5_K_M'),
            ('q5_0', 'Q5_0'),
            ('q5_1', 'Q5_1'),
            ('q4_k_m', 'Q4_K_M'),
            ('q4_0', 'Q4_0'),
            ('q4_1', 'Q4_1'),
            ('q3_k_m', 'Q3_K_M'),
            ('q3_k_s', 'Q3_K_S'),
            ('q2_k', 'Q2_K'),
        ]
        
        for pattern, quant_name in quant_patterns:
            if pattern in name_lower:
                quant_type = quant_name
                break
        
        return ModelInfo(
            file_path=str(file_path),
            file_name=file_name,
            file_size_mb=round(file_size_mb, 2),
            format=file_format,
            family=family,
            quant_type=quant_type
        )
        
    except Exception:
        return None

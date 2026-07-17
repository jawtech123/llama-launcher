"""Quantization Recommendation Engine."""
from typing import Dict, Any


def recommend_quantization(model_family: str, vram_mb: int, target: str = "balanced") -> Dict[str, Any]:
    """Recommend quantization level based on model family, VRAM, and target."""
    recommendations = {
        "q2_k": {"quality": "low", "size_multiplier": 0.35},
        "q3_k_m": {"quality": "low-mid", "size_multiplier": 0.45},
        "q4_0": {"quality": "mid", "size_multiplier": 0.5},
        "q4_k_m": {"quality": "mid-high", "size_multiplier": 0.55},
        "q5_0": {"quality": "high", "size_multiplier": 0.65},
        "q5_k_m": {"quality": "high", "size_multiplier": 0.65},
        "q6_k": {"quality": "very-high", "size_multiplier": 0.75},
        "q8_0": {"quality": "lossless", "size_multiplier": 0.95}
    }
    
    # Model size estimates (in GB for base models)
    model_sizes = {
        "llama": {"3b": 2.0, "7b": 4.0, "8b": 4.5, "13b": 7.0, "34b": 19.0, "70b": 40.0},
        "mistral": {"7b": 4.0, "8b": 4.5},
        "qwen": {"7b": 4.0, "14b": 8.0, "32b": 18.0, "72b": 40.0},
        "default": {"7b": 4.0, "8b": 4.5, "13b": 7.0, "34b": 19.0}
    }
    
    # Determine base model size
    base_size_gb = 4.0  # Default 7B/8B model
    family_lower = model_family.lower()
    
    for model_type, sizes in model_sizes.items():
        if model_type in family_lower:
            # Try to extract model size from family name
            for size_key in sizes.keys():
                if size_key in family_lower:
                    base_size_gb = sizes[size_key]
                    break
            else:
                base_size_gb = list(sizes.values())[0]
            break
    
    base_size_mb = base_size_gb * 1024
    
    # Find recommended quantization
    recommended_q = "q4_k_m"
    explanation = f"Q4_K_M is recommended for {model_family} models with {vram_mb}MB VRAM. Provides best balance of quality and size."
    fallback_options = ["q3_k_m", "q5_k_m"]
    
    # Adjust based on VRAM and target
    if target == "speed":
        # Prefer smaller quantizations for speed
        if vram_mb < 8192:
            recommended_q = "q3_k_m"
            explanation = f"Q3_K_M recommended for {model_family} with {vram_mb}MB VRAM. Optimized for faster inference with acceptable quality loss."
            fallback_options = ["q2_k", "q4_0"]
        elif vram_mb < 12288:
            recommended_q = "q4_0"
            explanation = f"Q4_0 recommended for {model_family} with {vram_mb}MB VRAM. Faster inference with good quality."
            fallback_options = ["q3_k_m", "q4_k_m"]
        else:
            recommended_q = "q4_k_m"
            fallback_options = ["q3_k_m", "q5_k_m"]
            
    elif target == "quality":
        # Prefer larger quantizations for quality
        if vram_mb >= 24576:
            recommended_q = "q6_k"
            explanation = f"Q6_K recommended for {model_family} with {vram_mb}MB VRAM. Highest quality with minimal size increase over Q5."
            fallback_options = ["q5_k_m", "q8_0"]
        elif vram_mb >= 12288:
            recommended_q = "q5_k_m"
            explanation = f"Q5_K_M recommended for {model_family} with {vram_mb}MB VRAM. Near-lossless quality with good size efficiency."
            fallback_options = ["q6_k", "q4_k_m"]
        else:
            recommended_q = "q4_k_m"
            fallback_options = ["q5_k_m", "q3_k_m"]
            
    else:  # balanced
        if vram_mb < 6144:
            recommended_q = "q3_k_m"
            explanation = f"Q3_K_M recommended for {model_family} with {vram_mb}MB VRAM. Necessary for smaller VRAM configurations."
            fallback_options = ["q2_k", "q4_0"]
        elif vram_mb < 12288:
            recommended_q = "q4_k_m"
            explanation = f"Q4_K_M recommended for {model_family} with {vram_mb}MB VRAM. Best balance of quality and size for 12GB class GPUs."
            fallback_options = ["q3_k_m", "q5_k_m"]
        elif vram_mb < 24576:
            recommended_q = "q5_k_m"
            explanation = f"Q5_K_M recommended for {model_family} with {vram_mb}MB VRAM. Excellent quality with 24GB class GPUs."
            fallback_options = ["q6_k", "q4_k_m"]
        else:
            recommended_q = "q6_k"
            explanation = f"Q6_K recommended for {model_family} with {vram_mb}MB VRAM. Near-lossless quality for 24GB+ GPUs."
            fallback_options = ["q5_k_m", "q8_0"]
    
    # Check if recommended quantization fits in VRAM
    est_size_mb = base_size_mb * recommendations[recommended_q]["size_multiplier"]
    if est_size_mb > vram_mb * 0.8:
        # Recommend smaller quantization
        for q in ["q3_k_m", "q4_0", "q2_k"]:
            if base_size_mb * recommendations[q]["size_multiplier"] <= vram_mb * 0.8:
                recommended_q = q
                explanation = f"Adjusted to {q.upper()} due to VRAM constraints. Estimated size: ~{int(est_size_mb)}MB."
                break
    
    return {
        "recommended_quantization": recommended_q,
        "explanation": explanation,
        "fallback_options": fallback_options,
        "estimated_size_mb": int(base_size_mb * recommendations[recommended_q]["size_multiplier"]),
        "quality_level": recommendations[recommended_q]["quality"]
    }

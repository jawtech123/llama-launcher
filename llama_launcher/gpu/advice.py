"""Which llama.cpp build to use for the detected GPU, and the knobs it needs.

Pure functions over :class:`GPUInfo`, shown on the GPU Info tab. The advice
is deliberately conservative: it names the backend and the build flag, and
the one environment variable AMD users most often need.
"""
import re
from typing import List, Tuple

from .types import GPUInfo


#: AMD targets ROCm's HIP runtime does not ship kernels for, with the
#: compatible target to pretend to be via HSA_OVERRIDE_GFX_VERSION.
_GFX_OVERRIDES = {
    # RDNA2 derivatives (6700/6600/6500 series, Ryzen 6000-8000 iGPUs)
    "gfx1031": "10.3.0", "gfx1032": "10.3.0", "gfx1033": "10.3.0", "gfx1034": "10.3.0",
    "gfx1035": "10.3.0", "gfx1036": "10.3.0",
    # RDNA3 derivatives (7700/7600, 780M/880M iGPUs)
    "gfx1101": "11.0.0", "gfx1102": "11.0.0", "gfx1103": "11.0.0",
    "gfx1150": "11.0.0", "gfx1151": "11.0.0", "gfx1152": "11.0.0",
}


def suggest_backend(gpu: GPUInfo) -> Tuple[str, List[str]]:
    """Return ``(backend title, detail lines)`` for the GPU Info tab."""
    if gpu is None:
        return ("Unknown", ["Connect to the server to detect its GPU."])

    if gpu.backend == "cuda" or gpu.vendor == "nvidia":
        lines = ["Build llama.cpp with -DGGML_CUDA=ON (or use a prebuilt CUDA release)."]
        cc = (gpu.compute_capability or "").strip()
        if cc:
            arch = cc.replace(".", "")
            lines.append(f"Compute capability {cc}: add -DCMAKE_CUDA_ARCHITECTURES={arch} "
                         f"for a faster, smaller build.")
        if gpu.notes:
            lines.extend(gpu.notes)
        return ("CUDA", lines)

    if gpu.vendor == "amd":
        targets = sorted({c.compute_capability for c in (gpu.gpu_infos or [gpu])
                          if c.compute_capability})
        if gpu.rocm_available or gpu.backend == "hip":
            lines = ["ROCm found: build llama.cpp with -DGGML_HIP=ON"
                     + (f" -DAMDGPU_TARGETS={';'.join(targets)}" if targets else "")
                     + ". The Vulkan build (-DGGML_VULKAN=ON) also works and needs no ROCm."]
            for t in targets:
                override = _GFX_OVERRIDES.get(t)
                if override:
                    lines.append(
                        f"{t} is not an official ROCm target: if llama-server fails to start, set "
                        f"HSA_OVERRIDE_GFX_VERSION={override} in Server Management > Edit Server "
                        f"> Environment.")
            if not targets:
                lines.append("If llama-server reports an unsupported gfx target, set "
                             "HSA_OVERRIDE_GFX_VERSION (10.3.0 for RDNA2, 11.0.0 for RDNA3) in "
                             "Server Management > Edit Server > Environment.")
            title = "HIP (ROCm)"
        else:
            lines = ["No ROCm tooling on this host: build llama.cpp with -DGGML_VULKAN=ON "
                     "(needs the Vulkan SDK / mesa-vulkan-drivers). Install ROCm for the "
                     "usually faster HIP build."]
            title = "Vulkan"
        if any(c.integrated for c in (gpu.gpu_infos or [gpu])):
            lines.append("An integrated GPU shares system RAM; its reported VRAM is only "
                         "the reserved carve-out. Expect CPU-like speeds.")
        lines.extend(n for n in gpu.notes if n not in lines)
        return (title, lines)

    if gpu.backend == "metal":
        return ("Metal", ["Apple Silicon: the default macOS build uses Metal with unified memory."])
    if gpu.backend == "vulkan":
        return ("Vulkan", ["Build llama.cpp with -DGGML_VULKAN=ON."] + list(gpu.notes))
    if gpu.backend == "sycl":
        return ("SYCL", ["Intel GPU: build with -DGGML_SYCL=ON (oneAPI), or use Vulkan."])
    return ("CPU", ["No GPU detected; llama-server will run on the CPU. "
                    "Use a small quantization and keep the context modest."])


def cuda_arch_flag(compute_capability: str) -> str:
    """"12.0" -> "120", for CMAKE_CUDA_ARCHITECTURES."""
    return re.sub(r"[^0-9]", "", compute_capability or "")

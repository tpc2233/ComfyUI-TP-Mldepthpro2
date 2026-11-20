import hashlib
import logging
from pathlib import Path
import tempfile
from urllib.request import urlopen

from filelock import FileLock
from safetensors import safe_open
import torch

import folder_paths
from comfy import model_management as mm, model_patcher, utils

from .depth_pro_2 import DepthPro2, PRESETS


ASSETS = {
    "S": (97264508, "583c4d630779fd231f969c738733624963ad34bd3c29f9c64088257e3436f97c"),
    "B": (386076596, "244ff85e0d9bde6e0f985de1c61420abc4c2110265b495c974556225ac2b60ed"),
    "L": (1335873268, "d799d8a91f2e8dc91715f9b33a405a1ff0727212041d206ebbb04e6c1424ff9f"),
    "H": (3488493012, "cfc532336019abd2128b04be2c2967175522e46de69222ef72b87c6675ee3b17"),
    "focal": (93840880, "8f567d1c97ff916137288668f6bd306ce2094da2e2b852b0e4539dbb139c8c86"),
}
VARIANTS = {"S": "S", "B": "B", "L": "L", "H+": "H"}
PRECISIONS = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

folder_paths.add_model_folder_path("depth_pro_2", str(Path(folder_paths.models_dir) / "depth_pro_2"))
folder_paths.folder_names_and_paths["depth_pro_2"][1].add(".safetensors")


def ensure_weights(asset):
    size, checksum = ASSETS[asset]
    filename = f"depthpro2-{asset.lower()}.safetensors"
    existing = folder_paths.get_full_path("depth_pro_2", filename)
    if existing is not None:
        return existing
    destination = Path(folder_paths.get_folder_paths("depth_pro_2")[0]) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if destination.is_file():
            return str(destination)
        url = f"https://huggingface.co/apple/DepthPro2-{asset}/resolve/main/model.safetensors"
        logging.info("Depth Pro 2: downloading %s (%.0f MB)", filename, size / 1e6)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as f:
            temporary = Path(f.name)
        try:
            digest = hashlib.sha256()
            received = 0
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    mm.throw_exception_if_processing_interrupted()
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
            if received != size or digest.hexdigest() != checksum:
                raise RuntimeError(f"Depth Pro 2: checksum mismatch for {filename}; rerun to download again.")
            with safe_open(str(temporary), framework="pt", device="cpu") as weights:
                if not list(weights.keys()):
                    raise RuntimeError(f"Empty checkpoint: {filename}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return str(destination)


def load_model(variant, precision, estimate_focal, device):
    asset = VARIANTS[variant]
    load_device = mm.get_torch_device() if device == "auto" else torch.device("cpu")
    if device not in ("auto", "cpu"):
        raise ValueError(f"Unknown device: {device}")
    if precision == "auto":
        dtype = torch.float32
        if mm.should_use_bf16(load_device):
            dtype = torch.bfloat16
        elif mm.should_use_fp16(load_device):
            dtype = torch.float16
    else:
        dtype = PRECISIONS[precision]
    path = ensure_weights(asset)
    focal_path = ensure_weights("focal") if estimate_focal else None
    config = PRESETS[f"depthpro2-{asset.lower()}"]
    model = DepthPro2(
        backbone_preset=config.backbone_preset,
        reassembled_blocks=config.reassembled_blocks,
        cross_frame_blocks=config.cross_frame_blocks,
        dims_reassemble=config.dims_reassemble,
        dim_decoder=config.dim_decoder,
        fov=estimate_focal,
        precision=dtype,
    )
    weights = utils.load_torch_file(path, safe_load=True)
    if focal_path is not None:
        weights.update({f"fov.{key}": value for key, value in utils.load_torch_file(focal_path, safe_load=True).items()})
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model_patcher.ModelPatcher(model, load_device=load_device, offload_device=torch.device("cpu"))

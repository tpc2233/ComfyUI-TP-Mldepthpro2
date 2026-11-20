import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

import folder_paths
from comfy import model_management as mm, utils
from comfy.ldm.colormap import turbo
from comfy_api.latest import io

from .model_loading import load_model


DepthModel = io.Custom("TP_MLDEPTHPRO2_MODEL")
MetricDepth = io.Custom("TP_MLDEPTHPRO2_DEPTH")
CATEGORY = "TP/Depth Pro 2"


class TPDepthPro2Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TPDepthPro2Loader", display_name="Load Depth Pro 2 (TP)", category=CATEGORY,
            description="Downloads missing Apple weights into models/depth_pro_2 when executed. Uses ComfyUI model management.",
            inputs=[
                io.Combo.Input("model", options=["S", "B", "L", "H+"], default="L"),
                io.Combo.Input("precision", options=["auto", "bf16", "fp16", "fp32"], default="auto",
                               tooltip="Auto selects BF16 where supported. Model heads and stored weights remain float32."),
                io.Boolean.Input("estimate_focal", default=True,
                                 tooltip="Load the shared focal estimator. Disable only when supplying a known focal length."),
                io.Combo.Input("device", options=["auto", "cpu"], default="auto",
                               tooltip="Auto uses the device selected by ComfyUI."),
            ],
            outputs=[DepthModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, precision="auto", estimate_focal=True, device="auto"):
        return io.NodeOutput(load_model(model, precision, estimate_focal, device))


class TPDepthPro2Video(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TPDepthPro2Video", display_name="Depth Pro 2 Video (TP)", category=CATEGORY,
            description="Treats the IMAGE batch as one ordered video. Preserves resolution and frame count. A single image also works. Each execution starts a fresh temporal session.",
            inputs=[
                DepthModel.Input("model"), io.Image.Input("images"),
                io.Combo.Input("mode", options=["buffered", "unbuffered"], default="buffered"),
                io.Int.Input("window", default=16, min=1, max=1024,
                             tooltip="Buffered clip length, or unbuffered attention history including the current frame."),
                io.Int.Input("overlap", default=4, min=0, max=1023,
                             tooltip="Buffered only: must be smaller than window. Ignored in unbuffered mode."),
                io.Float.Input("focal_length_px", default=0.0, min=0.0, max=1000000.0, step=1.0,
                               tooltip="0 estimates one focal length over the first window. Otherwise use calibrated fx in pixels at the input resolution."),
            ],
            outputs=[MetricDepth.Output(display_name="depth"), io.Float.Output(display_name="focal_length_px")],
        )

    @classmethod
    def execute(cls, model, images, mode="buffered", window=16, overlap=4, focal_length_px=0.0):
        if images.ndim != 4 or images.shape[-1] not in (3, 4) or not images.is_floating_point():
            raise ValueError("Expected ComfyUI IMAGE: float [frames, height, width, 3 or 4] in RGB order.")
        frames_count, height, width, _ = images.shape
        if min(frames_count, height, width) < 1:
            raise ValueError("The IMAGE batch is empty.")
        if mode not in ("buffered", "unbuffered") or window < 1:
            raise ValueError("Choose buffered or unbuffered mode and window >= 1.")
        if mode == "buffered" and not 0 <= overlap < window:
            raise ValueError("Buffered overlap must be >= 0 and smaller than window.")
        if not math.isfinite(focal_length_px) or focal_length_px < 0:
            raise ValueError("Focal length must be 0 (estimate) or a positive finite number in pixels.")
        net = model.model
        if focal_length_px == 0 and net.fov is None:
            raise ValueError("Enable estimate_focal in the loader, or supply focal_length_px > 0.")
        padded_pixels = math.ceil(height / 32) * 32 * math.ceil(width / 32) * 32
        active_window = window if mode == "unbuffered" else min(frames_count, window)
        memory_required = padded_pixels * active_window * max(256, net.decoder.dim_out * 2)
        mm.load_models_gpu([model], memory_required=memory_required, force_full_load=True)
        device = model.load_device
        mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device).view(3, 1, 1)

        def frames():
            for frame in images:
                mm.throw_exception_if_processing_interrupted()
                yield (frame[..., :3].movedim(-1, 0).to(device=device, dtype=torch.float32).contiguous() - mean) / std

        meters = torch.empty((frames_count, height, width), device="cpu", dtype=torch.float32)
        progress = utils.ProgressBar(frames_count)
        focal = None
        count = 0
        predictions = net.infer_video(frames(), window=window, overlap=overlap,
                                      f_px=focal_length_px or None, mode=mode)
        try:
            for count, prediction in enumerate(predictions, start=1):
                mm.throw_exception_if_processing_interrupted()
                meters[count - 1].copy_(prediction["depth"])
                focal = float(prediction["focallength_px"])
                progress.update(1)
        finally:
            predictions.close()
        if count != frames_count:
            raise RuntimeError(f"Depth Pro 2 returned {count} frames for {frames_count} input frames.")
        return io.NodeOutput({"depth": meters, "focallength_px": focal}, focal)


class TPDepthPro2Preview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TPDepthPro2Preview", display_name="Depth Pro 2 Preview (TP)", category=CATEGORY,
            description="Outputs a standard IMAGE batch. Uses a fixed scale for the whole video; raw metric depth stays unchanged.",
            inputs=[
                MetricDepth.Input("depth"),
                io.Combo.Input("style", options=["grayscale", "turbo"], default="grayscale"),
                io.Combo.Input("normalization", options=["first_frame", "whole_video", "metric_range"], default="first_frame",
                               tooltip="First frame matches the upstream scale with 20% headroom. Whole video includes later close objects. Metric range uses near_m/far_m."),
                io.Float.Input("near_m", default=0.5, min=0.001, max=1000000.0, step=0.1),
                io.Float.Input("far_m", default=100.0, min=0.001, max=1000000.0, step=1.0),
                io.Boolean.Input("invert", default=False, tooltip="Default grayscale is white near, black far."),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, depth, style="grayscale", normalization="first_frame", near_m=0.5, far_m=100.0, invert=False):
        meters = depth["depth"]
        if style not in ("grayscale", "turbo"):
            raise ValueError(f"Unknown preview style: {style}")
        if normalization == "metric_range":
            if not (math.isfinite(near_m) and math.isfinite(far_m) and 0 < near_m < far_m):
                raise ValueError("Metric range requires 0 < near_m < far_m, both finite.")
            offset, scale = 1 / far_m, 1 / near_m - 1 / far_m
        elif normalization in ("first_frame", "whole_video"):
            reference = meters[:1] if normalization == "first_frame" else meters
            offset = 0.0
            scale = max(float(frame.reciprocal().max()) for frame in reference) * 1.2
            if scale == 0:
                scale = 1.0
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        output = torch.empty((*meters.shape, 3), device="cpu", dtype=torch.float32)
        for index, frame in enumerate(meters):
            mm.throw_exception_if_processing_interrupted()
            normalized = ((frame.reciprocal() - offset) / scale).clamp(0, 1)
            if invert:
                normalized = 1 - normalized
            output[index] = turbo(normalized) if style == "turbo" else normalized.unsqueeze(-1).expand(-1, -1, 3)
        return io.NodeOutput(output)


class TPDepthPro2RawImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TPDepthPro2RawImage", display_name="Depth Pro 2 Raw Float32 (TP)", category=CATEGORY,
            description="Unclipped float32 depth as an IMAGE batch for data-aware EXR savers. Values are repeated in R/G/B; no normalization or color conversion. For PNG16, use the grayscale Preview node instead.",
            inputs=[
                MetricDepth.Input("depth"),
                io.Combo.Input("encoding", options=["meters", "inverse_meters"], default="meters",
                               tooltip="Meters preserves camera-z depth, including infinity for sky. Inverse meters is 1/depth, with sky at zero. Neither mode is restricted to 0–1."),
            ],
            outputs=[io.Image.Output(display_name="images_float32")],
        )

    @classmethod
    def execute(cls, depth, encoding="meters"):
        values = depth["depth"].to(device="cpu", dtype=torch.float32)
        if encoding == "inverse_meters":
            values = values.reciprocal()
        elif encoding != "meters":
            raise ValueError(f"Unknown depth encoding: {encoding}")
        return io.NodeOutput(values.unsqueeze(-1).repeat(1, 1, 1, 3))


class TPDepthPro2SaveDepth(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TPDepthPro2SaveDepth", display_name="Save Depth Pro 2 Metric Depth (TP)", category=CATEGORY,
            description="Saves float32 depth in meters and focal length as one NPZ per frame under ComfyUI/output. Sky may be infinite.",
            inputs=[MetricDepth.Input("depth"), io.String.Input("filename_prefix", default="depth_pro_2/depth")],
            outputs=[io.String.Output(display_name="directory")], is_output_node=True,
        )

    @classmethod
    def execute(cls, depth, filename_prefix="depth_pro_2/depth"):
        meters = depth["depth"]
        folder, filename, _, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), meters.shape[2], meters.shape[1])
        destination = Path(tempfile.mkdtemp(prefix=filename + "_", dir=folder))
        focal = np.float32(depth["focallength_px"])
        progress = utils.ProgressBar(len(meters))
        for index, frame in enumerate(meters):
            mm.throw_exception_if_processing_interrupted()
            np.savez(destination / f"{index:06d}.npz", depth=frame.numpy(), focallength_px=focal)
            progress.update(1)
        (destination / "metadata.json").write_text(json.dumps({
            "units": "meters", "depth_axis": "camera_z", "frames": len(meters),
            "height": meters.shape[1], "width": meters.shape[2], "focallength_px": float(focal),
        }, indent=2) + "\n")
        return io.NodeOutput(str(destination))

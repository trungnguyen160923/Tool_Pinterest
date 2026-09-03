from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


class ClipEmbedder:
    """Optional CLIP backend loaded lazily so classic mode has no heavy deps."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "auto") -> None:
        try:
            import torch
            from transformers import CLIPModel
            try:
                from transformers import CLIPImageProcessorPil as ImageProcessor
            except ImportError:
                from transformers import CLIPImageProcessor as ImageProcessor
        except Exception as exc:
            raise RuntimeError(
                "CLIP mode requires torch and transformers. Install task6 requirements first."
            ) from exc

        self.torch = torch
        self.model_id = model_id
        self.name = f"clip:{model_id}"
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.processor = self._from_pretrained(ImageProcessor, model_id)
        self.model = self._from_pretrained(CLIPModel, model_id).to(self.device)
        self.model.eval()

    @staticmethod
    def _from_pretrained(cls, model_id: str):
        try:
            return cls.from_pretrained(model_id, local_files_only=True)
        except Exception:
            return cls.from_pretrained(model_id)

    def _embedding_tensor(self, output):
        if self.torch.is_tensor(output):
            return output

        for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
            value = getattr(output, attr, None)
            if value is None:
                continue
            if attr == "last_hidden_state":
                return value[:, 0]
            return value

        if isinstance(output, (tuple, list)) and output:
            for value in output:
                if self.torch.is_tensor(value):
                    return value[:, 0] if value.ndim == 3 else value

        raise RuntimeError(f"Unsupported CLIP output type: {type(output).__name__}")

    def embed_image(self, path: Path) -> list[float]:
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            features = self._embedding_tensor(self.model.get_image_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
        return [float(value) for value in features[0].detach().cpu().tolist()]


def create_embedder(mode: str, model_id: str, device: str = "auto") -> ClipEmbedder | None:
    if mode == "classic":
        return None
    try:
        return ClipEmbedder(model_id=model_id, device=device)
    except Exception:
        if mode == "clip":
            raise
        return None

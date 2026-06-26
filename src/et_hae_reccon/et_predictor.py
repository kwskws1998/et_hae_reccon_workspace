"""Frozen emotion ET predictor adapters for ET-HAE data preparation."""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np
import torch

from et_hae_reccon.constants import FEATURE_NAMES, TRT_FEATURE

SKBOY_EMOTION_ET_2ND_REPO_ID = "skboy/emotion_et_2nd_model"
SKBOY_EMOTION_ET_2ND_SUBFOLDER = "hf_emotion_et_aug_lr2e-5_len256_seed123"
SKBOY_EMOTION_ET_2ND_WEIGHTS = "et_predictor2_iitb_sa1_sa2_lr2e5_len256_seed123.safetensors"


@dataclass(frozen=True)
class PredictedETWord:
    word: str
    trt: float
    features: dict[str, float]


class WordETPredictor(Protocol):
    def predict_words(self, text: str) -> list[PredictedETWord]:
        ...


class HFTokenTRTRegressor(torch.nn.Module):
    """RoBERTa-style encoder with a scalar TRT head."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        freeze_encoder: bool = False,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.decoder = torch.nn.Linear(self.encoder.config.hidden_size, 1)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.decoder(hidden).squeeze(-1)


@dataclass(frozen=True)
class HFArtifactLocation:
    weights_filename: str
    subfolder: str | None


def resolve_hf_artifact_location(
    weights_filename: str,
    subfolder: str | None = None,
) -> HFArtifactLocation:
    weights_path = PurePosixPath(weights_filename)
    if weights_path.parent != PurePosixPath("."):
        inferred = str(weights_path.parent)
        if subfolder is not None and subfolder != inferred:
            raise ValueError("weights_filename subfolder conflicts with subfolder.")
        subfolder = inferred
        weights_filename = weights_path.name
    return HFArtifactLocation(weights_filename=weights_filename, subfolder=subfolder)


class SkboyEmotionETPredictor:
    """Frozen adapter around the exported skboy emotion ET predictor."""

    def __init__(
        self,
        repo_id: str = SKBOY_EMOTION_ET_2ND_REPO_ID,
        weights_filename: str = SKBOY_EMOTION_ET_2ND_WEIGHTS,
        subfolder: str | None = SKBOY_EMOTION_ET_2ND_SUBFOLDER,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        from huggingface_hub import hf_hub_download, snapshot_download

        location = resolve_hf_artifact_location(weights_filename, subfolder)
        download_kwargs: dict[str, str | bool] = {
            "repo_id": repo_id,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            download_kwargs["cache_dir"] = str(cache_dir)
        if location.subfolder is not None:
            download_kwargs["subfolder"] = location.subfolder
        model_py = hf_hub_download(filename="model.py", **download_kwargs)
        module = load_module(Path(model_py))
        if not hasattr(module, "load_et_predictor") or not hasattr(module, "predict_word_features"):
            raise ImportError("Expected load_et_predictor and predict_word_features in model.py.")
        patterns = ["*"] if location.subfolder is None else [f"{location.subfolder}/*"]
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
            allow_patterns=patterns,
        )
        model_dir = Path(snapshot_dir)
        if location.subfolder is not None:
            model_dir = model_dir / location.subfolder
        self.model, self.tokenizer = module.load_et_predictor(
            model_dir,
            weight_name=location.weights_filename,
        )
        self.predict_word_features_fn = module.predict_word_features

    def predict_words(self, text: str) -> list[PredictedETWord]:
        words, features = self.predict_word_features_fn(text, self.model, self.tokenizer)
        feature_array = np.asarray(features, dtype=np.float64)
        if feature_array.ndim != 2 or feature_array.shape[1] < len(FEATURE_NAMES):
            raise ValueError("ET predictor returned an unexpected feature matrix shape.")
        trt_index = FEATURE_NAMES.index(TRT_FEATURE)
        rows: list[PredictedETWord] = []
        for word, feature_row in zip(words, feature_array):
            features_dict = {
                name: float(feature_row[index])
                for index, name in enumerate(FEATURE_NAMES)
            }
            rows.append(PredictedETWord(word=str(word), trt=float(feature_row[trt_index]), features=features_dict))
        if not rows:
            raise ValueError("ET predictor returned no words.")
        return rows


class HeuristicETPredictor:
    """Deterministic fallback predictor for offline smoke tests."""

    def predict_words(self, text: str) -> list[PredictedETWord]:
        rows: list[PredictedETWord] = []
        for raw_word in re.findall(r"\S+", text):
            cleaned = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", raw_word)
            score = max(0.0, len(cleaned) / 10.0)
            rows.append(
                PredictedETWord(
                    word=raw_word,
                    trt=score,
                    features={
                        "nFix": 1.0 + score,
                        "FFD": score,
                        "GPT": score,
                        "TRT": score,
                        "fixProp": min(1.0, score),
                    },
                )
            )
        if not rows:
            raise ValueError("Cannot predict ET for empty text.")
        return rows


class CheckpointTRTPredictor:
    """Frozen adapter for checkpoints produced by the TRT-only training repo."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str | None = None,
        device: str | torch.device = "auto",
        local_files_only: bool = False,
    ) -> None:
        from transformers import AutoTokenizer

        self.device = resolve_device(str(device))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        self.model_name = model_name or str(checkpoint.get("model_name", args.get("model_name", "roberta-base")))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                add_prefix_space=True,
                local_files_only=local_files_only,
            )
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_files_only)
        self.model = HFTokenTRTRegressor(model_name=self.model_name, local_files_only=local_files_only).to(self.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict_words(self, text: str) -> list[PredictedETWord]:
        words = text.strip().split()
        if not words:
            raise ValueError("Cannot predict TRT for empty text.")
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        predictions = self.model(input_ids=input_ids, attention_mask=attention_mask).squeeze(0).clamp_min(0.0)
        word_ids = encoded.word_ids(batch_index=0)
        output = np.zeros((len(words),), dtype=np.float64)
        seen: set[int] = set()
        for token_index, word_index in enumerate(word_ids):
            if word_index is None or word_index in seen or word_index >= len(words):
                continue
            output[word_index] = float(predictions[token_index].detach().cpu())
            seen.add(word_index)
        return [predicted_et_word(word, value) for word, value in zip(words, output)]


class HFExportTRTPredictor:
    """Frozen adapter for exported TRT-only Hugging Face model folders."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        repo_id: str | None = None,
        weight_name: str | None = None,
        subfolder: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        device: str | torch.device = "auto",
    ) -> None:
        if model_dir is None:
            if repo_id is None:
                raise ValueError("repo_id or model_dir is required for trt_hf_export backend.")
            from huggingface_hub import snapshot_download

            snapshot_dir = Path(
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    local_files_only=local_files_only,
                    allow_patterns=["*"] if subfolder is None else [f"{subfolder}/*"],
                )
            )
            model_dir = snapshot_dir / subfolder if subfolder is not None else snapshot_dir
        self.model_dir = Path(model_dir)
        module = load_module(self.model_dir / "model.py")
        if not hasattr(module, "load_et_predictor"):
            raise ImportError("Expected load_et_predictor in exported model.py.")
        self.predict_trt_fn = getattr(module, "predict_word_trt", None)
        self.predict_features_fn = getattr(module, "predict_word_features", None)
        if self.predict_trt_fn is None and self.predict_features_fn is None:
            raise ImportError("Expected predict_word_trt or predict_word_features in exported model.py.")
        weight_name = weight_name or getattr(module, "DEFAULT_WEIGHT", None)
        self.model, self.tokenizer = module.load_et_predictor(
            self.model_dir,
            weight_name=weight_name,
            device=resolve_device(str(device)),
        )

    def predict_words(self, text: str) -> list[PredictedETWord]:
        if self.predict_trt_fn is not None:
            words, trt = self.predict_trt_fn(text, self.model, self.tokenizer)
        else:
            words, features = self.predict_features_fn(text, self.model, self.tokenizer)
            feature_array = np.asarray(features, dtype=np.float64)
            trt_index = 0 if feature_array.shape[1] == 1 else FEATURE_NAMES.index(TRT_FEATURE)
            trt = feature_array[:, trt_index]
        return [predicted_et_word(str(word), float(value)) for word, value in zip(words, trt)]


def predicted_et_word(word: str, trt: float) -> PredictedETWord:
    clean_trt = max(0.0, float(trt))
    features = {name: 0.0 for name in FEATURE_NAMES}
    features[TRT_FEATURE] = clean_trt
    return PredictedETWord(word=word, trt=clean_trt, features=features)


def load_word_et_predictor(
    backend: str,
    repo_id: str = SKBOY_EMOTION_ET_2ND_REPO_ID,
    weights_filename: str = SKBOY_EMOTION_ET_2ND_WEIGHTS,
    subfolder: str | None = SKBOY_EMOTION_ET_2ND_SUBFOLDER,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    trt_checkpoint_path: str | Path | None = None,
    trt_model_name: str | None = None,
    trt_model_dir: str | Path | None = None,
    trt_repo_id: str | None = None,
    trt_weight_name: str | None = None,
    trt_subfolder: str | None = None,
    device: str | torch.device = "auto",
) -> WordETPredictor:
    if backend == "skboy":
        return SkboyEmotionETPredictor(
            repo_id=repo_id,
            weights_filename=weights_filename,
            subfolder=subfolder,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if backend == "heuristic":
        return HeuristicETPredictor()
    if backend == "trt_checkpoint":
        if trt_checkpoint_path is None:
            raise ValueError("trt_checkpoint_path is required for trt_checkpoint backend.")
        return CheckpointTRTPredictor(
            checkpoint_path=trt_checkpoint_path,
            model_name=trt_model_name,
            device=device,
            local_files_only=local_files_only,
        )
    if backend == "trt_hf_export":
        return HFExportTRTPredictor(
            model_dir=trt_model_dir,
            repo_id=trt_repo_id,
            weight_name=trt_weight_name,
            subfolder=trt_subfolder,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            device=device,
        )
    raise ValueError(f"Unsupported ET predictor backend: {backend}")


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("emotion_et_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import ET predictor module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

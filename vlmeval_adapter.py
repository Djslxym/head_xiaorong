"""VLMEvalKit adapters for HAWK with optional per-sample dynamic Head weights."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .config import HawkConfig
from .integration import configure_model, get_last_pruning_stats


def _get_sample_id(message: list[dict[str, Any]]) -> int | None:
    """Extract RealWorldQA sample ID from an image filename such as 500.webp."""
    for part in message:
        if part.get("type") != "image":
            continue

        value = str(part.get("value", ""))
        filename = Path(value.split("?")[0]).stem

        match = re.search(r"(\d+)$", filename)
        if match:
            return int(match.group(1))

    return None


def _load_dynamic_weights(
    prediction_path: str,
) -> dict[int, tuple[float, ...]]:
    """Load per-sample predicted Head weights."""
    path = Path(prediction_path).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.is_file():
        raise FileNotFoundError(
            f"Dynamic prediction file not found: {path}"
        )

    print(f"[HAWK Dynamic] Loading predictions: {path}", flush=True)

    try:
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        data = torch.load(path, map_location="cpu")

    if not isinstance(data, dict):
        raise TypeError("预测文件必须是包含 sample_ids 和预测权重的字典")

    if "sample_ids" not in data:
        raise KeyError("预测文件缺少 sample_ids")

    sample_ids = torch.as_tensor(data["sample_ids"]).reshape(-1).tolist()

    # 兼容不同预测文件中的权重字段名称。
    prediction_keys = (
        "predictions",
        "predicted_weights",
        "test_predictions",
        "weights",
    )

    predictions = None

    for key in prediction_keys:
        if key in data:
            candidate = torch.as_tensor(data[key])
            if candidate.ndim == 2:
                predictions = candidate
                break

    # 如果字段名称不同，自动寻找形状为 N×28 的二维张量。
    if predictions is None:
        for key, value in data.items():
            try:
                candidate = torch.as_tensor(value)
            except (TypeError, ValueError):
                continue

            if candidate.ndim == 2 and candidate.shape[1] == 28:
                predictions = candidate
                print(
                    f"[HAWK Dynamic] Found prediction tensor: {key}",
                    flush=True,
                )
                break

    if predictions is None:
        raise ValueError("预测文件中没有找到形状为 N×28 的权重张量")

    if len(sample_ids) != predictions.shape[0]:
        raise ValueError(
            f"样本数量不一致：sample_ids={len(sample_ids)}, "
            f"predictions={predictions.shape[0]}"
        )

    if predictions.shape[1] != 28:
        raise ValueError(
            f"Head 权重维度错误：期望 28，实际 {predictions.shape[1]}"
        )

    result: dict[int, tuple[float, ...]] = {}

    for sample_id, weights in zip(sample_ids, predictions):
        sample_id = int(sample_id)
        result[sample_id] = tuple(
            float(x) for x in weights.tolist()
        )

    print(
        f"[HAWK Dynamic] Loaded {len(result)} samples, "
        f"{predictions.shape[1]} Heads",
        flush=True,
    )

    return result


class HawkQwen2VLChat:
    """Factory-compatible HAWK wrapper around VLMEvalKit's Qwen2VLChat."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

        class _ConfiguredHawkQwen2VLChat(Qwen2VLChat):

            def __init__(
                self,
                *model_args: Any,
                hawk_keep_ratio: float = 0.2,
                hawk_head_weights: list[float] | tuple[float, ...] | None = None,
                hawk_score_normalization: str = "l2",
                hawk_min_visual_tokens: int = 1,
                hawk_trace_path: str | None = None,
                hawk_dynamic_weights_path: str | None = "outputs/head_predictor/test_predictions.pt",
                **model_kwargs: Any,
            ) -> None:

                self._hawk_keep_ratio = hawk_keep_ratio
                self._hawk_trace_path = hawk_trace_path

                super().__init__(*model_args, **model_kwargs)

                # 每个 worker 只使用一张 GPU，将模型放到对应 GPU。
                self.model.to("cuda")

                configure_model(
                    self.model,
                    HawkConfig(
                        keep_ratio=hawk_keep_ratio,
                        **(
                            {"head_weights": tuple(hawk_head_weights)}
                            if hawk_head_weights is not None
                            else {}
                        ),
                        score_normalization=hawk_score_normalization,
                        min_visual_tokens=hawk_min_visual_tokens,
                    ),
                )

                # 加载一次预测结果，后续样本直接查字典。
                self._hawk_dynamic_weights = None

                if (
                    hawk_dynamic_weights_path
                    and __import__("os").environ.get(
                        "HAWK_DISABLE_DYNAMIC_WEIGHTS", "0"
                    ) != "1"
                ):
                    self._hawk_dynamic_weights = _load_dynamic_weights(
                        hawk_dynamic_weights_path
                    )

            def _restore_original_prompt_for_vlmeval(
                self,
                original_generate: Any,
                *generate_args: Any,
                **generate_kwargs: Any,
            ) -> torch.Tensor:

                generated = original_generate(
                    *generate_args,
                    **generate_kwargs,
                )

                stats = get_last_pruning_stats(self.model)
                input_ids = generate_kwargs.get("input_ids")

                if (
                    stats is None
                    or input_ids is None
                    or generated.ndim != 2
                    or input_ids.ndim != 2
                ):
                    return generated

                kept_prompt = int(stats["kept_total_tokens"])
                generated_only = generated[:, kept_prompt:]

                return torch.cat(
                    [
                        input_ids.to(generated.device),
                        generated_only,
                    ],
                    dim=1,
                )

            def generate_inner_transformers(
                self,
                message: list[dict[str, Any]],
                dataset: str | None = None,
            ) -> str:

                original_generate = self.model.generate

                # integration.py 将配置保存在 self.model.model 上。
                hawk_model = self.model.model
                original_config = hawk_model._hawk_config

                sample_id = _get_sample_id(message)

                is_realworldqa = (
                    dataset is not None
                    and "realworldqa" in str(dataset).lower()
                )

                dynamic_weights = None

                if is_realworldqa and self._hawk_dynamic_weights is not None:

                    if sample_id is None:
                        raise RuntimeError(
                            "无法从图片路径识别 RealWorldQA 样本编号"
                        )

                    dynamic_weights = self._hawk_dynamic_weights.get(
                        sample_id
                    )

                    if dynamic_weights is None:
                        raise RuntimeError(
                            f"预测文件中找不到样本 {sample_id} 的动态权重"
                        )

                try:
                    if dynamic_weights is not None:

                        # 为当前样本临时设置自己的 28 维动态权重。
                        hawk_model._hawk_config = replace(
                            original_config,
                            head_weights=dynamic_weights,
                        )

                        print(
                            f"[HAWK Dynamic] sample_id={sample_id}, "
                            f"heads={len(dynamic_weights)}, "
                            f"keep_ratio={original_config.keep_ratio}",
                            flush=True,
                        )

                    def generate_with_original_prompt(
                        *generate_args: Any,
                        **generate_kwargs: Any,
                    ) -> torch.Tensor:

                        return self._restore_original_prompt_for_vlmeval(
                            original_generate,
                            *generate_args,
                            **generate_kwargs,
                        )

                    self.model.generate = generate_with_original_prompt

                    response = super().generate_inner_transformers(
                        message,
                        dataset=dataset,
                    )

                finally:
                    # 恢复 generate 函数和原始 HAWK 配置。
                    self.model.generate = original_generate
                    hawk_model._hawk_config = original_config

                self._write_hawk_trace(
                    message,
                    dataset,
                    response,
                    sample_id,
                )

                return response

            def _write_hawk_trace(
                self,
                message: list[dict[str, Any]],
                dataset: str | None,
                response: str,
                sample_id: int | None = None,
            ) -> None:

                if not self._hawk_trace_path:
                    return

                rank = int(os.environ.get("RANK", "0"))

                trace_base = Path(self._hawk_trace_path)
                trace_path = trace_base.with_name(
                    f"{trace_base.name}.rank{rank}.jsonl"
                )

                trace_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                text = "\n".join(
                    str(part["value"])
                    for part in message
                    if part.get("type") == "text"
                )

                images = [
                    str(part["value"])
                    for part in message
                    if part.get("type") == "image"
                ]

                record = {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "question_sha256": hashlib.sha256(
                        text.encode()
                    ).hexdigest(),
                    "images": images,
                    "response": response,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                    "pruning": get_last_pruning_stats(self.model),
                }

                with trace_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

        return _ConfiguredHawkQwen2VLChat(*args, **kwargs)


class HawkLocalImageMCQDataset:
    """Factory for an ImageMCQDataset backed by a prepared local TSV file."""

    MODALITY = "IMAGE"

    def __init__(
        self,
        dataset: str,
        data_file: str,
        skip_noimg: bool = True,
    ) -> None:
        del dataset, data_file, skip_noimg

    def __new__(
        cls,
        dataset: str,
        data_file: str,
        skip_noimg: bool = True,
    ) -> Any:

        from vlmeval.dataset.image_mcq import ImageMCQDataset
        from vlmeval.smp import load

        class _LocalImageMCQDataset(ImageMCQDataset):

            def __init__(self) -> None:
                self._hawk_data_file = str(
                    Path(data_file).expanduser().resolve()
                )
                super().__init__(
                    dataset=dataset,
                    skip_noimg=skip_noimg,
                )

            def load_data(self, dataset_name: str) -> Any:
                del dataset_name
                self.data_path = self._hawk_data_file
                data = load(self._hawk_data_file)

                # RealWorldQA 只使用测试集 index 500～764。
                if str(dataset).lower() == "realworldqa":
                    if "index" not in data.columns:
                        raise RuntimeError(
                            "RealWorldQA TSV 缺少 index 列"
                        )

                    data = data[
                        data["index"].astype(int) >= 500
                    ].copy()

                    data = data.reset_index(drop=True)

                    print(
                        f"[HAWK Dynamic] RealWorldQA test split: "
                        f"{len(data)} samples, "
                        f"index={data['index'].min()}~"
                        f"{data['index'].max()}",
                        flush=True,
                    )

                return data

        return _LocalImageMCQDataset()


class HawkLocalImageVQADataset:
    """Factory for an ImageVQADataset backed by a prepared local TSV file."""

    MODALITY = "IMAGE"

    def __init__(
        self,
        dataset: str,
        data_file: str,
        skip_noimg: bool = True,
    ) -> None:
        del dataset, data_file, skip_noimg

    def __new__(
        cls,
        dataset: str,
        data_file: str,
        skip_noimg: bool = True,
    ) -> Any:

        from vlmeval.dataset.image_vqa import ImageVQADataset
        from vlmeval.smp import load

        class _LocalImageVQADataset(ImageVQADataset):

            def __init__(self) -> None:
                self._hawk_data_file = str(
                    Path(data_file).expanduser().resolve()
                )
                super().__init__(
                    dataset=dataset,
                    skip_noimg=skip_noimg,
                )

            def load_data(self, dataset_name: str) -> Any:
                del dataset_name
                self.data_path = self._hawk_data_file
                return load(self._hawk_data_file)

        return _LocalImageVQADataset()


class HawkLocalImageYORNDataset:
    """Factory for an ImageYORNDataset backed by a prepared local TSV file."""

    MODALITY = "IMAGE"

    def __init__(
        self,
        dataset: str,
        data_file: str,
        skip_noimg: bool = True,
    ) -> None:
        del dataset, data_file, skip_noimg

    def __new__(
        cls,
        dataset: str,
        data_file: str,
        skip_noimg: bool = True,
    ) -> Any:

        from vlmeval.dataset.image_yorn import ImageYORNDataset
        from vlmeval.smp import load

        class _LocalImageYORNDataset(ImageYORNDataset):

            def __init__(self) -> None:
                self._hawk_data_file = str(
                    Path(data_file).expanduser().resolve()
                )
                super().__init__(
                    dataset=dataset,
                    skip_noimg=skip_noimg,
                )

            def load_data(self, dataset_name: str) -> Any:
                del dataset_name
                self.data_path = self._hawk_data_file
                return load(self._hawk_data_file)

        return _LocalImageYORNDataset()

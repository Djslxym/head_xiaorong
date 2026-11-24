import os
import json
import argparse
from pathlib import Path

import pandas as pd


# ============================================================
# 1. 参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frozen Qwen2.5-VL image/text features."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen2.5-VL-7B-Instruct"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/vlmeval/RealWorldQA_local.tsv"
    )

    parser.add_argument(
        "--image_root",
        type=str,
        default="data/vlmeval/images/RealWorldQA"
    )

    parser.add_argument(
        "--label_path",
        type=str,
        default="outputs/head_labels/realworldqa_layer0_202611192314.jsonl"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/vlm_features"
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="2"
    )

    parser.add_argument(
        "--train_size",
        type=int,
        default=500
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug only: maximum samples per split. 0 means all."
    )

    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Re-extract all cached samples."
    )

    return parser.parse_args()


# ============================================================
# 2. GPU 设置
# ============================================================

def configure_gpu(gpu_id):

    gpu_id = str(gpu_id).strip()

    if "," in gpu_id:
        raise ValueError(
            "当前程序只支持单 GPU，例如 --gpus 2"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    print("\n========== GPU Configuration ==========")
    print("Physical GPU:", gpu_id)
    print("CUDA_VISIBLE_DEVICES:", gpu_id)


# ============================================================
# 3. 加载模型
# ============================================================

def load_model(model_path):

    import torch
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor
    )

    print("\n========== Loading Processor ==========")

    processor = AutoProcessor.from_pretrained(model_path)

    print("\n========== Loading Model on CPU ==========")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    model.eval()

    print("Model dtype:", next(model.parameters()).dtype)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    print("\n========== Moving Model to GPU ==========")

    model = model.to(torch.device("cuda:0"))

    torch.cuda.synchronize()

    # 冻结全部参数
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    embedding = model.model.language_model.embed_tokens

    print("\n========== Model Verification ==========")

    print("Visual encoder:", type(model.visual))
    print("Text embedding:", type(embedding))
    print("Hidden size:", model.config.hidden_size)
    print("Visual output size:", model.config.vision_config.out_hidden_size)

    print("Visual device:", next(model.visual.parameters()).device)
    print("Embedding device:", embedding.weight.device)

    parameter_devices = sorted(
        set(str(p.device) for p in model.parameters())
    )

    print("All parameter devices:", parameter_devices)

    if parameter_devices != ["cuda:0"]:
        raise RuntimeError(
            f"Model is not fully on cuda:0: {parameter_devices}"
        )

    if model.config.hidden_size != 3584:
        raise RuntimeError(
            f"Unexpected hidden size: {model.config.hidden_size}"
        )

    if model.config.vision_config.out_hidden_size != 3584:
        raise RuntimeError("Unexpected visual output dimension.")

    if embedding.weight.device.type != "cuda":
        raise RuntimeError("Embedding is not on CUDA.")

    allocated = torch.cuda.memory_allocated(0)

    print(
        "GPU allocated:",
        round(allocated / 1024**3, 2),
        "GB"
    )

    print("Model loading verification PASSED.")

    return model, processor


# ============================================================
# 4. 构建与标签生成程序一致的问题文本
# ============================================================

def build_question_text(row):

    question = str(row["question"]).strip()

    option_lines = []

    for option in ["A", "B", "C", "D"]:

        value = row.get(option, "")

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            option_lines.append(f"{option}. {value}")

    options_text = "\n".join(option_lines)

    question_text = (
        f"Question: {question}\n"
        f"Options:\n"
        f"{options_text}\n"
        f"Please select the correct answer from the options above. "
    )

    return question_text


# ============================================================
# 5. 构建多模态消息
# ============================================================

def build_messages(question_text, image_path):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path)
                },
                {
                    "type": "text",
                    "text": question_text
                }
            ]
        }
    ]

    return messages


# ============================================================
# 6. 图片路径处理
# ============================================================

def resolve_image_path(row, image_root):

    image_name = str(row["image_path"]).strip()

    image_path = Path(image_name)

    if not image_path.is_absolute():
        image_path = Path(image_root) / image_path

    if not image_path.exists():

        fallback_path = Path(image_root) / Path(image_name).name

        if fallback_path.exists():
            image_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

    return image_path.resolve()


# ============================================================
# 7. 图片特征提取
# ============================================================

def extract_visual_feature(
    messages,
    processor,
    model
):

    import torch

    from qwen_vl_utils import process_vision_info

    device = torch.device("cuda:0")

    # 与 generate_head_labels.py 使用相同的视觉预处理方式
    image_inputs, video_inputs = process_vision_info(messages)

    # 与标签生成程序一致的 Chat Template
    prefix_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    common_kwargs = {
        "images": image_inputs,
        "return_tensors": "pt",
        "padding": True,
    }

    if video_inputs is not None:
        common_kwargs["videos"] = video_inputs

    model_inputs = processor(
        text=[prefix_text],
        **common_kwargs
    )

    if "pixel_values" not in model_inputs:
        raise RuntimeError("Processor did not return pixel_values.")

    if "image_grid_thw" not in model_inputs:
        raise RuntimeError("Processor did not return image_grid_thw.")

    # 视觉编码器输入
    visual_dtype = next(model.visual.parameters()).dtype

    pixel_values = model_inputs["pixel_values"].to(
        device=device,
        dtype=visual_dtype
    )

    image_grid_thw = model_inputs["image_grid_thw"].to(
        device=device
    )

    # 只运行视觉编码器，不运行语言模型 Transformer
    with torch.inference_mode():

        visual_tokens = model.visual(
            pixel_values,
            image_grid_thw
        )

        if not torch.is_tensor(visual_tokens):
            raise RuntimeError(
                "Visual encoder output is not a Tensor."
            )

        if visual_tokens.ndim != 2:
            raise RuntimeError(
                f"Unexpected visual feature shape: "
                f"{tuple(visual_tokens.shape)}"
            )

        if visual_tokens.shape[-1] != 3584:
            raise RuntimeError(
                f"Unexpected visual dimension: "
                f"{visual_tokens.shape[-1]}"
            )

        # 对视觉 Token 做平均池化
        visual_feature = visual_tokens.float().mean(dim=0)

    if not torch.isfinite(visual_feature).all():
        raise RuntimeError("Visual feature contains NaN or Inf.")

    return visual_feature.cpu()


# ============================================================
# 8. 文本特征提取
# ============================================================

def extract_text_feature(
    question_text,
    processor,
    model
):

    import torch

    device = torch.device("cuda:0")

    # 这里只编码问题和选项，不包含标准答案
    encoded = processor.tokenizer(
        question_text,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False
    )

    input_ids = encoded["input_ids"].to(device)

    attention_mask = encoded["attention_mask"].to(device)

    embedding = model.model.language_model.embed_tokens

    with torch.inference_mode():

        # 直接获取 Token Embedding
        text_tokens = embedding(input_ids)

        # 转成 FP32 后计算平均值
        text_tokens = text_tokens.float()

        mask = attention_mask.unsqueeze(-1).float()

        text_sum = (text_tokens * mask).sum(dim=1)

        valid_token_count = mask.sum(dim=1).clamp(min=1.0)

        text_feature = text_sum / valid_token_count

        text_feature = text_feature.squeeze(0)

    if text_feature.shape != (3584,):
        raise RuntimeError(
            f"Unexpected text feature shape: "
            f"{tuple(text_feature.shape)}"
        )

    if not torch.isfinite(text_feature).all():
        raise RuntimeError("Text feature contains NaN or Inf.")

    return text_feature.cpu()


# ============================================================
# 9. 单条样本特征提取
# ============================================================

def extract_one_sample(
    row,
    image_path,
    processor,
    model
):

    import torch

    question_text = build_question_text(row)

    messages = build_messages(
        question_text,
        image_path
    )

    visual_feature = extract_visual_feature(
        messages,
        processor,
        model
    )

    text_feature = extract_text_feature(
        question_text,
        processor,
        model
    )

    # 两个分支分别得到 3584 维
    # 拼接为 7168 维
    fused_feature = torch.cat(
        [visual_feature, text_feature],
        dim=0
    )

    if fused_feature.shape != (7168,):
        raise RuntimeError(
            f"Unexpected fused feature shape: "
            f"{tuple(fused_feature.shape)}"
        )

    return {
        "visual_feature": visual_feature.float(),
        "text_feature": text_feature.float(),
        "fused_feature": fused_feature.float()
    }


# ============================================================
# 10. 加载标签并验证样本对应关系
# ============================================================

def load_labels(label_path, train_sample_ids):

    labels = {}

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            sample_id = int(record["sample_id"])

            if sample_id in labels:
                raise RuntimeError(
                    f"Duplicate label sample_id: {sample_id}"
                )

            if record.get("status") != "success":
                raise RuntimeError(
                    f"Label is not successful: {sample_id}"
                )

            if not record.get("valid", False):
                raise RuntimeError(
                    f"Invalid head label: {sample_id}"
                )

            weights = record.get("target_weights")

            if weights is None or len(weights) != 28:
                raise RuntimeError(
                    f"Invalid target_weights: {sample_id}"
                )

            labels[sample_id] = weights

    expected = set(train_sample_ids)
    actual = set(labels.keys())

    missing = expected - actual
    extra = actual - expected

    if missing or extra:

        raise RuntimeError(
            "Label/sample ID mismatch.\n"
            f"Missing labels: {sorted(missing)[:20]}\n"
            f"Extra labels: {sorted(extra)[:20]}"
        )

    print("\n========== Label Verification ==========")
    print("Label count:", len(labels))
    print("All training sample IDs matched.")

    return labels


# ============================================================
# 11. 单条样本缓存
# ============================================================

def cache_path_for(output_dir, sample_id):

    return (
        Path(output_dir)
        / "cache"
        / f"sample_{sample_id:06d}.pt"
    )


def save_sample_cache(path, sample_id, features):

    import torch

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tmp_path = path.with_suffix(".tmp")

    torch.save(
        {
            "sample_id": int(sample_id),
            "visual_feature": features["visual_feature"],
            "text_feature": features["text_feature"],
            "fused_feature": features["fused_feature"]
        },
        tmp_path
    )

    os.replace(tmp_path, path)


def load_sample_cache(path, sample_id):

    import torch

    try:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        if int(data["sample_id"]) != int(sample_id):
            return None

        if data["visual_feature"].shape != (3584,):
            return None

        if data["text_feature"].shape != (3584,):
            return None

        if data["fused_feature"].shape != (7168,):
            return None

        if not torch.isfinite(data["fused_feature"]).all():
            return None

        return data

    except Exception:
        return None


# ============================================================
# 12. 提取一个数据划分
# ============================================================

def extract_split(
    split_name,
    dataframe,
    labels,
    args,
    model,
    processor
):

    import torch

    print(f"\n========== Extracting {split_name} ==========")
    print("Samples:", len(dataframe))

    output_dir = Path(args.output_dir)

    cache_dir = output_dir / "cache"

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    errors = []

    for position, (_, row) in enumerate(dataframe.iterrows()):

        sample_id = int(row["index"])

        cache_path = cache_path_for(
            output_dir,
            sample_id
        )

        if cache_path.exists() and not args.overwrite_cache:

            cached = load_sample_cache(
                cache_path,
                sample_id
            )

            if cached is not None:

                results.append(cached)

                if (position + 1) % 10 == 0:
                    print(
                        f"[{split_name}] "
                        f"{position + 1}/{len(dataframe)} "
                        f"loaded from cache"
                    )

                continue

        try:

            image_path = resolve_image_path(
                row,
                args.image_root
            )

            features = extract_one_sample(
                row,
                image_path,
                processor,
                model
            )

            save_sample_cache(
                cache_path,
                sample_id,
                features
            )

            result = {
                "sample_id": sample_id,
                **features
            }

            results.append(result)

            if (position + 1) % 10 == 0 or position == 0:

                torch.cuda.synchronize()

                print(
                    f"[{split_name}] "
                    f"{position + 1}/{len(dataframe)} "
                    f"sample_id={sample_id} "
                    f"visual={tuple(features['visual_feature'].shape)} "
                    f"text={tuple(features['text_feature'].shape)}"
                )

        except Exception as e:

            errors.append(
                {
                    "sample_id": sample_id,
                    "error": repr(e)
                }
            )

            print(
                f"\n[ERROR] split={split_name}, "
                f"sample_id={sample_id}: {repr(e)}"
            )

            import traceback
            traceback.print_exc()

        finally:

            torch.cuda.synchronize()

    if errors:

        error_path = output_dir / f"{split_name}_errors.json"

        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(
                errors,
                f,
                ensure_ascii=False,
                indent=2
            )

        raise RuntimeError(
            f"{split_name} has {len(errors)} failed samples. "
            f"Successful samples remain cached. "
            f"Rerun the program to retry failures. "
            f"Error file: {error_path}"
        )

    results.sort(
        key=lambda x: x["sample_id"]
    )

    expected_ids = sorted(
        int(row["index"])
        for _, row in dataframe.iterrows()
    )

    actual_ids = [
        int(item["sample_id"])
        for item in results
    ]

    if actual_ids != expected_ids:
        raise RuntimeError(
            f"{split_name} sample order or IDs do not match."
        )

    # 整理张量
    visual_features = torch.stack(
        [item["visual_feature"] for item in results]
    ).float()

    text_features = torch.stack(
        [item["text_feature"] for item in results]
    ).float()

    fused_features = torch.stack(
        [item["fused_feature"] for item in results]
    ).float()

    package = {
        "split": split_name,
        "sample_ids": torch.tensor(
            actual_ids,
            dtype=torch.long
        ),
        "visual_features": visual_features,
        "text_features": text_features,
        "fused_features": fused_features,
        "questions": [
            str(row["question"])
            for _, row in dataframe.iterrows()
        ],
        "image_paths": [
            str(resolve_image_path(row, args.image_root))
            for _, row in dataframe.iterrows()
        ],
    }

    # 训练集额外保存标签
    if split_name == "train":

        package["target_weights"] = torch.tensor(
            [
                labels[sample_id]
                for sample_id in actual_ids
            ],
            dtype=torch.float32
        )

    output_path = output_dir / f"{split_name}_features.pt"

    tmp_path = output_path.with_suffix(".tmp")

    torch.save(package, tmp_path)

    os.replace(tmp_path, output_path)

    print(f"\n{split_name} extraction completed.")
    print("Saved:", output_path)
    print("Visual:", tuple(visual_features.shape))
    print("Text:", tuple(text_features.shape))
    print("Fused:", tuple(fused_features.shape))

    if "target_weights" in package:
        print(
            "Target weights:",
            tuple(package["target_weights"].shape)
        )

    return package


# ============================================================
# 13. 主函数
# ============================================================

def main():

    args = parse_args()

    configure_gpu(args.gpus)

    import torch

    torch.manual_seed(42)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    print("\n========== Loading Dataset ==========")

    dataframe = pd.read_csv(
        args.data_path,
        sep="\t"
    )

    print("Total samples:", len(dataframe))
    print("Columns:", dataframe.columns.tolist())

    required_columns = [
        "index",
        "question",
        "image_path",
        "A",
        "B",
        "C",
        "D"
    ]

    for column in required_columns:

        if column not in dataframe.columns:
            raise RuntimeError(
                f"Missing dataset column: {column}"
            )

    if len(dataframe) != 765:
        raise RuntimeError(
            f"Expected 765 RealWorldQA samples, "
            f"got {len(dataframe)}."
        )

    if dataframe["index"].duplicated().any():
        raise RuntimeError("Duplicate dataset index found.")

    train_size = args.train_size

    if train_size != 500:
        raise RuntimeError(
            "This experiment is configured for 500 training samples."
        )

    if len(dataframe) <= train_size:
        raise RuntimeError("No independent test samples.")

    train_df = dataframe.iloc[:train_size].copy()

    test_df = dataframe.iloc[train_size:].copy()

    if args.limit > 0:

        train_df = train_df.iloc[:args.limit].copy()

        test_df = test_df.iloc[:args.limit].copy()

    train_ids = [
        int(x)
        for x in train_df["index"].tolist()
    ]

    test_ids = [
        int(x)
        for x in test_df["index"].tolist()
    ]

    if set(train_ids) & set(test_ids):
        raise RuntimeError("Training/test sample overlap.")

    print("\n========== Dataset Split ==========")
    print("Training samples:", len(train_df))
    print("Training IDs:", train_ids[0], "to", train_ids[-1])
    print("Test samples:", len(test_df))
    print("Test IDs:", test_ids[0], "to", test_ids[-1])

    # --------------------------------------------------------
    # 加载标签
    # --------------------------------------------------------

    labels = load_labels(
        args.label_path,
        [
            int(x)
            for x in dataframe.iloc[:train_size]["index"].tolist()
        ]
    )

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------

    model, processor = load_model(
        args.model_path
    )

    # --------------------------------------------------------
    # 提取训练集特征
    # --------------------------------------------------------

    train_package = extract_split(
        "train",
        train_df,
        labels,
        args,
        model,
        processor
    )

    # --------------------------------------------------------
    # 提取测试集特征
    # --------------------------------------------------------

    test_package = extract_split(
        "test",
        test_df,
        labels,
        args,
        model,
        processor
    )

    # --------------------------------------------------------
    # 最终检查
    # --------------------------------------------------------

    print("\n========== FINAL VERIFICATION ==========")

    print(
        "Train fused:",
        tuple(train_package["fused_features"].shape)
    )

    print(
        "Train targets:",
        tuple(train_package["target_weights"].shape)
    )

    print(
        "Test fused:",
        tuple(test_package["fused_features"].shape)
    )

    # 根据 --limit 动态确定预期样本数量
    if args.limit is None or args.limit <= 0:
        expected_train_size = 500
        expected_test_size = 265
    else:
        expected_train_size = min(500, args.limit)
        expected_test_size = min(265, args.limit)

    print("\n========== EXPECTED SHAPES ==========")
    print(f"Expected train: ({expected_train_size}, 7168)")
    print(f"Expected targets: ({expected_train_size}, 28)")
    print(f"Expected test: ({expected_test_size}, 7168)")

    assert train_package["fused_features"].shape == (
        expected_train_size, 7168
    ), "Train feature shape mismatch"

    assert train_package["target_weights"].shape == (
        expected_train_size, 28
    ), "Train target shape mismatch"

    assert test_package["fused_features"].shape == (
        expected_test_size, 7168
    ), "Test feature shape mismatch"

    train_ids = set(train_package["sample_ids"].tolist())
    test_ids = set(test_package["sample_ids"].tolist())

    assert train_ids.isdisjoint(test_ids), (
        "Training and test sample IDs overlap!"
    )

    print("\nAll feature extraction checks PASSED.")

    print("\nOutput files:")
    print(output_dir / "train_features.pt")
    print(output_dir / "test_features.pt")


if __name__ == "__main__":
    main()

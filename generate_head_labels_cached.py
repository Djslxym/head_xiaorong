import os
import json
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sample-wise Attention Head importance labels."
    )

    parser.add_argument("--model_path", type=str,
                        default="models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data_path", type=str,
                        default="data/vlmeval/RealWorldQA_local.tsv")
    parser.add_argument("--image_root", type=str,
                        default="data/vlmeval/images/RealWorldQA")
    parser.add_argument("--output", type=str,
                        default="outputs/head_labels/realworldqa_layer0.jsonl")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Maximum number of samples to process. 0 means all samples."
    )
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--eps", type=float, default=1e-12)

    return parser.parse_args()


def load_model(model_path, device_map):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print("\n========== Loading Model ==========")
    processor = AutoProcessor.from_pretrained(model_path)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device_map,
        attn_implementation="sdpa",
    )

    model.eval()
    print("Model loaded successfully.")
    return model, processor


def build_messages(row, image_path):
    """
    Construct the RealWorldQA prompt.
    """

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

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question_text}
            ]
        }
    ]


def prepare_model_inputs(messages, answer, processor, model):
    """
    Prepare multimodal FULL inputs.

    The ground-truth answer is appended to the prompt. Its token at
    answer_position is predicted by logits at answer_position - 1.
    This preserves the original teacher-forced NLL calculation.
    """

    import torch
    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages)

    prefix_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    full_messages = messages + [
        {"role": "assistant", "content": answer}
    ]

    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    common_kwargs = {
        "images": image_inputs,
        "return_tensors": "pt",
        "padding": True,
    }
    if video_inputs is not None:
        common_kwargs["videos"] = video_inputs

    # Keep the original Prefix/Full token alignment verification.
    prefix_inputs = processor(text=[prefix_text], **common_kwargs)
    full_inputs = processor(text=[full_text], **common_kwargs)

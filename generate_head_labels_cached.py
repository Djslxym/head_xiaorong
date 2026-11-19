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

    prefix_ids = prefix_inputs["input_ids"]
    full_ids = full_inputs["input_ids"]
    prefix_len = prefix_ids.shape[1]

    if full_ids.shape[1] <= prefix_len:
        raise RuntimeError(
            "Full input is not longer than prefix input. "
            "Cannot locate answer token."
        )

    if not torch.equal(prefix_ids[0], full_ids[0, :prefix_len]):
        raise RuntimeError(
            "Prefix token IDs do not match the beginning of full input. "
            "Cannot safely determine answer token position."
        )

    answer_token_ids = processor.tokenizer(
        answer,
        add_special_tokens=False
    )["input_ids"]

    if len(answer_token_ids) != 1:
        raise RuntimeError(
            f"Answer {answer!r} is not a single token: {answer_token_ids}"
        )

    answer_position = prefix_len
    target_token_id = full_ids[0, answer_position].item()

    if target_token_id != answer_token_ids[0]:
        raise RuntimeError(
            "Answer token verification failed. "
            f"Expected {answer_token_ids[0]}, got {target_token_id}."
        )

    input_device = model.get_input_embeddings().weight.device
    model_inputs = {}

    for key, value in full_inputs.items():
        if torch.is_tensor(value):
            model_inputs[key] = value.to(input_device)
        else:
            model_inputs[key] = value

    return model_inputs, answer_position, target_token_id


def calculate_answer_loss(
    model,
    model_inputs,
    answer_position,
    target_token_id
):
    """
    Calculate NLL of the ground-truth answer token.
    The answer token at position t is predicted by logits at position t-1.
    """

    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        outputs = model(**model_inputs)

        answer_logits = outputs.logits[
            0, answer_position - 1, :
        ].float()

        target = torch.tensor(
            [target_token_id],
            dtype=torch.long,
            device=answer_logits.device
        )

        loss = F.cross_entropy(answer_logits.unsqueeze(0), target)
        loss_value = loss.item()

        del outputs, answer_logits, target, loss

    return loss_value


def install_visual_encoder_cache(model):
    """
    Cache the output of Qwen2.5-VL's visual module for the current sample.

    The first model forward of a sample runs the visual encoder and stores
    its output. The following baseline/ablation forwards reuse that output.
    The cache is cleared after each sample.

    This avoids depending on a Transformers-specific `mm_encoder_outputs`
    argument. It wraps the visual module's forward method directly.
    """

    visual = getattr(model.model, "visual", None)
    if visual is None:
        raise RuntimeError(
            "Cannot find model.model.visual. "
            "This script expects the standard Qwen2.5-VL module layout."
        )

    original_forward = visual.forward

    cache_state = {
        "enabled": False,
        "output": None,
        "vision_forward_calls": 0,
        "cache_hits": 0,
    }

    def detach_tree(value):
        import torch

        if torch.is_tensor(value):
            # detach() shares the original storage and avoids an extra
            # full-size GPU copy. The model is in eval/inference mode.
            return value.detach()
        if isinstance(value, tuple):
            return tuple(detach_tree(item) for item in value)
        if isinstance(value, list):
            return [detach_tree(item) for item in value]
        if isinstance(value, dict):
            return {key: detach_tree(item) for key, item in value.items()}
        return value

    def cached_visual_forward(*args, **kwargs):
        if not cache_state["enabled"]:
            return original_forward(*args, **kwargs)

        if cache_state["output"] is not None:
            cache_state["cache_hits"] += 1
            return cache_state["output"]

        result = original_forward(*args, **kwargs)
        cache_state["vision_forward_calls"] += 1
        cache_state["output"] = detach_tree(result)
        return result

    # `original_forward` is already a bound method, so the wrapper does not
    # need an explicit `self` argument.
    visual.forward = cached_visual_forward

    print("\n========== Visual Encoder Cache ==========")
    print("Visual module:", visual.__class__.__name__)
    print("Visual forward cache installed.")

    return cache_state, visual, original_forward


def clear_visual_cache(cache_state):
    """Clear cached visual features and disable cache mode."""
    cache_state["enabled"] = False
    cache_state["output"] = None
    cache_state["vision_forward_calls"] = 0
    cache_state["cache_hits"] = 0


def install_dynamic_head_hook(model, layer_index):
    """
    Install one dynamic hook on the selected layer's o_proj.

    state["head"] = None: normal forward.
    state["head"] = h: zero the selected Query Head's head_dim slice.
    """

    layers = model.model.language_model.layers

    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(
            f"Invalid layer index: {layer_index}. "
            f"Model has {len(layers)} layers."
        )

    attention = layers[layer_index].self_attn
    num_heads = attention.num_heads
    head_dim = attention.head_dim
    input_dim = attention.o_proj.in_features

    if input_dim != num_heads * head_dim:
        raise RuntimeError(
            "o_proj input dimension does not match "
            "num_heads * head_dim."
        )

    print("\n========== Attention Structure ==========")
    print("Layer:", layer_index)
    print("Query Heads:", num_heads)
    print("KV Heads:", attention.num_key_value_heads)
    print("Head Dimension:", head_dim)
    print("o_proj Input:", input_dim)
    print("o_proj Output:", attention.o_proj.out_features)

    state = {"head": None}

    def dynamic_ablation_hook(module, inputs):
        head_index = state["head"]
        if head_index is None:
            return None

        hidden_states = inputs[0]
        start = head_index * head_dim
        end = start + head_dim

        modified = hidden_states.clone()
        modified[..., start:end] = 0
        return (modified,) + tuple(inputs[1:])

    hook_handle = attention.o_proj.register_forward_pre_hook(
        dynamic_ablation_hook
    )

    print("Dynamic Head Ablation Hook installed.")
    return state, hook_handle, num_heads, head_dim


def calculate_sample_losses(
    model,
    model_inputs,
    answer_position,
    target_token_id,
    state,
    num_heads,
    visual_cache_state
):
    """
    Calculate one baseline loss and independent single-head ablation losses.

    The first (baseline) forward computes visual features and fills the cache.
    The subsequent 28 ablation forwards reuse those visual features.
    """

    # Start a fresh visual-feature cache for this sample.
    clear_visual_cache(visual_cache_state)
    visual_cache_state["enabled"] = True

    try:
        state["head"] = None
        baseline_loss = calculate_answer_loss(
            model, model_inputs, answer_position, target_token_id
        )

        head_losses = []

        for head_index in range(num_heads):
            state["head"] = head_index

            loss_value = calculate_answer_loss(
                model, model_inputs, answer_position, target_token_id
            )
            head_losses.append(loss_value)

            print(
                f"\rHead ablation: {head_index + 1}/{num_heads} "
                f"| Loss: {loss_value:.6f}",
                end="",
                flush=True
            )

        print()

        vision_calls = visual_cache_state["vision_forward_calls"]
        cache_hits = visual_cache_state["cache_hits"]

        print(
            f"Visual encoder calls for this sample: {vision_calls} "
            f"| Cached reuse calls: {cache_hits}"
        )

        if vision_calls != 1:
            print(
                "[Warning] Expected exactly one visual encoder call "
                f"during this sample, observed {vision_calls}."
            )

        return baseline_loss, head_losses

    finally:
        state["head"] = None
        clear_visual_cache(visual_cache_state)


def generate_labels(baseline_loss, head_losses, eps):
    """
    delta_h = max(0, L_h - L_base)
    q_h = delta_h / sum(delta)
    """

    deltas = [
        max(0.0, loss - baseline_loss)
        for loss in head_losses
    ]

    delta_sum = sum(deltas)

    if delta_sum <= eps:
        target_weights = [0.0] * len(deltas)
        valid = False
    else:
        target_weights = [
            delta / delta_sum for delta in deltas
        ]
        valid = True

    return deltas, target_weights, valid


def load_processed_ids(output_path):
    """Read existing successful JSONL results for resume support."""

    processed_ids = set()

    if not os.path.exists(output_path):
        return processed_ids

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("status") == "success":
                    processed_ids.add(str(record["sample_id"]))
            except Exception:
                continue

    return processed_ids


def append_jsonl(output_path, record):
    """Append one completed sample immediately."""

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def main():
    args = parse_args()

    print("========== Arguments ==========")
    print("Model:", args.model_path)
    print("Dataset:", args.data_path)
    print("Image root:", args.image_root)
    print("Layer:", args.layer)
    print("Limit:", args.limit)
    print("Output:", args.output)

    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(args.data_path)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    df = pd.read_csv(args.data_path, sep="\t")

    required_columns = [
        "answer", "question", "A", "B", "C", "D", "image_path"
    ]
    for column in required_columns:
        if column not in df.columns:
            raise RuntimeError(f"Missing dataset column: {column}")

    if args.limit > 0:
        df = df.head(args.limit)

    print("\n========== Dataset ==========")
    print("Samples to scan:", len(df))

    model, processor = load_model(args.model_path, args.device_map)

    state, hook_handle, num_heads, head_dim = (
        install_dynamic_head_hook(model, args.layer)
    )

    visual_cache_state, visual_module, original_visual_forward = (
        install_visual_encoder_cache(model)
    )

    processed_ids = load_processed_ids(args.output)
    print("\nAlready processed:", len(processed_ids))

    success_count = 0
    invalid_count = 0
    error_count = 0
    skipped_count = 0

    try:
        for position, (_, row) in enumerate(df.iterrows()):
            sample_id = str(
                row["index"] if "index" in df.columns else position
            )

            if sample_id in processed_ids:
                skipped_count += 1
                continue

            answer = str(row["answer"]).strip().upper()
            if answer not in ["A", "B", "C", "D"]:
                print(
                    f"\n[Skip] Sample {sample_id}: "
                    f"Invalid answer {answer!r}"
                )
                error_count += 1
                continue

            image_name = str(row["image_path"]).strip()
            image_path = Path(image_name)

            if not image_path.is_absolute():
                image_path = Path(args.image_root) / image_path

            if not image_path.exists():
                fallback_path = (
                    Path(args.image_root) / Path(image_name).name
                )
                if fallback_path.exists():
                    image_path = fallback_path
                else:
                    print(f"\n[Error] Image not found: {image_path}")
                    error_count += 1
                    continue

            print("\n" + "=" * 65)
            print(f"Sample {sample_id} ({position + 1}/{len(df)})")
            print("Question:", row["question"])
            print("Ground Truth:", answer)
            print("Image:", image_path)

            try:
                messages = build_messages(row, image_path)

                model_inputs, answer_position, target_token_id = (
                    prepare_model_inputs(
                        messages, answer, processor, model
                    )
                )

                baseline_loss, head_losses = calculate_sample_losses(
                    model,
                    model_inputs,
                    answer_position,
                    target_token_id,
                    state,
                    num_heads,
                    visual_cache_state
                )

                deltas, target_weights, valid = generate_labels(
                    baseline_loss, head_losses, args.eps
                )

                record = {
                    "sample_id": sample_id,
                    "question": str(row["question"]),
                    "answer": answer,
                    "image_path": str(image_path),
                    "layer": args.layer,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "baseline_loss": baseline_loss,
                    "head_losses": head_losses,
                    "deltas": deltas,
                    "target_weights": target_weights,
                    "valid": valid,
                    "status": "success"
                }

                append_jsonl(args.output, record)
                success_count += 1
                if not valid:
                    invalid_count += 1

                print("\n========== Sample Result ==========")
                print(f"Baseline loss: {baseline_loss:.6f}")
                print(f"Valid label: {valid}")
                print("Target weights:")
                print([round(x, 6) for x in target_weights])
                print(
                    f"Completed: {success_count} "
                    f"| Invalid: {invalid_count} "
                    f"| Errors: {error_count} "
                    f"| Skipped: {skipped_count}"
                )

                del model_inputs

            except Exception as e:
                error_count += 1
                print(
                    f"\n[Error] Sample {sample_id}: "
                    f"{type(e).__name__}: {e}"
                )
                state["head"] = None
                clear_visual_cache(visual_cache_state)

                append_jsonl(
                    args.output,
                    {
                        "sample_id": sample_id,
                        "status": "error",
                        "error": str(e)
                    }
                )

                import traceback
                traceback.print_exc()

    finally:
        state["head"] = None
        clear_visual_cache(visual_cache_state)
        hook_handle.remove()

        # Restore the original visual forward method.
        visual_module.forward = original_visual_forward

    print("\n" + "=" * 65)
    print("========== Finished ==========")
    print("Successful samples:", success_count)
    print("Invalid labels:", invalid_count)
    print("Errors:", error_count)
    print("Skipped existing samples:", skipped_count)
    print("Output:", args.output)


if __name__ == "__main__":
    main()

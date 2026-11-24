import os
import json
import argparse
from pathlib import Path

import pandas as pd



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sample-wise Attention Head importance labels."
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
        "--output",
        type=str,
        default="outputs/head_labels/realworldqa_layer0.jsonl"
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=0
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of samples. 0 means all samples."
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Physical GPU ID, e.g. 2"
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12
    )

    return parser.parse_args()


def configure_gpus(gpu_ids):
    """
    Configure visible physical GPU before importing torch.

    Example:
        --gpus 2

    Physical GPU 2 becomes logical cuda:0.
    """

    gpu_ids = str(gpu_ids).strip()

    if not gpu_ids:
        raise ValueError("--gpus cannot be empty.")

    gpu_list = [
        x.strip()
        for x in gpu_ids.split(",")
        if x.strip()
    ]

    if len(gpu_list) != 1:
        raise ValueError(
            "This script currently supports one GPU only. "
            "Please specify one GPU, e.g. --gpus 2."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[0]

    print("\n========== GPU Configuration ==========")
    print("Selected physical GPU:", gpu_list[0])
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

    return gpu_list[0]


def load_model(model_path):
    """
    Load model on CPU first, then explicitly move the entire
    model to cuda:0.

    This is the loading strategy verified successfully
    in the previous script.
    """

    import torch
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor
    )

    print("\n========== Loading Model ==========")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Visible GPU count:", torch.cuda.device_count())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if torch.cuda.device_count() < 1:
        raise RuntimeError("No visible CUDA GPU.")

    print("GPU:", torch.cuda.get_device_name(0))

    free_memory, total_memory = torch.cuda.mem_get_info(0)

    print(
        "GPU free memory:",
        round(free_memory / 1024**3, 2),
        "GB"
    )

    print(
        "GPU total memory:",
        round(total_memory / 1024**3, 2),
        "GB"
    )

    if free_memory < 12 * 1024**3:
        print(
            "[Warning] Available GPU memory is below 12 GB. "
            "Model loading or inference may fail."
        )

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(model_path)

    print("\nLoading model on CPU first...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    model.eval()

    print("Model loaded on CPU.")
    print("Model dtype:", next(model.parameters()).dtype)

    print("\nMoving entire model to cuda:0...")

    model = model.to(torch.device("cuda:0"))

    torch.cuda.synchronize()

    print("\n========== Actual Model Device Check ==========")

    embedding_device = (
        model.get_input_embeddings().weight.device
    )

    parameter_devices = sorted(
        set(
            str(parameter.device)
            for parameter in model.parameters()
        )
    )

    lm_head_device = model.lm_head.weight.device

    print("Input embedding device:", embedding_device)
    print("Model parameter devices:", parameter_devices)
    print("LM head device:", lm_head_device)

    if embedding_device.type != "cuda":
        raise RuntimeError(
            f"Model embedding is on {embedding_device}, not CUDA."
        )

    if parameter_devices != ["cuda:0"]:
        raise RuntimeError(
            f"Model parameters are not all on cuda:0: "
            f"{parameter_devices}"
        )

    if lm_head_device.type != "cuda":
        raise RuntimeError(
            f"LM head is not on CUDA: {lm_head_device}"
        )

    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)

    print("\n========== GPU Memory After Loading ==========")

    print(
        "GPU 0 allocated:",
        round(allocated / 1024**3, 2),
        "GB"
    )

    print(
        "GPU 0 reserved:",
        round(reserved / 1024**3, 2),
        "GB"
    )

    if allocated < 10 * 1024**3:
        raise RuntimeError(
            "GPU memory allocation is unexpectedly low. "
            "Please check whether the model is fully loaded."
        )

    print("\nGPU loading verification PASSED.")

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


def prepare_model_inputs(
    messages,
    answer,
    processor,
    model
):
    """
    Prepare multimodal prefix/full inputs.

    Prefix:
        user image + question + options + assistant generation prompt

    Full:
        Prefix + ground-truth answer

    The answer token position is determined from actual
    processor tokenization.
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
        {
            "role": "assistant",
            "content": answer
        }
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

    prefix_inputs = processor(
        text=[prefix_text],
        **common_kwargs
    )

    full_inputs = processor(
        text=[full_text],
        **common_kwargs
    )

    prefix_ids = prefix_inputs["input_ids"]
    full_ids = full_inputs["input_ids"]

    prefix_len = prefix_ids.shape[1]

    if full_ids.shape[1] <= prefix_len:
        raise RuntimeError(
            "Full input is not longer than prefix input. "
            "Cannot locate answer token."
        )

    if not torch.equal(
        prefix_ids[0],
        full_ids[0, :prefix_len]
    ):
        raise RuntimeError(
            "Prefix token IDs do not match the beginning "
            "of full input. Cannot safely determine answer position."
        )

    answer_token_ids = processor.tokenizer(
        answer,
        add_special_tokens=False
    )["input_ids"]

    if len(answer_token_ids) != 1:
        raise RuntimeError(
            f"Answer {answer!r} is not a single token: "
            f"{answer_token_ids}"
        )

    answer_position = prefix_len

    target_token_id = full_ids[
        0,
        answer_position
    ].item()

    if target_token_id != answer_token_ids[0]:
        raise RuntimeError(
            "Answer token verification failed. "
            f"Expected {answer_token_ids[0]}, "
            f"got {target_token_id}."
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
    Original full-logits Loss calculation.

    IMPORTANT:
        Do not optimize this implementation.

    The model calculates logits for the entire sequence.
    Then the logits at answer_position - 1 are selected
    to predict the ground-truth answer token.
    """

    import torch
    import torch.nn.functional as F

    with torch.inference_mode():

        # Keep the original full model forward.
        outputs = model(**model_inputs)

        # Original full-vocabulary logits at answer prediction position.
        answer_logits = outputs.logits[
            0,
            answer_position - 1,
            :
        ].float()

        target = torch.tensor(
            [target_token_id],
            dtype=torch.long,
            device=answer_logits.device
        )

        # Original cross entropy calculation.
        loss = F.cross_entropy(
            answer_logits.unsqueeze(0),
            target
        )

        loss_value = loss.item()

        del outputs
        del answer_logits
        del target
        del loss

    return loss_value


def install_dynamic_head_hook(model, layer_index):
    """
    Install one dynamic hook on the selected layer's o_proj.

    state["head"] = None:
        No ablation, normal forward.

    state["head"] = h:
        Zero the 128-dimensional segment corresponding
        to Query Head h before o_proj.
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

    state = {
        "head": None
    }

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
    num_heads
):
    """
    Calculate:
        1 baseline loss
        N single-head ablation losses
    """

    state["head"] = None

    print("\nCalculating baseline loss...")

    baseline_loss = calculate_answer_loss(
        model,
        model_inputs,
        answer_position,
        target_token_id
    )

    print(f"Baseline loss: {baseline_loss:.6f}")

    head_losses = []

    for head_index in range(num_heads):

        state["head"] = head_index

        loss_value = calculate_answer_loss(
            model,
            model_inputs,
            answer_position,
            target_token_id
        )

        head_losses.append(loss_value)

        print(
            f"\rHead ablation: {head_index + 1}/{num_heads} "
            f"| Loss: {loss_value:.6f}",
            end="",
            flush=True
        )

    print()

    state["head"] = None

    return baseline_loss, head_losses


def generate_labels(
    baseline_loss,
    head_losses,
    eps
):
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
            delta / delta_sum
            for delta in deltas
        ]

        valid = True

    return deltas, target_weights, valid


def load_processed_ids(output_path):
    """
    Read existing successful JSONL results for resume support.
    """

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
                    processed_ids.add(
                        str(record["sample_id"])
                    )

            except Exception:
                continue

    return processed_ids


def append_jsonl(output_path, record):
    """
    Append one completed sample immediately.
    """

    with open(output_path, "a", encoding="utf-8") as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

        f.flush()


def main():

    args = parse_args()

    # Configure GPU before importing torch.
    physical_gpu = configure_gpus(args.gpus)

    import torch

    print("\n========== Arguments ==========")
    print("Model:", args.model_path)
    print("Dataset:", args.data_path)
    print("Image root:", args.image_root)
    print("Layer:", args.layer)
    print("Limit:", args.limit)
    print("Physical GPU:", physical_gpu)
    print("Device: cuda:0")
    print("Output:", args.output)

    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(args.data_path)

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True
    )

    df = pd.read_csv(
        args.data_path,
        sep="\t"
    )

    required_columns = [
        "answer",
        "question",
        "A",
        "B",
        "C",
        "D",
        "image_path"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise RuntimeError(
                f"Missing dataset column: {column}"
            )

    if args.limit > 0:
        df = df.head(args.limit)

    print("\n========== Dataset ==========")
    print("Samples to scan:", len(df))

    # Load model only once.
    model, processor = load_model(
        args.model_path
    )

    # Install one dynamic hook.
    state, hook_handle, num_heads, head_dim = (
        install_dynamic_head_hook(
            model,
            args.layer
        )
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
                row["index"]
                if "index" in df.columns
                else position
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
                    Path(args.image_root) /
                    Path(image_name).name
                )

                if fallback_path.exists():

                    image_path = fallback_path

                else:

                    print(
                        f"\n[Error] Image not found: {image_path}"
                    )

                    error_count += 1
                    continue

            print("\n" + "=" * 65)

            print(
                f"Sample {sample_id} "
                f"({position + 1}/{len(df)})"
            )

            print("Question:", row["question"])
            print("Ground Truth:", answer)
            print("Image:", image_path)

            try:

                messages = build_messages(
                    row,
                    image_path
                )

                model_inputs, answer_position, target_token_id = (
                    prepare_model_inputs(
                        messages,
                        answer,
                        processor,
                        model
                    )
                )

                # Baseline + all single-head ablations.
                baseline_loss, head_losses = (
                    calculate_sample_losses(
                        model,
                        model_inputs,
                        answer_position,
                        target_token_id,
                        state,
                        num_heads
                    )
                )

                deltas, target_weights, valid = generate_labels(
                    baseline_loss,
                    head_losses,
                    args.eps
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

                append_jsonl(
                    args.output,
                    record
                )

                success_count += 1

                if not valid:
                    invalid_count += 1

                print("\n========== Sample Result ==========")
                print(f"Baseline loss: {baseline_loss:.6f}")
                print(f"Valid label: {valid}")

                print("Target weights:")
                print([
                    round(x, 6)
                    for x in target_weights
                ])

                print(
                    f"Completed: {success_count} "
                    f"| Invalid: {invalid_count} "
                    f"| Errors: {error_count} "
                    f"| Skipped: {skipped_count}"
                )

                del model_inputs
                del messages

                torch.cuda.synchronize()

            except Exception as e:

                error_count += 1

                print(
                    f"\n[Error] Sample {sample_id}: "
                    f"{type(e).__name__}: {e}"
                )

                state["head"] = None

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
        hook_handle.remove()

    print("\n" + "=" * 65)
    print("========== Finished ==========")
    print("Successful samples:", success_count)
    print("Invalid labels:", invalid_count)
    print("Errors:", error_count)
    print("Skipped existing samples:", skipped_count)
    print("Output:", args.output)


if __name__ == "__main__":
    main()

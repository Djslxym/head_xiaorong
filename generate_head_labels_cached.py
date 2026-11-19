import os
import sys
import json
import argparse
import traceback
from pathlib import Path

import pandas as pd


# ============================================================
# 1. Arguments
# ============================================================

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
        help="Maximum samples. 0 means all samples."
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="One physical GPU ID, e.g. 0 or 2"
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12
    )

    return parser.parse_args()


# ============================================================
# 2. Configure GPU
# ============================================================

def configure_gpus(gpu_string):
    """
    Set CUDA_VISIBLE_DEVICES before importing torch.

    Example:
        --gpus 2

    Physical GPU 2 becomes logical cuda:0.
    """

    gpu_string = gpu_string.strip().replace(" ", "")

    if not gpu_string:
        raise ValueError("--gpus cannot be empty.")

    gpu_ids = gpu_string.split(",")

    if not all(x.isdigit() for x in gpu_ids):
        raise ValueError(
            f"Invalid GPU IDs: {gpu_string}"
        )

    if len(gpu_ids) != 1:
        raise ValueError(
            "This version supports one GPU only. "
            "Please specify one GPU, e.g. --gpus 2."
        )

    if "torch" in sys.modules:
        raise RuntimeError(
            "PyTorch was imported before CUDA_VISIBLE_DEVICES "
            "was configured. Please start a fresh Python process."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_string

    print("\n========== GPU Configuration ==========")
    print("Selected physical GPU:", gpu_string)
    print("CUDA_VISIBLE_DEVICES:", gpu_string)


# ============================================================
# 3. Load model
# ============================================================

def load_model(model_path):
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
        raise RuntimeError(
            "CUDA is unavailable. Cannot run GPU inference."
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected one visible GPU, "
            f"but found {torch.cuda.device_count()}."
        )

    device = torch.device("cuda:0")

    print("GPU:", torch.cuda.get_device_name(0))

    free_memory, total_memory = torch.cuda.mem_get_info(0)

    print(
        f"GPU free memory: {free_memory / 1024**3:.2f} GB"
    )

    print(
        f"GPU total memory: {total_memory / 1024**3:.2f} GB"
    )

    if free_memory < 12 * 1024**3:
        print(
            "\n[Warning] GPU free memory is below 12 GB. "
            "Model loading may fail due to insufficient memory."
        )

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        model_path
    )

    print("\nLoading model on CPU first...")

    # Important:
    # Do not use device_map here.
    # The previous device_map configuration did not place
    # the model on CUDA in this environment.
    #
    # Keep torch_dtype="auto" to preserve the checkpoint dtype.

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    model.eval()

    print("\nModel loaded on CPU.")
    print(
        "Model dtype:",
        next(model.parameters()).dtype
    )

    # --------------------------------------------------------
    # Explicitly move the entire model to GPU
    # --------------------------------------------------------

    print("\nMoving entire model to cuda:0...")

    try:
        model = model.to(device)
        torch.cuda.synchronize()

    except Exception as e:
        raise RuntimeError(
            "Failed to move the model to GPU. "
            f"{type(e).__name__}: {e}"
        ) from e

    # --------------------------------------------------------
    # Verify model placement
    # --------------------------------------------------------

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

    print("Input embedding device:", embedding_device)
    print("Model parameter devices:", parameter_devices)

    if embedding_device.type != "cuda":
        raise RuntimeError(
            f"Model embedding is on {embedding_device}, "
            "not CUDA."
        )

    if any(
        parameter.device.type != "cuda"
        for parameter in model.parameters()
    ):
        raise RuntimeError(
            "Some model parameters are still on CPU."
        )

    # Check output head as well.
    print(
        "LM head device:",
        model.lm_head.weight.device
    )

    torch.cuda.synchronize()

    allocated = (
        torch.cuda.memory_allocated(0) / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved(0) / 1024**3
    )

    print("\n========== GPU Memory After Loading ==========")

    print(
        f"GPU 0 allocated: {allocated:.2f} GB"
    )

    print(
        f"GPU 0 reserved: {reserved:.2f} GB"
    )

    if allocated < 10:
        raise RuntimeError(
            f"GPU allocated memory is only {allocated:.2f} GB. "
            "This is unexpectedly low for the 7B model."
        )

    print("\nGPU loading verification PASSED.")

    return model, processor


# ============================================================
# 4. Build messages
# ============================================================

def build_messages(row, image_path):

    question = str(row["question"]).strip()

    option_lines = []

    for option in ["A", "B", "C", "D"]:

        value = row.get(option, "")

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            option_lines.append(
                f"{option}. {value}"
            )

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


# ============================================================
# 5. Prepare model inputs
# ============================================================

def prepare_model_inputs(
    messages,
    answer,
    processor,
    model
):
    """
    Prefix:
        image + question + options + assistant generation prompt

    Full:
        Prefix + ground-truth answer

    The answer token at position t is predicted by logits at t-1.
    """

    import torch

    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(
        messages
    )

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
            "Full input is not longer than prefix input."
        )

    if not torch.equal(
        prefix_ids[0],
        full_ids[0, :prefix_len]
    ):
        raise RuntimeError(
            "Prefix token IDs do not match the beginning "
            "of full input."
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

    # The entire model is on one GPU.
    input_device = (
        model.get_input_embeddings().weight.device
    )

    model_inputs = {}

    for key, value in full_inputs.items():

        if torch.is_tensor(value):
            model_inputs[key] = value.to(input_device)

        else:
            model_inputs[key] = value

    return (
        model_inputs,
        answer_position,
        target_token_id
    )


# ============================================================
# 6. Calculate answer NLL
# ============================================================

def calculate_answer_loss(
    model,
    model_inputs,
    answer_position,
    target_token_id
):
    """
    Calculate NLL of the ground-truth answer token.

    NLL = -log P(ground_truth_answer | image, question, options)

    Only calculate vocabulary logits at the answer position.
    """

    import torch
    import torch.nn.functional as F

    with torch.inference_mode():

        # Run the multimodal model.
        outputs = model.model(
            **model_inputs,
            use_cache=False,
            return_dict=True
        )

        # Hidden state immediately before the answer token.
        answer_hidden = outputs.last_hidden_state[
            0,
            answer_position - 1,
            :
        ]

        # Calculate vocabulary logits only for this position.
        answer_logits = model.lm_head(
            answer_hidden.unsqueeze(0)
        ).float()

        target = torch.tensor(
            [target_token_id],
            dtype=torch.long,
            device=answer_logits.device
        )

        loss = F.cross_entropy(
            answer_logits,
            target
        )

        loss_value = loss.item()

        del outputs
        del answer_hidden
        del answer_logits
        del target
        del loss

    return loss_value


# ============================================================
# 7. Install dynamic Head ablation hook
# ============================================================

def install_dynamic_head_hook(
    model,
    layer_index
):
    """
    Zero one Query Head's output dimensions before o_proj.
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

    hook_handle = (
        attention.o_proj.register_forward_pre_hook(
            dynamic_ablation_hook
        )
    )

    print("Dynamic Head Ablation Hook installed.")

    return (
        state,
        hook_handle,
        num_heads,
        head_dim
    )


# ============================================================
# 8. Calculate baseline and Head ablation losses
# ============================================================

def calculate_sample_losses(
    model,
    model_inputs,
    answer_position,
    target_token_id,
    state,
    num_heads
):
    """
    One baseline forward + independent Head ablation forwards.
    """

    state["head"] = None

    print("\nCalculating baseline loss...")

    baseline_loss = calculate_answer_loss(
        model,
        model_inputs,
        answer_position,
        target_token_id
    )

    print(
        f"Baseline loss: {baseline_loss:.6f}"
    )

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


# ============================================================
# 9. Generate labels
# ============================================================

def generate_labels(
    baseline_loss,
    head_losses,
    eps
):

    deltas = [
        max(0.0, loss - baseline_loss)
        for loss in head_losses
    ]

    delta_sum = sum(deltas)

    if delta_sum <= eps:

        target_weights = [
            0.0
            for _ in deltas
        ]

        valid = False

    else:

        target_weights = [
            delta / delta_sum
            for delta in deltas
        ]

        valid = True

    return (
        deltas,
        target_weights,
        valid
    )


# ============================================================
# 10. Resume support
# ============================================================

def load_processed_ids(output_path):

    processed_ids = set()

    if not os.path.exists(output_path):
        return processed_ids

    with open(
        output_path,
        "r",
        encoding="utf-8"
    ) as f:

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


def append_jsonl(
    output_path,
    record
):

    with open(
        output_path,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

        f.flush()


# ============================================================
# 11. Main
# ============================================================

def main():

    args = parse_args()

    # Must happen before importing torch.
    configure_gpus(args.gpus)

    print("\n========== Arguments ==========")

    print("Model:", args.model_path)
    print("Dataset:", args.data_path)
    print("Image root:", args.image_root)
    print("Layer:", args.layer)
    print("Limit:", args.limit)
    print("Physical GPU:", args.gpus)
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

    # Load model once.
    model, processor = load_model(
        args.model_path
    )

    state, hook_handle, num_heads, head_dim = (
        install_dynamic_head_hook(
            model,
            args.layer
        )
    )

    processed_ids = load_processed_ids(
        args.output
    )

    print(
        "\nAlready processed:",
        len(processed_ids)
    )

    success_count = 0
    invalid_count = 0
    error_count = 0
    skipped_count = 0

    try:

        for position, (_, row) in enumerate(
            df.iterrows()
        ):

            sample_id = str(
                row["index"]
                if "index" in df.columns
                else position
            )

            if sample_id in processed_ids:

                skipped_count += 1
                continue

            answer = str(
                row["answer"]
            ).strip().upper()

            if answer not in ["A", "B", "C", "D"]:

                print(
                    f"\n[Skip] Sample {sample_id}: "
                    f"Invalid answer {answer!r}"
                )

                error_count += 1
                continue

            image_name = str(
                row["image_path"]
            ).strip()

            image_path = Path(image_name)

            if not image_path.is_absolute():

                image_path = (
                    Path(args.image_root) / image_path
                )

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

                (
                    model_inputs,
                    answer_position,
                    target_token_id
                ) = prepare_model_inputs(
                    messages,
                    answer,
                    processor,
                    model
                )

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

                (
                    deltas,
                    target_weights,
                    valid
                ) = generate_labels(
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

                print(
                    f"Baseline loss: {baseline_loss:.6f}"
                )

                print("Valid label:", valid)

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

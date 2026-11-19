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

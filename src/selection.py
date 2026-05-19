def select_winner(metrics, selection_metric):
    candidates = []
    for model_name, values in metrics.items():
        candidates.append({
            "model_name": model_name,
            "primary_metric": values[selection_metric],
            "macro_precision": values["macro_precision"],
        })

    # Deterministic tie break: highest primary metric, highest precision, alphabetically simpler name.
    candidates = sorted(
        candidates,
        key=lambda x: (-x["primary_metric"], -x["macro_precision"], x["model_name"])
    )
    return candidates[0], candidates

# api/app/services/model_infer.py
from transformers import pipeline, AutoTokenizer
from ..config import settings
from typing import Optional

_CACHE: dict[str, Optional[list]] = {"pipes": None}

def get_pipes():
    """Load all cross-validation fold models for ensemble inference."""
    if _CACHE["pipes"] is None:
        pipes = []
        for fold_idx in range(1, settings.HF_NUM_FOLDS + 1):
            model_path = f"{settings.HF_MODEL_BASE_DIR}/{settings.HF_MODEL_PREFIX}_fold-{fold_idx}"
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                model_max_length=settings.MAX_TOKENS,
                truncation_side="right",
            )
            pipe = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=tokenizer,
                device=settings.HF_DEVICE,
            )
            pipes.append(pipe)
        _CACHE["pipes"] = pipes
    return _CACHE["pipes"]

def predict_batch(titles: list[str], abstracts: list[str]) -> list[dict]:
    """
    Ensemble inference across all cross-validation folds for multi-label classification.

    For multi-label models, each label is predicted independently (sigmoid activation),
    so multiple labels can be active simultaneously with their own probabilities.

    Returns a dict per text with:
      - scores: dict mapping each label to its ensemble probability (0..1)
      - all:    full list of {label, score} for all classes
    """
    pipes = get_pipes()

    # Combine titles and abstracts
    texts = [f"{title}. {abstract}".strip() for title, abstract in zip(titles, abstracts)]

    # Collect predictions from all folds
    all_fold_predictions = []
    for pipe in pipes:
        outputs = pipe(
            texts,
            truncation=True,
            max_length=settings.MAX_TOKENS,
            padding=True,
            top_k=None,  # Return scores for all labels (multi-label: each label independent)
        )
        all_fold_predictions.append(outputs)

    # Ensemble: average predictions across folds for each label
    results = []
    for text_idx in range(len(texts)):
        # Collect scores for each label across all folds
        label_scores = {}  # {label: [score_fold1, score_fold2, ...]}

        for fold_outputs in all_fold_predictions:
            # For multi-label models with top_k=None, output is a list of {label, score} dicts
            # Each label has an independent probability (0-1) from sigmoid activation
            output = fold_outputs[text_idx]
            if isinstance(output, list):
                for item in output:
                    label = item["label"]
                    score = float(item["score"])
                    if label not in label_scores:
                        label_scores[label] = []
                    label_scores[label].append(score)
            else:
                # Fallback for single prediction (shouldn't happen with top_k=None)
                label = output.get("label", "LABEL_0")
                score = float(output.get("score", 0.0))
                if label not in label_scores:
                    label_scores[label] = []
                label_scores[label].append(score)

        # Average scores across folds for each label
        ensemble_scores = {
            label: sum(scores) / len(scores)
            for label, scores in label_scores.items()
        }

        # Build the output format
        all_labels = [
            {"label": label, "score": score}
            for label, score in sorted(ensemble_scores.items())
        ]

        results.append({
            "scores": ensemble_scores,
            "all": all_labels
        })

    return results
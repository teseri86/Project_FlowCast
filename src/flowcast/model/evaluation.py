"""
One way to grade every model, so results can be compared fairly.

Models give different kinds of answers:
  - score models (baseline, regressors)  -> a 0-100 score
  - 7-level classifier                   -> one of the 7 levels
  - 3-level classifier                   -> drought / normal / high_water

Every answer can be turned into the 3 groups (drought / normal / high_water), so
the main measures use those 3 groups. Measures that only some models can have
(like the average miss in score points) are left empty for the others.

THE MEASURES (plain language)
  drought_caught_%        of the real drought days, how many the model called drought   (recall)
  drought_right_%         when the model said drought, how often it was right           (precision)
  high_water_caught_%     same, for high water (score 90+)
  high_water_right_%
  normal_caught_%         of the real normal days, how many it called normal
  balanced_accuracy_%     average of the three "caught" numbers - fair even though most days
                          are normal (a model that always says "normal" scores only 33%)
  accuracy_%              share of all days in the right group (flattered by the many normal days)
  level7_exact_%          share of days in exactly the right one of the 7 levels (not for 3-level models)
  avg_miss_points         average distance from the true 0-100 score (score models only)
"""

import numpy as np
import pandas as pd

from src.flowcast.model.build_training_table import LEVELS_3, LEVELS_7, to_level

LEVEL7_TO_3 = {
    "severe_drought": "drought", "moderate_drought": "drought", "mild_drought": "drought",
    "normal": "normal",
    "above_normal_flow": "high_water", "high_flow": "high_water", "very_high_flow": "high_water",
}


def _pct(numerator_mask, denominator_mask):
    total = denominator_mask.sum()
    return float(numerator_mask.sum() / total * 100) if total else np.nan


def grade(true_pct, predicted, kind):
    """Grade one model.

    true_pct:  the real 0-100 scores (target_pct)
    predicted: the model's answers
    kind:      "score" (0-100 numbers), "level7" (7-level labels) or "level3" (3-level labels)
    """
    true_pct = np.asarray(true_pct, dtype=float)
    true3 = np.asarray(to_level(pd.Series(true_pct), LEVELS_3), dtype=object)
    true7 = np.asarray(to_level(pd.Series(true_pct), LEVELS_7), dtype=object)

    if kind == "score":
        pred_pct = np.clip(np.asarray(predicted, dtype=float), 0, 100)
        pred3 = np.asarray(to_level(pd.Series(pred_pct), LEVELS_3), dtype=object)
        pred7 = np.asarray(to_level(pd.Series(pred_pct), LEVELS_7), dtype=object)
    elif kind == "level7":
        pred7 = np.asarray(predicted, dtype=object)
        pred3 = np.array([LEVEL7_TO_3.get(x, x) for x in pred7], dtype=object)
        pred_pct = None
    elif kind == "level3":
        pred3 = np.asarray(predicted, dtype=object)
        pred7, pred_pct = None, None
    else:
        raise ValueError(f"unknown kind: {kind}")

    result = {}
    caught = []
    for group in ["drought", "high_water", "normal"]:
        is_true, is_pred = true3 == group, pred3 == group
        result[f"{group}_caught_%"] = _pct(is_true & is_pred, is_true)
        if group != "normal":
            result[f"{group}_right_%"] = _pct(is_true & is_pred, is_pred)
        caught.append(result[f"{group}_caught_%"])
    result["balanced_accuracy_%"] = float(np.nanmean(caught))
    result["accuracy_%"] = float(np.mean(true3 == pred3) * 100)
    result["level7_exact_%"] = float(np.mean(true7 == pred7) * 100) if pred7 is not None else np.nan
    result["avg_miss_points"] = float(np.mean(np.abs(pred_pct - true_pct))) if pred_pct is not None else np.nan
    return result


MEASURE_ORDER = ["drought_caught_%", "drought_right_%", "high_water_caught_%", "high_water_right_%",
                 "normal_caught_%", "balanced_accuracy_%", "accuracy_%", "level7_exact_%", "avg_miss_points"]
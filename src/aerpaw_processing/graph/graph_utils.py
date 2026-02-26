from collections import defaultdict
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from aerpaw_processing.preprocessing.preprocess_utils import (
    get_index_col,
    get_label_col,
    get_column_order,
)
from aerpaw_processing.resources.config.config_init import load_env


load_env()


logger = logging.getLogger(__name__)


def get_mutual_correlation_scores(df: pd.DataFrame, label: str) -> dict[str, float]:
    working_data = df.copy()

    label_col = get_label_col(working_data, label)

    x = working_data.drop(columns=[label_col])
    x = x.select_dtypes(include=["number"])
    y = working_data[label_col]

    scores_dict: dict[str, float] = {}

    for col in x.columns:
        valid_mask = x[col].notna() & y.notna()
        x_valid = x.loc[valid_mask, col]
        y_valid = y.loc[valid_mask]

        if len(x_valid) > 1:
            score = mutual_info_regression(
                x_valid.to_frame(), y_valid, random_state=42
            )[0]
            scores_dict[col] = score

    if not scores_dict:
        error = "Mutual information regression failed: No valid overlapping data points found for analysis."
        logger.error(error)
        raise ValueError(error)

    return scores_dict


def graph_mutual_correlation(
    df: pd.DataFrame,
    label: str,
    project_cords: bool = False,
    save_path: str | None = None,
):
    scores_dict = get_mutual_correlation_scores(df, label)

    col_order = get_column_order(project_cords=project_cords)
    col_order.reverse()
    found_cols: list[str] = []
    for col in col_order:
        for score_key in scores_dict.keys():
            if col in score_key and score_key != get_index_col():
                found_cols.append(score_key)
    scores_dict = {col: scores_dict[col] for col in found_cols}

    scores_series = pd.Series(scores_dict)

    plt.figure(figsize=(8, max(4, len(scores_series) * 0.3)))
    scores_series.plot(kind="barh", color="steelblue", edgecolor="black")

    plt.title(f"Mutual Correlation with {get_label_col(df, label)}")
    plt.xlabel("Score")
    plt.ylabel("Features")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return scores_series


def graph_average_mutual_correlation(
    dfs: list[pd.DataFrame],
    label: str,
    project_cords: bool = False,
    save_path: str | None = None,
) -> pd.Series:
    """
    Graphs the average mutual information correlation scores for features
    across a list of DataFrames.
    """
    column_scores_tracker: dict[str, list[float]] = defaultdict(list)

    for i, df in enumerate(dfs):
        try:
            scores_dict = get_mutual_correlation_scores(df, label)
            for col, score in scores_dict.items():
                column_scores_tracker[col].append(score)
        except ValueError as e:
            logger.warning(f"Skipping DataFrame at index {i} due to error: {e}")
            continue

    if not column_scores_tracker:
        error = "Failed to calculate mutual correlation across any of the provided DataFrames."
        logger.error(error)
        raise ValueError(error)

    avg_scores_dict = {
        col: sum(scores) / len(scores) for col, scores in column_scores_tracker.items()
    }

    col_order = get_column_order(project_cords=project_cords)
    col_order.reverse()

    found_cols: list[str] = []
    for col in col_order:
        for score_key in avg_scores_dict.keys():
            if col in score_key and score_key != get_index_col():
                found_cols.append(score_key)

    ordered_avg_scores = {col: avg_scores_dict[col] for col in found_cols}
    scores_series = pd.Series(ordered_avg_scores)

    plt.figure(figsize=(8, max(4, len(scores_series) * 0.3)))
    scores_series.plot(kind="barh", color="steelblue", edgecolor="black")

    title_label = get_label_col(dfs[0], label) if dfs else label

    plt.title(f"Average Mutual Correlation with {title_label}")
    plt.xlabel("Average Score")
    plt.ylabel("Features")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return scores_series

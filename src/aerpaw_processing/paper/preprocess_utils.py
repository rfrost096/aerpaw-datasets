
import pandas as pd
import os
import logging
from enum import Enum
from typing import cast

from aerpaw_processing.resources.config.config_init import load_env, CONFIG

load_env()

logger = logging.getLogger(__name__)

TECH_LIST = ["NR_5G", "LTE_4G"]

class StepEnum(Enum):
    """All configured steps used for inter-step referencing"""
    READ_DATA = "read_data"
    RENAME_COLUMNS = "rename_columns"
    REMOVE_COLUMNS = "remove_columns"
    COMBINE_TECH_FILES = "combine_tech_files"
    COMBINE_FLIGHT_TECHS = "combine_flight_techs"
    INTERPOLATE_TO_LABEL = "interpolate_to_label"
    PROJECT_COORDINATES = "project_coordinates"
    CALCULATE_BIN = "calculate_bin"
    CORRELATION_COMPUTATION = "correlation_computation"
    FAST_FADING_CORRELATION = "fast_fading_correlation"

def add_step_entry(step: StepEnum, step_data: pd.DataFrame, context: pd.DataFrame):
    """Add a processing step entry to working context.

    Take the current working context and append a new entry with a step field
    and the data generated in the step. This will be used to keep track of changes
    made between processing steps.

    Returns:
        pd.DataFrame: context with added step data
    """
    context.loc[len(context)] = [step.value, step_data]

    return context


def get_step_entry(step: StepEnum, context: pd.DataFrame) -> pd.DataFrame:
    """Get step data from step entry. Check to make sure the step exists.

    Returns:
        pd.DataFrame: previous step data
    """
    step_df: pd.DataFrame = cast(pd.DataFrame, context[context["step"] == step.value])

    if not step_df.empty:
        return step_df["step_data"].iloc[0].copy()
    else:
        raise ValueError("Step not in context")


def get_col_tech_name(col: str, tech: str):
    return f"{col}_{tech}"

def get_label_col(df_iter, label: str = "RSRP"):
    candidates: list[str] = [label]
    candidates.extend([get_col_tech_name(label, tech) for tech in TECH_LIST])
    for potential_col in candidates:
        if potential_col in df_iter.columns:
            return potential_col 
    return None


def get_tech_independent_cols():
    """Get columns that are independent of flight technology."""
    tech_independent_cols: list[str] = []
    for cat in CONFIG.categories:
        if cat.category in ("Timestamp", "Location", "Base Station"):
            tech_independent_cols.extend([col.name for col in cat.cols])
    
    return tech_independent_cols


def get_alias_map():
    """Get a map that connects column alias names to the same processing name.

    The CONFIG file has a list of column categories, which each have a list of
    columns. The columns have an alias_list attribute that list all possible alias's
    that mean that specific column. The columns have a name attribute for the alias
    to be renamed to.

    The map is to be used to rename all columns across all datasets to a standardized
    naming scheme so they are easily comparable and can be operated on by the same
    functions.

    Returns:
        dict[str, str]: A map where each key = possible alias for value = standardized
        name
    """
    alias_map: dict[str, str] = {}
    for category in CONFIG.categories:
        for col in category.cols:
            for alias in col.alias_list:
                alias_map[alias] = col.name
            alias_map[col.name] = col.name

    return alias_map



def get_col_categorical_map():
    """Get a map from column name to boolean true/false that is true if column is a categorical
    value and false if it is a continuous value."""
    col_categorical_map: dict[str, bool] = {}
    tech_ind_cols = get_tech_independent_cols()
    for cat in CONFIG.categories:
        for col in cat.cols:
            if col.name not in tech_ind_cols:
                for tech in TECH_LIST:
                    col_categorical_map[get_col_tech_name(col.name, tech)] = col.categorical
            else:
                col_categorical_map[col.name] = col.categorical
    return col_categorical_map


def get_env_var(env_var_name: str):
    var_val = os.getenv(env_var_name)
    if var_val is None:
        error = f"Environment variable {env_var_name} not set"
        logger.error(error)
        raise EnvironmentError(error)

    return var_val

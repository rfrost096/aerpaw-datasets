import logging
from aerpaw_processing.preprocessing.preprocess_main import process_datasets
from aerpaw_processing.preprocessing.preprocess_utils import (
    get_all_flight_ids,
    get_dataset_and_flight_from_id,
    get_label_col,
)
import aerpaw_processing.graph.graph_utils as graph_utils
from aerpaw_processing.resources.config.config_init import load_env


load_env()
logger = logging.getLogger(__name__)


def graph_mutual():
    flight_id = get_all_flight_ids()[0]  # Example: Get the first flight ID
    label = "RSRP"  # Example: Set label to "RSRP"
    project_cords = True  # Example: Set project_cords to True
    save_path = None  # Example: Set save_path to None

    if flight_id not in get_all_flight_ids():
        raise ValueError(
            f"Invalid flight ID: {flight_id}. Available IDs: {get_all_flight_ids()}"
        )

    data_dict = process_datasets(
        save_cleaned_data=False, relative_time=True, project_coords=project_cords
    )
    dataset_num, flight_name = get_dataset_and_flight_from_id(flight_id)
    df = data_dict[dataset_num][flight_name]

    label_col = get_label_col(df, label)

    graph_utils.graph_mutual_correlation(
        df,
        label_col=label_col,
        project_cords=project_cords,
        save_path=save_path,
    )


def graph_avg_mutual():
    project_cords = True  # Example: Set project_cords to True
    save_path = None  # Example: Set save_path to None

    data_dict = process_datasets(
        save_cleaned_data=False, relative_time=True, project_coords=project_cords
    )
    df_list = [df for dataset in data_dict.values() for df in dataset.values()]

    logger.info(
        f"Average mutual correlation only works with RSRP. Graphing correlation with RSRP."
    )

    graph_utils.graph_average_mutual_correlation(
        df_list,
        label="RSRP",
        project_cords=project_cords,
        save_path=save_path,
    )


def graph_label():
    flight_ids = get_all_flight_ids()[6:8]  # Example: Get the first 3 flight IDs
    label = "PCI"  # Example: Set label to "RSRP"
    project_cords = True  # Example: Set project_cords to True
    save_path = None  # Example: Set save_path to None
    alt_median_deviation = False  # Example: Set alt_median_deviation to True
    relative_time = True  # Example: Set relative_time to False
    graph_towers = True  # Example: Set graph_towers to False
    filter_features_bool = False  # Example: Set filter_features_bool to True
    categorical = True  # Example: Set categorical to False

    data_dict = process_datasets(
        save_cleaned_data=False,
        filter_features_bool=filter_features_bool,
        relative_time=relative_time,
        project_coords=project_cords,
        alt_median_abs_deviation=alt_median_deviation,
    )

    df, label_col = graph_utils.combine_dfs_graph(data_dict, flight_ids, label)

    graph_utils.graph_label(
        df,
        label_col=label_col,
        filter_features=filter_features_bool,
        categorical=categorical,
        project_cords=project_cords,
        graph_towers=graph_towers,
        relative_time=relative_time,
        save_path=save_path,
    )


def graph_label_temporal():
    flight_ids = get_all_flight_ids()[:2]  # Example: Get the first 3 flight IDs
    label = "RSRP"  # Example: Set label to "RSRP"
    project_cords = True  # Example: Set project_cords to True
    save_path = None  # Example: Set save_path to None
    alt_median_deviation = True  # Example: Set alt_median_deviation to True
    relative_time = True  # Example: Set relative_time to False
    graph_towers = True  # Example: Set graph_towers to False

    data_dict = process_datasets(
        save_cleaned_data=False,
        relative_time=relative_time,
        project_coords=project_cords,
        alt_median_abs_deviation=alt_median_deviation,
    )

    df, label_col = graph_utils.combine_dfs_graph(data_dict, flight_ids, label)

    graph_utils.graph_label_temporal(
        df,
        label_col=label_col,
        project_cords=project_cords,
        alt_median_abs_deviation=alt_median_deviation,
        relative_time=relative_time,
        save_path=save_path,
        graph_towers=graph_towers,
    )


graph_label()

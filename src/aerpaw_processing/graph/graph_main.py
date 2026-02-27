import argparse
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


def graph_mutual_helper(
    flight_id: str,
    label: str,
    project_cords: bool,
    save_path: str | None,
):
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


def graph_mutual():
    all_flight_ids = get_all_flight_ids()

    parser = argparse.ArgumentParser(
        description="Graph mutual correlation for a single flight.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--flight-id",
        type=str,
        default=all_flight_ids[0],
        help=(
            "Flight ID to graph.\n"
            f"Valid values: {', '.join(all_flight_ids)}\n"
            f"(default: {all_flight_ids[0]})"
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default="RSRP",
        help="Label column to use for mutual correlation (default: RSRP).",
    )
    parser.add_argument(
        "--no-project-coords",
        action="store_true",
        default=False,
        help="Disable coordinate projection (default: projection enabled).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the graph. If not provided, the graph is displayed instead.",
    )

    args = parser.parse_args()

    if args.flight_id not in all_flight_ids:
        parser.error(
            f"Invalid flight ID: {args.flight_id}. "
            f"Valid values: {', '.join(all_flight_ids)}"
        )

    graph_mutual_helper(
        flight_id=args.flight_id,
        label=args.label,
        project_cords=not args.no_project_coords,
        save_path=args.save_path,
    )


def graph_avg_mutual_helper(
    project_cords: bool,
    label: str = "RSRP",
    save_path: str | None = None,
):
    data_dict = process_datasets(
        save_cleaned_data=False, relative_time=True, project_coords=project_cords
    )
    df_list = [df for dataset in data_dict.values() for df in dataset.values()]

    if "rsrp" not in label.lower():
        error = "Average mutual correlation is only implemented for RSRP. Please specify a label containing 'RSRP' or graph mutual correlation for individual flights instead."
        logger.error(error)
        raise ValueError(error)

    graph_utils.graph_average_mutual_correlation(
        df_list,
        label=label,
        project_cords=project_cords,
        save_path=save_path,
    )


def graph_avg_mutual():
    parser = argparse.ArgumentParser(
        description="Graph average mutual correlation across all flights (RSRP only).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--no-project-coords",
        action="store_true",
        default=False,
        help="Disable coordinate projection (default: projection enabled).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the graph. If not provided, the graph is displayed instead.",
    )

    args = parser.parse_args()

    graph_avg_mutual_helper(
        project_cords=not args.no_project_coords,
        save_path=args.save_path,
    )


def graph_label_helper(
    flight_ids: list[str],
    label: str,
    project_cords: bool,
    filter_features: bool,
    categorical: bool,
    relative_time: bool,
    graph_towers: bool,
    alt_median_abs_deviation: bool,
    save_path: str | None,
):
    data_dict = process_datasets(
        save_cleaned_data=False,
        filter_features_bool=filter_features,
        relative_time=relative_time,
        project_coords=project_cords,
        alt_median_abs_deviation=alt_median_abs_deviation,
    )

    df, label_col = graph_utils.combine_dfs_graph(data_dict, flight_ids, label)

    graph_utils.graph_label(
        df,
        label_col=label_col,
        filter_features=filter_features,
        categorical=categorical,
        project_cords=project_cords,
        graph_towers=graph_towers,
        relative_time=relative_time,
        save_path=save_path,
    )


def graph_label():
    all_flight_ids = get_all_flight_ids()

    parser = argparse.ArgumentParser(
        description="Graph a label across one or more flights.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--flight-ids",
        type=str,
        default=",".join(all_flight_ids[6:8]),
        help=(
            "Comma-separated list of flight IDs to include.\n"
            f"Valid values: {', '.join(all_flight_ids)}\n"
            f"(default: {','.join(all_flight_ids[6:8])})"
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default="PCI",
        help="Label column to graph (default: PCI).",
    )
    parser.add_argument(
        "--no-project-coords",
        action="store_true",
        default=False,
        help="Disable coordinate projection (default: projection enabled).",
    )
    parser.add_argument(
        "--alt-median-deviation",
        action="store_true",
        default=False,
        help="Use median absolute deviation for altitude filtering (default: False).",
    )
    parser.add_argument(
        "--no-relative-time",
        action="store_true",
        default=False,
        help="Use absolute timestamps instead of relative time (default: relative time enabled).",
    )
    parser.add_argument(
        "--no-graph-towers",
        action="store_true",
        default=False,
        help="Disable tower overlay on the graph (default: towers enabled).",
    )
    parser.add_argument(
        "--filter-features",
        action="store_true",
        default=False,
        help="Enable feature filtering (default: False).",
    )
    parser.add_argument(
        "--no-categorical",
        action="store_true",
        default=False,
        help="Treat the label as continuous instead of categorical (default: categorical enabled).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the graph. If not provided, the graph is displayed instead.",
    )

    args = parser.parse_args()

    flight_ids = [fid.strip() for fid in args.flight_ids.split(",")]
    invalid = [fid for fid in flight_ids if fid not in all_flight_ids]
    if invalid:
        parser.error(
            f"Invalid flight ID(s): {', '.join(invalid)}. "
            f"Valid values: {', '.join(all_flight_ids)}"
        )

    graph_label_helper(
        flight_ids=flight_ids,
        label=args.label,
        project_cords=not args.no_project_coords,
        filter_features=args.filter_features,
        categorical=not args.no_categorical,
        relative_time=not args.no_relative_time,
        graph_towers=not args.no_graph_towers,
        alt_median_abs_deviation=args.alt_median_deviation,
        save_path=args.save_path,
    )


def graph_label_temporal():
    all_flight_ids = get_all_flight_ids()

    parser = argparse.ArgumentParser(
        description="Graph a label over time across one or more flights.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--flight-ids",
        type=str,
        default=",".join(all_flight_ids[:2]),
        help=(
            "Comma-separated list of flight IDs to include.\n"
            f"Valid values: {', '.join(all_flight_ids)}\n"
            f"(default: {','.join(all_flight_ids[:2])})"
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default="RSRP",
        help="Label column to graph (default: RSRP).",
    )
    parser.add_argument(
        "--no-project-coords",
        action="store_true",
        default=False,
        help="Disable coordinate projection (default: projection enabled).",
    )
    parser.add_argument(
        "--no-alt-median-deviation",
        action="store_true",
        default=False,
        help="Disable median absolute deviation for altitude filtering (default: enabled).",
    )
    parser.add_argument(
        "--no-relative-time",
        action="store_true",
        default=False,
        help="Use absolute timestamps instead of relative time (default: relative time enabled).",
    )
    parser.add_argument(
        "--no-graph-towers",
        action="store_true",
        default=False,
        help="Disable tower overlay on the graph (default: towers enabled).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the graph. If not provided, the graph is displayed instead.",
    )

    args = parser.parse_args()

    flight_ids = [fid.strip() for fid in args.flight_ids.split(",")]
    invalid = [fid for fid in flight_ids if fid not in all_flight_ids]
    if invalid:
        parser.error(
            f"Invalid flight ID(s): {', '.join(invalid)}. "
            f"Valid values: {', '.join(all_flight_ids)}"
        )

    project_cords = not args.no_project_coords
    relative_time = not args.no_relative_time
    graph_towers = not args.no_graph_towers
    alt_median_deviation = not args.no_alt_median_deviation

    graph_label_temporal_helper(
        flight_ids=flight_ids,
        label=args.label,
        project_cords=project_cords,
        alt_median_abs_deviation=alt_median_deviation,
        relative_time=relative_time,
        graph_towers=graph_towers,
        save_path=args.save_path,
    )


def graph_label_temporal_helper(
    flight_ids: list[str],
    label: str,
    project_cords: bool,
    alt_median_abs_deviation: bool,
    relative_time: bool,
    graph_towers: bool,
    save_path: str | None,
):
    data_dict = process_datasets(
        save_cleaned_data=False,
        relative_time=relative_time,
        project_coords=project_cords,
        alt_median_abs_deviation=alt_median_abs_deviation,
    )

    df, label_col = graph_utils.combine_dfs_graph(data_dict, flight_ids, label)

    graph_utils.graph_label_temporal(
        df,
        label_col=label_col,
        project_cords=project_cords,
        alt_median_abs_deviation=alt_median_abs_deviation,
        relative_time=relative_time,
        save_path=save_path,
        graph_towers=graph_towers,
    )


def graph_spatial_rsrp_correlation_helper(
    flight_id: str,
    label: str,
    save_path: str | None,
):
    data_dict = process_datasets(
        save_cleaned_data=False,
        relative_time=True,
        project_coords=False,
        alt_median_abs_deviation=True,
    )

    dataset_num, flight_name = get_dataset_and_flight_from_id(flight_id)
    df = data_dict[dataset_num][flight_name]

    label_col = get_label_col(df, label)

    graph_utils.graph_spatial_rsrp_correlation(df, label_col, save_path=save_path)


def graph_spatial_rsrp_correlation():

    all_flight_ids = get_all_flight_ids()

    parser = argparse.ArgumentParser(
        description="Graph spatial correlation of RSRP for a single flight.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--flight-id",
        type=str,
        default=",".join(all_flight_ids[:2]),
        help=(
            f"Valid values: {', '.join(all_flight_ids)}\n"
            f"(default: {','.join(all_flight_ids[:2])})"
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default="RSRP",
        help="Label column to graph (default: RSRP).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the graph. If not provided, the graph is displayed instead.",
    )

    args = parser.parse_args()

    flight_id: str = args.flight_id.strip()
    if flight_id not in all_flight_ids:
        parser.error(
            f"Invalid flight ID: {flight_id}. "
            f"Valid values: {', '.join(all_flight_ids)}"
        )

    graph_spatial_rsrp_correlation_helper(
        flight_id=flight_id,
        label=args.label,
        save_path=args.save_path,
    )


def graph_fast_fading_correlation_helper(
    flight_id: str,
    label: str,
    save_path: str | None,
):
    data_dict = process_datasets(
        save_cleaned_data=False,
        relative_time=True,
        project_coords=False,
        alt_median_abs_deviation=False,
    )

    dataset_num, flight_name = get_dataset_and_flight_from_id(flight_id)
    df = data_dict[dataset_num][flight_name]

    label_col = get_label_col(df, label)

    graph_utils.graph_fast_fading_correlation(df, label_col, save_path=save_path)


def graph_fast_fading_correlation():

    all_flight_ids = get_all_flight_ids()

    parser = argparse.ArgumentParser(
        description="Graph fast fading correlation of RSRP for a single flight.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--flight-id",
        type=str,
        default=",".join(all_flight_ids[:2]),
        help=(
            f"Valid values: {', '.join(all_flight_ids)}\n"
            f"(default: {','.join(all_flight_ids[:2])})"
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default="RSRP",
        help="Label column to graph (default: RSRP).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the graph. If not provided, the graph is displayed instead.",
    )

    args = parser.parse_args()

    flight_id: str = args.flight_id.strip()
    if flight_id not in all_flight_ids:
        parser.error(
            f"Invalid flight ID: {flight_id}. "
            f"Valid values: {', '.join(all_flight_ids)}"
        )

    graph_fast_fading_correlation_helper(
        flight_id=flight_id,
        label=args.label,
        save_path=args.save_path,
    )

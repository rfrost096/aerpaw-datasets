import logging
import pandas as pd
import os
import argparse
from aerpaw_processing.resources.step_tracker import StepTracker
from aerpaw_processing.preprocessing.preprocess_utils import (
    load_datasets,
    convert_columns,
    merge_datasets,
    rename_tech_columns,
    format_timestamp,
    filter_features,
    convert_to_relative_time,
    project_coordinates,
    get_median_abs_deviation,
    get_flight_id,
    get_index_col,
)
from aerpaw_processing.resources.config.config_init import CONFIG, load_env

load_env()

logger = logging.getLogger(__name__)
step = StepTracker()


def process_datasets(
    filter_features_bool: bool = True,
    relative_time: bool = True,
    project_coords: bool = True,
    alt_median_abs_deviation: bool = False,
    fill: bool = True,
    save_cleaned_data: bool = True,
) -> dict[int, dict[str, pd.DataFrame]]:
    """
    Processes, cleans, and merges flight datasets based on configuration settings.

    Iterates through datasets and flights defined in 'config_file.yaml', loading and
    merging sensor/technology files. Applies formatting, downselects features, and
    executes optional transformations such as relative time conversion, coordinate
    projection, outlier filtering, and missing value imputation.

    Args:
        filter_features_bool: If True, downselects to configured feature columns.
        relative_time: If True, converts absolute timestamps to relative time.
        project_coords: If True, projects lon/lat into planar (x, y) coordinates
            using the base tower location as the origin.
        alt_median_abs_deviation: If True, filters rows using median absolute
            deviation on the altitude column.
        fill: If True, fills missing values via forward-then-backward fill.
        save_cleaned_data: If True, saves processed DataFrames as CSVs to the
            directory specified by the DATASET_CLEAN_HOME env variable.

    Returns:
        Nested dict ``{dataset_num: {flight_name: DataFrame}}``.
    """
    step.next_step()
    data_dict: dict[int, dict[str, pd.DataFrame]] = {}

    for dataset in CONFIG.datasets:
        logger.debug(
            f"{step.enter_level()} Load dataset files for dataset: {dataset.name}"
        )

        for flight in dataset.flights:
            logger.debug(f"{step.enter_level()} Load files for flight: {flight.name}")

            flight_tech_data_dict: dict[str, pd.DataFrame] = {}

            for tech in flight.tech_list:
                logger.debug(
                    f"{step.enter_level()} Load tech files for tech: {tech.name}"
                )

                tech_data_list = load_datasets(dataset.num, tech.files)

                logger.debug(
                    f"{step.next_step()} Convert column names for tech: {tech.name}"
                )
                tech_data_list = convert_columns(tech_data_list, flight.merge_col)

                logger.debug(
                    f"{step.next_step()} Merge flight files for tech: {tech.name}"
                )

                if len(tech_data_list) > 1:
                    if flight.merge_col is None:
                        msg = (
                            f"Merge column not specified for flight {flight.name}, "
                            f"tech {tech.name} with multiple files."
                        )
                        logger.error(msg)
                        raise ValueError(msg)
                    tech_data = merge_datasets(tech_data_list, flight.merge_col)
                    logger.debug(f"{step.info()} Merged files for tech: {tech.name}")
                else:
                    tech_data = tech_data_list[0]
                    logger.debug(
                        f"{step.info()} Single file loaded for tech: {tech.name}"
                    )

                flight_tech_data_dict[tech.name] = tech_data
                step.exit_level()

            common_cols: set[str] = set.intersection(  # type: ignore
                *[set(d.columns) for d in flight_tech_data_dict.values()]
            )

            flight_tech_data_list: list[pd.DataFrame] = []
            for tech_name, tech_data in flight_tech_data_dict.items():
                logger.debug(f"{step.enter_level()} Rename tech columns: {flight.name}")
                flight_tech_data_list.append(
                    rename_tech_columns(
                        tech_data, tech_name, flight.merge_col, common_cols
                    )
                )
                step.exit_level()

            logger.debug(
                f"{step.continue_step()} Merge tech data for flight: {flight.name}"
            )

            if len(flight_tech_data_list) > 1:
                if flight.merge_col is None:
                    msg = (
                        f"Merge column not specified for flight {flight.name} "
                        "with multiple techs."
                    )
                    logger.error(msg)
                    raise ValueError(msg)
                flight_data = merge_datasets(flight_tech_data_list, flight.merge_col)
                logger.debug(
                    f"{step.info()} Merged tech data for flight: {flight.name}"
                )
            else:
                flight_data = flight_tech_data_list[0]
                logger.debug(
                    f"{step.info()} Single tech data for flight: {flight.name}"
                )

            logger.debug(
                f"{step.next_step()} Format timestamp for flight: {flight.name}"
            )
            flight_data = format_timestamp(flight_data)

            if filter_features_bool:
                logger.debug(
                    f"{step.next_step()} Downselect features for flight: {flight.name}"
                )
                flight_data = filter_features(flight_data)
            else:
                logger.debug(
                    f"{step.next_step()} Skipping feature downselection for flight: {flight.name}"
                )

            if relative_time:
                logger.debug(
                    f"{step.next_step()} Convert to relative time for flight: {flight.name}"
                )
                flight_data = convert_to_relative_time(flight_data)
            else:
                logger.debug(
                    f"{step.next_step()} Keeping absolute timestamp for flight: {flight.name}"
                )

            if project_coords:
                logger.debug(
                    f"{step.next_step()} Projecting coordinates for flight: {flight.name}"
                )
                flight_data = project_coordinates(flight_data)
            else:
                logger.debug(
                    f"{step.next_step()} Keeping lon/lat for flight: {flight.name}"
                )

            if alt_median_abs_deviation:
                logger.debug(
                    f"{step.next_step()} Filtering by altitude MAD for flight: {flight.name}"
                )
                flight_data = get_median_abs_deviation(
                    flight_data, project_cords=project_coords
                )
            else:
                logger.debug(
                    f"{step.next_step()} Skipping altitude MAD filter for flight: {flight.name}"
                )

            logger.debug(
                f"{step.next_step()} Enumerating rows for flight: {flight.name}"
            )
            flight_data.insert(0, get_index_col(), range(len(flight_data)))

            if fill:
                flight_data = flight_data.ffill().bfill()
                logger.debug(
                    f"{step.next_step()} Filled missing data for flight: {flight.name}"
                )
            else:
                logger.debug(
                    f"{step.next_step()} Skipping fill for flight: {flight.name}"
                )

            if save_cleaned_data:
                logger.debug(
                    f"{step.next_step()} Saving cleaned data for flight: {flight.name}"
                )
                output_dir = os.getenv("DATASET_CLEAN_HOME")
                if output_dir is None:
                    msg = "Environment variable 'DATASET_CLEAN_HOME' is not set."
                    logger.error(msg)
                    raise EnvironmentError(msg)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, get_flight_id(dataset.num, flight.name) + ".csv"
                )
                flight_data.to_csv(output_path, index=False)
            else:
                logger.debug(
                    f"{step.next_step()} Skipping save for flight: {flight.name}"
                )

            data_dict.setdefault(dataset.num, {})[flight.name] = flight_data
            step.exit_level()

        step.exit_level()

    return data_dict


def main():
    parser = argparse.ArgumentParser(
        description="Process datasets with options for relative time and coordinate projection."
    )
    parser.add_argument(
        "--no-save-data",
        dest="save_data",
        action="store_false",
        default=True,
        help="Do not save the cleaned data to CSV files.",
    )
    parser.add_argument(
        "--no-relative-time",
        dest="relative_time",
        action="store_false",
        default=True,
        help="Do not convert timestamps to relative time.",
    )
    parser.add_argument(
        "--no-project-coords",
        dest="project_coords",
        action="store_false",
        default=True,
        help="Do not project coordinates to a planar coordinate system.",
    )
    parser.add_argument(
        "--alt-median-abs-deviation",
        action="store_true",
        help="Filter data based on median absolute deviation for altitude.",
    )
    parser.add_argument(
        "--no-fill",
        dest="fill",
        action="store_false",
        default=True,
        help="Do not fill missing data using forward and backward fill.",
    )

    args = parser.parse_args()
    process_datasets(
        save_cleaned_data=args.save_data,
        relative_time=args.relative_time,
        project_coords=args.project_coords,
        alt_median_abs_deviation=args.alt_median_abs_deviation,
        fill=args.fill,
    )


if __name__ == "__main__":
    main()

import logging
import pandas as pd
import os
from aerpaw_processing.resources.step_tracker import StepTracker
from aerpaw_processing.preprocessing.utils import (
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
)
from aerpaw_processing.resources.config.config_init import CONFIG, load_env

load_env()


logger = logging.getLogger(__name__)
step = StepTracker()


def process_datasets(
    save_cleaned_data: bool = True,
    relative_time: bool = False,
    project_coords: bool = False,
    alt_median_abs_deviation: bool = False,
    fill: bool = True,
):

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

                tech_data: pd.DataFrame

                if len(tech_data_list) > 1:
                    if flight.merge_col is None:
                        error = f"Merge column not specified for flight {flight.name}, tech {tech.name} with multiple files."
                        logger.error(error)
                        raise ValueError(error)
                    tech_data = merge_datasets(tech_data_list, flight.merge_col)
                    logger.debug(f"{step.info()}Merged files for tech: {tech.name}")
                else:
                    tech_data = tech_data_list[0]
                    logger.debug(
                        f"{step.info()}Single file loaded for tech: {tech.name}"
                    )

                flight_tech_data_dict[tech.name] = tech_data

                step.exit_level()

            flight_tech_data_list: list[pd.DataFrame] = []

            common_cols: set[str] = set.intersection(  # type: ignore
                *[set(data.columns) for data in flight_tech_data_dict.values()]
            )

            for tech_name, flight_tech_data in flight_tech_data_dict.items():

                logger.debug(f"{step.enter_level()} Rename tech columns: {flight.name}")

                tech_data = rename_tech_columns(
                    flight_tech_data, tech_name, flight.merge_col, common_cols
                )

                flight_tech_data_list.append(tech_data)

                step.exit_level()

            logger.debug(
                f"{step.continue_step()} Merge tech data for flight: {flight.name}"
            )

            flight_data: pd.DataFrame

            if len(flight_tech_data_list) > 1:

                if flight.merge_col is None:
                    error = f"Merge column not specified for flight {flight.name}, with multiple techs."
                    logger.error(error)
                    raise ValueError(error)

                flight_data = merge_datasets(flight_tech_data_list, flight.merge_col)
                logger.debug(f"{step.info()}Merged tech data for flight: {flight.name}")

            else:
                flight_data = flight_tech_data_list[0]
                logger.debug(f"{step.info()}Single tech data for flight: {flight.name}")

            logger.debug(
                f"{step.next_step()} Format Timestamp data for flight: {flight.name}"
            )

            flight_data = format_timestamp(flight_data)

            logger.debug(
                f"{step.next_step()} Downselect data features for flight: {flight.name}"
            )

            flight_data = filter_features(flight_data)

            if relative_time:
                logger.debug(
                    f"{step.next_step()} Convert Timestamp to relative time for flight: {flight.name}"
                )
                flight_data = convert_to_relative_time(flight_data)
            else:
                logger.debug(
                    f"{step.next_step()}Keeping absolute Timestamp for flight: {flight.name}"
                )

            if project_coords:
                logger.debug(
                    f"{step.next_step()} Projecting coordinates for flight: {flight.name}"
                )
                flight_data = project_coordinates(flight_data)
            else:
                logger.debug(
                    f"{step.next_step()} Keeping longitude and latitude for flight: {flight.name}"
                )

            if alt_median_abs_deviation:
                logger.debug(
                    f"{step.next_step()} Filtering based on median absolute deviation for Altitude for flight: {flight.name}"
                )
                flight_data = get_median_abs_deviation(flight_data)
            else:
                logger.debug(
                    f"{step.next_step()} Not filtering based on median absolute deviation for Altitude for flight: {flight.name}"
                )

            logger.debug(
                f"{step.next_step()} Enumerating data for flight: {flight.name}"
            )

            flight_data.insert(0, "Index", range(0, len(flight_data)))

            if fill:
                flight_data = flight_data.ffill().bfill()
                logger.debug(
                    f"{step.next_step()} Filled data for flight: {flight.name}"
                )
            else:
                logger.debug(
                    f"{step.next_step()} Not filling data for flight: {flight.name}"
                )

            if save_cleaned_data:
                logger.debug(
                    f"{step.next_step()} Saving cleaned data for flight: {flight.name}"
                )

                output_dir = os.getenv("DATASET_CLEAN_HOME")

                if output_dir is None:
                    error = "Environment variable 'DATASET_CLEAN_HOME' is not set."
                    logger.error(error)
                    raise EnvironmentError(error)

                os.makedirs(output_dir, exist_ok=True)

                output_path = os.path.join(
                    output_dir,
                    get_flight_id(dataset.num, flight.name) + ".csv",
                )
                flight_data.to_csv(output_path, index=False)
            else:
                logger.debug(
                    f"{step.next_step()} Not saving cleaned data for flight: {flight.name}"
                )

            if dataset.num not in data_dict:
                data_dict[dataset.num] = {}
            data_dict[dataset.num][flight.name] = flight_data

            step.exit_level()
        step.exit_level()

    return data_dict


if __name__ == "__main__":
    process_datasets(relative_time=True, project_coords=True)

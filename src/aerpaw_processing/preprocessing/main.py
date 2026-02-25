import logging
import pandas as pd
from aerpaw_processing.preprocessing.step_tracker import StepTracker
from aerpaw_processing.preprocessing.utils import (
    load_datasets,
    convert_columns,
    merge_datasets,
)
from aerpaw_processing.resources.config.config_init import CONFIG, load_env

load_env()


logger = logging.getLogger(__name__)
step = StepTracker()


def main():
    for dataset in CONFIG.datasets:
        logger.info(f"Processing dataset: {dataset.name}")
        logger.info(f"Number of flights: {len(dataset.flights)}")

        logger.info(f"{step.next_step()} Load dataset files")

        for flight in dataset.flights:
            logger.info(f"{step.enter_level()} Load files for flight: {flight.name}")

            tech_data: dict[str, pd.DataFrame] = {}

            for tech in flight.tech_list:

                logger.info(
                    f"{step.enter_level()} Load tech files for tech: {tech.name}"
                )

                data_list = load_datasets(dataset.num, tech.files)

                logger.info(
                    f"{step.next_step()} Convert column names for tech: {tech.name}"
                )

                data_list = convert_columns(data_list, CONFIG)

                logger.info(
                    f"{step.next_step()} Merge flight files for tech: {tech.name}"
                )

                data: pd.DataFrame

                if len(data_list) > 1:
                    if flight.merge_col is None:
                        error = f"Merge column not specified for flight {flight.name}, tech {tech.name} with multiple files."
                        logger.error(error)
                        raise ValueError(error)
                    data = merge_datasets(data_list, flight.merge_col)
                    logger.info(f"Merged files for tech: {tech.name}")
                else:
                    data = data_list[0]
                    logger.info(f"Single file loaded for tech: {tech.name}")

                tech_data[tech.name] = data

                step.exit_level()

            logger.info(f"{step.next_step()} Merge tech data for flight: {flight.name}")

        logger.info(
            "Step 1: Load dataset files for each flight and each flight technology"
        )

        logger.info("Step 1: Merge dataset files for flights with multiple files")

        for flight in dataset.flights:
            for tech in flight.tech_list:
                for tech in flight.tech_list:
                    if len(tech.files) > 1:
                        logger.info(
                            f"Merging files for {tech.name} in flight {flight.name}"
                        )
                        tech.files
                    else:
                        tech.merged_data = pd.read_csv(tech.files[0])


if __name__ == "__main__":
    main()

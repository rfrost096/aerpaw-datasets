from aerpaw_processing.details import get_all_flight_details
from aerpaw_processing.utils import load_config
from aerpaw_processing.feature_analysis import graph_feature
import os
import pandas as pd

script_path = os.path.realpath(__file__)
script_directory = os.path.dirname(script_path)


def main():
    all_flight_details = get_all_flight_details()
    if all_flight_details is None:
        print("No flight details found.")
        return
    all_flight_details.load_all_analysis_data()
    config = load_config()

    with open(os.path.join(script_directory, "table_slide.tex"), "r") as template_file:
        table_template = template_file.read()

    with open(os.path.join(script_directory, "summary_slides.tex"), "w") as out_file:

        column_summary = all_flight_details.get_latex_column_summary()
        column_summary_title = "Dataset Features"

        current_slide = table_template.replace(
            "<<<INSERT_TITLE>>>", column_summary_title
        )
        current_slide = current_slide.replace("<<<INSERT_TABLE>>>", column_summary)

        out_file.write(current_slide)
        out_file.write("\n\n")

        column_checklist = all_flight_details.get_latex_column_checklist()
        column_checklist_title = "Dataset Features Checklist"

        current_slide = table_template.replace(
            "<<<INSERT_TITLE>>>", column_checklist_title
        )
        current_slide = current_slide.replace("<<<INSERT_TABLE>>>", column_checklist)

        out_file.write(current_slide)
        out_file.write("\n\n")

        for dataset in config.datasets:
            table_insert = all_flight_details.get_latex_dataset_summary(dataset.num)
            dataset_summary_title = f"Dataset {dataset.num} Summary"

            current_slide = table_template.replace(
                "<<<INSERT_TITLE>>>", dataset_summary_title
            )
            current_slide = current_slide.replace("<<<INSERT_TABLE>>>", table_insert)

            out_file.write(current_slide)
            out_file.write("\n\n")

    with open(os.path.join(script_directory, "graph_slide.tex"), "r") as template_file:
        graph_template = template_file.read()

    with open(
        os.path.join(script_directory, "correlation_slides.tex"), "w"
    ) as out_file:
        for dataset in config.datasets:
            flight_list = [
                flight.data
                for flight in all_flight_details.flights[dataset.num].values()
            ]
            dfs = pd.concat(flight_list, ignore_index=True)

            save_path = os.path.join(
                script_directory, f"dataset_{dataset.num}_correlation.png"
            )

            graph_feature(
                dfs,
                "mutual",
                save_path=save_path,
            )

            correlation_graph_title = f"Dataset {dataset.num} Feature Correlation"
            current_slide = graph_template.replace(
                "<<<INSERT_TITLE>>>", correlation_graph_title
            )

            current_slide = current_slide.replace("<<<INSERT_IMAGE_PATH>>>", save_path)
            out_file.write(current_slide)
            out_file.write("\n\n")


if __name__ == "__main__":
    main()

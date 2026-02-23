from aerpaw_processing.details import get_all_flight_details
from aerpaw_processing.utils import load_config
import os

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
        slide_template = template_file.read()

    with open(os.path.join(script_directory, "summary_slides.tex"), "w") as out_file:

        column_summary = all_flight_details.get_latex_column_summary()
        column_summary_title = "Dataset Features"

        current_slide = slide_template.replace(
            "<<<INSERT_TITLE>>>", column_summary_title
        )
        current_slide = current_slide.replace("<<<INSERT_TABLE>>>", column_summary)

        out_file.write(current_slide)
        out_file.write("\n\n")

        column_checklist = all_flight_details.get_latex_column_checklist()
        column_checklist_title = "Dataset Features Checklist"

        current_slide = slide_template.replace(
            "<<<INSERT_TITLE>>>", column_checklist_title
        )
        current_slide = current_slide.replace("<<<INSERT_TABLE>>>", column_checklist)

        out_file.write(current_slide)
        out_file.write("\n\n")

        for dataset in config.datasets:
            dataset_summary = all_flight_details.get_latex_dataset_summary(dataset.num)

            dataset_summary_title = f"Dataset {dataset.num} Summary"

            current_slide = slide_template.replace(
                "<<<INSERT_TITLE>>>", dataset_summary_title
            )
            current_slide = current_slide.replace("<<<INSERT_TABLE>>>", dataset_summary)

            out_file.write(current_slide)
            out_file.write("\n\n")


if __name__ == "__main__":
    main()

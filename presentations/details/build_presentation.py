from aerpaw_processing.graph.graph_utils import graph_average_mutual_correlation
from aerpaw_processing.resources.config.config_init import load_env
import os
from aerpaw_processing.analysis.analysis_main import (
    DatasetFlightDetails,
    FlightCharacteristic,
)

load_env()

script_path = os.path.realpath(__file__)
script_directory = os.path.dirname(script_path)


def main():
    details = DatasetFlightDetails()

    with open(
        os.path.join(script_directory, "../table_slide.tex"), "r"
    ) as template_file:
        table_template = template_file.read()

    with open(os.path.join(script_directory, "summary_slides.tex"), "w") as out_file:

        details_summary_characteristics = [
            FlightCharacteristic.UNIQUE_COLUMNS,
            FlightCharacteristic.NUM_ROWS,
            FlightCharacteristic.TIMESTAMP_MEAN_STD,
            FlightCharacteristic.DISTANCE_MEAN_STD,
        ]
        details_summary_df = details.get_characteristics(
            details_summary_characteristics
        )
        column_summary = details_summary_df.to_latex(
            index=False, escape=False, column_format="l p{4cm} r l l"
        ).replace("_", "\\_")
        column_summary = "\\tiny\n" + column_summary
        column_summary_title = "Dataset Features"

        current_slide = table_template.replace(
            "<<<INSERT_TITLE>>>", column_summary_title
        )
        current_slide = current_slide.replace("<<<INSERT_TABLE>>>", column_summary)

        out_file.write(current_slide)
        out_file.write("\n\n")

    with open(
        os.path.join(script_directory, "../graph_slide.tex"), "r"
    ) as template_file:
        graph_template = template_file.read()

    with open(
        os.path.join(script_directory, "correlation_slides.tex"), "w"
    ) as out_file:
        df_list = [df for df in details.flight_dict.values()]

        img_filename = f"flight_average_mutual_correlation.png"

        save_path = os.path.join(script_directory, img_filename)

        graph_average_mutual_correlation(
            df_list,
            label="RSRP",
            project_cords=True,
            save_path=save_path,
        )

        correlation_graph_title = f"Flight Average Mutual Correlation"
        current_slide = graph_template.replace(
            "<<<INSERT_TITLE>>>", correlation_graph_title
        )

        current_slide = current_slide.replace(
            "<<<INSERT_IMAGE_PATH>>>", "./" + img_filename
        )
        out_file.write(current_slide)
        out_file.write("\n\n")


if __name__ == "__main__":
    main()

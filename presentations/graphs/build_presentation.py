import os
from aerpaw_processing.graph.graph_main import (
    graph_avg_mutual_helper,
    graph_spatial_rsrp_correlation_helper,
    graph_fast_fading_correlation_helper,
)
from aerpaw_processing.preprocessing.preprocess_utils import get_all_flight_ids
from aerpaw_processing.resources.config.config_init import load_env

load_env()

script_path = os.path.realpath(__file__)
script_directory = os.path.dirname(script_path)


def main():
    with open(
        os.path.join(script_directory, "../graph_slide.tex"), "r"
    ) as template_file:
        graph_template = template_file.read()

    with open(
        os.path.join(script_directory, "average_mutual_slide.tex"), "w"
    ) as out_file:

        for tech in ["LTE_4G", "NR_5G"]:
            img_filename = f"average_mutual_helper_{tech}.png"
            save_path = os.path.join(script_directory, "images", img_filename)

            if not os.path.exists(save_path):
                graph_avg_mutual_helper(
                    project_cords=True, label="RSRP_" + tech, save_path=save_path
                )

            title = f"Average Mutual Correlation ({tech})".replace("_", "\\_")
            current_slide = graph_template.replace("<<<INSERT_TITLE>>>", title)

            current_slide = current_slide.replace(
                "<<<INSERT_IMAGE_PATH>>>", "./images/" + img_filename
            )
            out_file.write(current_slide)
            out_file.write("\n\n")

    flight_ids = get_all_flight_ids()

    with open(
        os.path.join(script_directory, "spatial_correlation_slides.tex"), "w"
    ) as out_file:

        for flight_id in flight_ids:
            img_filename = f"spatial_correlation_{flight_id}.png"
            save_path = os.path.join(script_directory, "images", img_filename)

            if not os.path.exists(save_path):
                graph_spatial_rsrp_correlation_helper(
                    flight_id=flight_id, label="RSRP", save_path=save_path
                )

            title = f"Spatial Correlation ({flight_id})".replace("_", "\\_")
            current_slide = graph_template.replace("<<<INSERT_TITLE>>>", title)

            current_slide = current_slide.replace(
                "<<<INSERT_IMAGE_PATH>>>", "./images/" + img_filename
            )
            out_file.write(current_slide)
            out_file.write("\n\n")

    with open(
        os.path.join(script_directory, "fast_fading_correlation_slides.tex"), "w"
    ) as out_file:

        for flight_id in flight_ids:
            img_filename = f"fast_fading_correlation_{flight_id}.png"
            save_path = os.path.join(script_directory, "images", img_filename)
            if not os.path.exists(save_path):
                graph_fast_fading_correlation_helper(
                    flight_id=flight_id, label="RSRP", save_path=save_path
                )
            title = f"Fast Fading Correlation ({flight_id})".replace("_", "\\_")
            current_slide = graph_template.replace("<<<INSERT_TITLE>>>", title)

            current_slide = current_slide.replace(
                "<<<INSERT_IMAGE_PATH>>>", "./images/" + img_filename
            )
            out_file.write(current_slide)
            out_file.write("\n\n")


if __name__ == "__main__":
    main()

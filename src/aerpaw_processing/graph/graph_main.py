from aerpaw_processing.preprocessing.preprocess_main import process_datasets
from aerpaw_processing.graph.graph_utils import graph_average_mutual_correlation
from aerpaw_processing.resources.config.config_init import load_env


load_env()


def main():
    data_dict = process_datasets(
        save_cleaned_data=False, relative_time=True, project_coords=True
    )
    df_list = [df for dataset in data_dict.values() for df in dataset.values()]
    graph_average_mutual_correlation(
        df_list,
        label="RSRP",
        project_cords=True,
        save_path=None,
    )


if __name__ == "__main__":
    main()

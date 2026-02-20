import pandas as pd
from aerpaw_processing.utils import (
    find_file,
    load_config,
    load_datasets,
    merge_datasets,
    convert_columns,
    merge_tech_datasets,
)


def main():
    config = load_config()

    for dataset in config.datasets:
        print(f"Dataset {dataset.num}")

        for flight in dataset.flights:
            print(f"\tFlight {flight.name}")

            data: dict[str, pd.DataFrame | None] = {"LTE_4G": None, "NR_5G": None}

            for tech in data.keys():
                file_list: list[str] | None = None

                if tech == "LTE_4G":
                    file_list = flight.files.LTE_4G
                elif tech == "NR_5G":
                    file_list = flight.files.NR_5G

                if file_list is not None:
                    abs_path_list = find_file(dataset.num, file_list)

                    if abs_path_list is None:
                        return

                    data_list = load_datasets(abs_path_list)

                    data_list = convert_columns(data_list, config)

                    if len(data_list) > 1:
                        data[tech] = merge_datasets(data_list, "ID")
                    else:
                        data[tech] = data_list[0]

            formatted_data: pd.DataFrame

            if data["LTE_4G"] is not None and data["NR_5G"] is not None:
                formatted_data = merge_tech_datasets(data["LTE_4G"], data["NR_5G"])
            elif data["LTE_4G"] is not None:
                formatted_data = data["LTE_4G"]
            elif data["NR_5G"] is not None:
                formatted_data = data["NR_5G"]
            else:
                return

            for col in formatted_data.keys():
                print(f"\t\t{col}")


if __name__ == "__main__":
    main()

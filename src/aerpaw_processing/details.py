import pandas as pd
from aerpaw_processing.utils import (
    find_file,
    load_config,
    load_datasets,
    merge_datasets,
    convert_columns,
    merge_tech_datasets,
)
from aerpaw_processing.resources.config import Category
import numpy as np
from enum import Enum
from pyproj import Geod
from typing import Any
import re

LONGITUDE_COLUMN_NAME = "Longitude"
LATITUDE_COLUMN_NAME = "Latitude"
ALTITUDE_COLUMN_NAME = "Altitude"
TIMESTAMP_COLUMN_NAME = "Timestamp"

config = load_config()

signal_quality_category: Category
location_category: Category

for category in config.categories:
    if category.category == "Signal Quality":
        signal_quality_category = category
    if category.category == "Location":
        location_category = category


class FlightType(Enum):
    SCAN_2D = "SCAN_2D"


class FlightDetails:

    data: pd.DataFrame
    name: str
    data_type: FlightType

    analysis_data: pd.DataFrame

    mad_filtered: pd.DataFrame

    duration: pd.Timedelta
    avg_time_step: pd.Timedelta
    avg_spatial_step: float
    area_covered: float
    sample_density: float
    avg_alt: float
    altitude_range: float

    details: pd.DataFrame

    def __init__(
        self, data: pd.DataFrame, name: str, sdata_type: FlightType = FlightType.SCAN_2D
    ):
        self.data = data
        self.name = name
        self.data_type = sdata_type

    def load_analysis_data(self):
        self.get_analysis_data()
        self.get_duration()
        self.get_avg_time_step()
        self.calculate_avg_spatial_step()
        self.get_area_covered()
        self.get_sample_density()
        self.get_avg_alt()
        self.get_altitude_range()

    def get_details(self):

        self.load_analysis_data()

        formatted_duration = (
            f"{self.duration.components.minutes}m {self.duration.components.seconds}s"
        )

        formatted_avg_ts = f"{self.avg_time_step.components.seconds}s {self.avg_time_step.components.microseconds}µs"

        formatted_spatial_step = f"{self.avg_spatial_step:.3g} m"

        formatted_area_covered = f"{self.area_covered:.3g} m²"

        formatted_sample_density = f"{self.sample_density:.3g} smple/m²"

        formatted_avg_alt = f"{self.avg_alt:.3g} m"

        formatted_altitude_range = f"{self.altitude_range:.3g} m"

        details_dict: dict[str, float | str] = {
            "Name": self.name,
            "Duration": formatted_duration,
            "Avg Time Step": formatted_avg_ts,
            "Avg Spatial Step": formatted_spatial_step,
            "Area Covered": formatted_area_covered,
            "Sample Density": formatted_sample_density,
            "Alt Avg": formatted_avg_alt,
            "Alt Range": formatted_altitude_range,
        }

        self.details = pd.DataFrame([details_dict])
        return self.details

    def get_analysis_data(self):
        if self.data_type == FlightType.SCAN_2D:
            self.analysis_data = self.get_mad()

    def get_mad(self):
        median_alt = self.data[ALTITUDE_COLUMN_NAME].median()
        mad = (self.data[ALTITUDE_COLUMN_NAME] - median_alt).abs().median()

        multiplier = 4.0

        lower_bound = median_alt - (multiplier * mad)
        upper_bound = median_alt + (multiplier * mad)

        self.mad_filtered = self.data[
            (self.data[ALTITUDE_COLUMN_NAME] >= lower_bound)
            & (self.data[ALTITUDE_COLUMN_NAME] <= upper_bound)
        ]
        return self.mad_filtered

    def get_duration(self):

        min_timestamp: pd.Timestamp = self.analysis_data[TIMESTAMP_COLUMN_NAME].min()
        max_timestamp: pd.Timestamp = self.analysis_data[TIMESTAMP_COLUMN_NAME].max()

        self.duration = max_timestamp - min_timestamp

        return self.duration

    def get_avg_time_step(self):
        timestamp_data: pd.Series[pd.Timestamp] = self.analysis_data[
            TIMESTAMP_COLUMN_NAME
        ]
        self.avg_time_step = timestamp_data.diff().mean()

        return self.avg_time_step

    def calculate_avg_spatial_step(self):
        sorted_data = self.analysis_data.sort_values(by=TIMESTAMP_COLUMN_NAME)

        lons = sorted_data[LONGITUDE_COLUMN_NAME].to_numpy(dtype=float)
        lats = sorted_data[LATITUDE_COLUMN_NAME].to_numpy(dtype=float)
        alts = sorted_data[ALTITUDE_COLUMN_NAME].to_numpy(dtype=float)

        if len(sorted_data) < 2:
            return 0.0

        geod = Geod(ellps="WGS84")

        _, _, distance_2d = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])

        distance_alt = alts[1:] - alts[:-1]

        spatial_steps = np.sqrt(distance_2d**2 + distance_alt**2)

        self.avg_spatial_step = spatial_steps.mean()

        return self.avg_spatial_step

    def get_area_covered(self):

        if self.data_type == FlightType.SCAN_2D:
            clean_data = self.analysis_data.dropna(
                subset=[LONGITUDE_COLUMN_NAME, LATITUDE_COLUMN_NAME]
            )

            if len(clean_data) < 3:
                return 0.0

            lons = clean_data[LONGITUDE_COLUMN_NAME].values
            lats = clean_data[LATITUDE_COLUMN_NAME].values

            geod = Geod(ellps="WGS84")

            area, _ = geod.polygon_area_perimeter(lons, lats)

            self.area_covered = abs(area)

            return self.area_covered

        else:
            print("Area covered not implemented for flight type:", self.data_type)
            return 0.0

    def get_sample_density(self):

        if self.area_covered == 0.0:
            return 0.0

        num_samples = len(self.analysis_data)

        self.sample_density = num_samples / self.area_covered

        return self.sample_density

    def get_avg_alt(self):

        self.avg_alt = self.analysis_data[ALTITUDE_COLUMN_NAME].mean()
        return self.avg_alt

    def get_altitude_range(self):

        alt_min = self.analysis_data[ALTITUDE_COLUMN_NAME].min()
        alt_max = self.analysis_data[ALTITUDE_COLUMN_NAME].max()

        self.altitude_range = alt_max - alt_min

        return self.altitude_range


class AllFlights:

    flights: dict[int, dict[str, FlightDetails]]

    def __init__(self, flights: dict[int, dict[str, pd.DataFrame]]):
        self.flights = {
            dataset_num: {
                flight_name: FlightDetails(flight_data, flight_name)
                for flight_name, flight_data in dataset_flights.items()
            }
            for dataset_num, dataset_flights in flights.items()
        }

    def load_all_analysis_data(self):
        for _, dataset in self.flights.items():
            for _, flight in dataset.items():
                flight.get_details()

    def get_latex_dataset_summary(self, dataset_num: int) -> str:
        flight_dict = self.flights[dataset_num]

        details_list = [flight.details for flight in flight_dict.values()]

        details_table = pd.concat(details_list, ignore_index=True)

        max_len = 20
        if "Name" in details_table.columns:
            details_table["Name"] = details_table["Name"].apply(
                lambda x: str(x)[:max_len] + "..." if len(str(x)) > max_len else str(x)  # type: ignore
            )

        num_cols = len(details_table.columns)
        col_format = "|" + "|".join(["c"] * num_cols) + "|"

        latex_table = details_table.to_latex(index=False, column_format=col_format)

        latex_table = latex_table.replace("\\toprule", "\\hline")
        latex_table = latex_table.replace("\\midrule", "\\hline")
        latex_table = latex_table.replace("\\bottomrule", "\\hline")

        latex_table = re.sub(r"\\\\", r"\\\\ \\hline", latex_table)

        latex_table = latex_table.replace("\\hline\n\\hline", "\\hline")

        latex_table = f"\\resizebox{{\\textwidth}}{{!}}{{{latex_table}}}"

        latex_lines = [
            r"\begin{table}[ht]",
            r"\centering",
            latex_table,
            f"\\caption{{Summary of Flights for Dataset {dataset_num}}}",
            r"\end{table}",
        ]

        return "\n".join(latex_lines)

    def get_column_summary(self) -> pd.DataFrame:
        """
        Generates a summary DataFrame grouping flights by their exact column sets.
        """
        grouped_flights: dict[tuple[str, ...], dict[str, list[Any]]] = {}

        for dataset_num, dataset_flights in self.flights.items():
            for flight_name, flight_details in dataset_flights.items():
                raw_cols = flight_details.data.columns.astype(str).tolist()
                cols_key = tuple(sorted(raw_cols))

                flight_id = f"{dataset_num}:{flight_name}"

                if cols_key not in grouped_flights:
                    grouped_flights[cols_key] = {
                        "Flight_IDs": [],
                        "Columns": raw_cols,
                    }

                grouped_flights[cols_key]["Flight_IDs"].append(flight_id)

        summary_rows = []
        for group_info in grouped_flights.values():
            summary_rows.append(
                {
                    "Flight_IDs": group_info["Flight_IDs"],
                    "Columns": group_info["Columns"],
                }
            )

        df = pd.DataFrame(summary_rows)  # type: ignore

        if not df.empty:
            df = df.sort_values(by="Flight_IDs", ascending=True).reset_index(drop=True)

        return df

    def get_latex_column_summary(self) -> str:
        """
        Generates a LaTeX table string summarizing flights and their columns.
        Optimized for Beamer: Reduced font sizes, tighter row spacing, and
        width scaling to ensure the table fits vertically and horizontally.
        """
        summary_df = self.get_column_summary()

        latex_lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\renewcommand{\arraystretch}{0.75}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{|p{0.25\textwidth}|p{0.75\textwidth}|}",
            r"\hline",
            r"\textbf{\scriptsize Flight IDs} & \textbf{\scriptsize Columns} \\",
            r"\hline",
        ]

        for _, row in summary_df.iterrows():
            flight_ids = sorted(row["Flight_IDs"])

            location_cols = [str(col.name) for col in location_category.cols]
            row_cols = [str(c) for c in row["Columns"] if c not in location_cols]

            cols_safe = [str(c).replace("_", r"\_") for c in row_cols]
            ids_safe = [str(fid).replace("_", r"\_") for fid in flight_ids]

            col_str = ", ".join(cols_safe)
            ids_str = r" \newline ".join(ids_safe)

            line = f"{{\\raggedright\\scriptsize {ids_str}}} & \\raggedright\\scriptsize {col_str} \\tabularnewline"

            latex_lines.append(line)
            latex_lines.append(r"\hline")

        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"}")
        latex_lines.append(r"\caption{Summary of Flight Column Sets}")
        latex_lines.append(r"\end{table}")

        return "\n".join(latex_lines)

    def get_latex_column_checklist(self) -> str:
        """
        Generates a LaTeX table acting as a checklist indicating whether
        specific signal quality columns exist in each flight's dataset.
        Optimized to fit better on presentation slides (e.g., Beamer).
        """
        target_cols = [col.name for col in signal_quality_category.cols]

        safe_headers = [col.replace("_", r"\_") for col in target_cols]

        latex_lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\footnotesize",
            r"\renewcommand{\arraystretch}{0.9}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{|l|" + "c|" * len(target_cols) + "}",
            r"\hline",
        ]

        header_cells = [r"\textbf{Flight ID}"] + [
            f"\\textbf{{{col}}}" for col in safe_headers
        ]
        latex_lines.append(" & ".join(header_cells) + r" \\")
        latex_lines.append(r"\hline")

        for dataset_num, dataset_flights in self.flights.items():
            for flight_name, flight_details in dataset_flights.items():
                flight_id = f"{dataset_num}:{flight_name}"
                safe_flight_id = flight_id.replace("_", r"\_")

                row_cells = [safe_flight_id]
                data_columns = [str(c) for c in flight_details.data.columns]

                for target_col in target_cols:
                    if any(target_col in data_col for data_col in data_columns):
                        row_cells.append(r"\checkmark")
                    else:
                        row_cells.append("")

                latex_lines.append(" & ".join(row_cells) + r" \\")
                latex_lines.append(r"\hline")

        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"}")
        latex_lines.append(
            r"\caption{Checklist of Signal Quality Columns in Each Flight}"
        )
        latex_lines.append(r"\end{table}")

        return "\n".join(latex_lines)


def get_all_flight_details():

    flights: dict[int, dict[str, pd.DataFrame]] = {}

    for dataset in config.datasets:

        for flight in dataset.flights:

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

            flights[dataset.num] = flights.get(dataset.num, {})
            flights[dataset.num][flight.name] = formatted_data

    all_flights = AllFlights(flights)

    return all_flights

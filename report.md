# Preprocessing Report

Generated on: 2026-03-22 20:13:53

## Step: read_data

- **Total files read**: 29

### Datasets Read

| Dataset | Tech | Flight | Filepath | Rows | Cols |
| --- | --- | --- | --- | --- | --- |
| 18 | LTE_4G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf1_cellid_with_header.csv | 2,898 | 6 |
| 18 | LTE_4G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf1_rsrp_with_header.csv | 2,898 | 6 |
| 18 | LTE_4G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf1_sinr_with_header.csv | 2,898 | 6 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_cellid_with_header.csv | 2,898 | 6 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_cqi_with_header.csv | 2,898 | 6 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_mcs_with_header.csv | 2,898 | 6 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_ri_with_header.csv | 2,898 | 6 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_rsrp_with_header.csv | 2,898 | 6 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_sinr_with_header.csv | 2,898 | 6 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/input_throughput_with_header.csv | 786 | 6 |
| 18 | LTE_4G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf1_cellid_with_header.csv | 2,949 | 6 |
| 18 | LTE_4G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf1_rsrp_with_header.csv | 2,949 | 6 |
| 18 | LTE_4G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf1_sinr_with_header.csv | 2,949 | 6 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_cellid_with_header.csv | 2,949 | 6 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_cqi_with_header.csv | 2,949 | 6 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_mcs_with_header.csv | 2,949 | 6 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_ri_with_header.csv | 2,949 | 6 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_rsrp_with_header.csv | 2,949 | 6 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_sinr_with_header.csv | 2,949 | 6 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/input_throughput_with_header.csv | 818 | 6 |
| 22 | LTE_4G | 4G LTE Flight | Logs/lte.csv | 2,021 | 23 |
| 24 | NR_5G | Nemo 5G 30m Flight 1 | Logs/nemo_5G_NR_altitude30m_flight1.csv | 30,730 | 282 |
| 24 | NR_5G | Nemo 5G 30m Flight 2 | Logs/nemo_5G_NR_altitude30m_flight2.csv | 30,981 | 283 |
| 24 | LTE_4G | PawPrints 4G 30m Flight 1 | Logs/pawprints_4G_LTE_altitude30m_flight1.csv | 2,421 | 29 |
| 24 | LTE_4G | PawPrints 4G 30m Flight 2 | Logs/pawprints_4G_LTE_altitude30m_flight2.csv | 2,078 | 30 |
| 24 | LTE_4G | PawPrints 4G 50m Flight | Logs/pawprints_4G_LTE_altitude50m.csv | 1,533 | 29 |
| 24 | NR_5G | PawPrints 5G 30m Flight 1 | Logs/pawprints_5G_NR_altitude30m_flight1.csv | 791 | 14 |
| 24 | NR_5G | PawPrints 5G 30m Flight 2 | Logs/pawprints_5G_NR_altitude30m_flight2.csv | 731 | 18 |
| 24 | NR_5G | PawPrints 5G 50m Flight | Logs/pawprints_5G_NR_altitude50m.csv | 709 | 14 |


## Step: rename_columns

- Columns renamed to standard names using `alias_map`.

### Columns Renamed per Dataset

| Dataset | Tech | Flight | Filepath | Cols Changed |
| --- | --- | --- | --- | --- |
| 18 | LTE_4G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf1_cellid_with_header.csv | 2 |
| 18 | LTE_4G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf1_rsrp_with_header.csv | 2 |
| 18 | LTE_4G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf1_sinr_with_header.csv | 2 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_cellid_with_header.csv | 2 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_cqi_with_header.csv | 2 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_mcs_with_header.csv | 2 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_ri_with_header.csv | 2 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_rsrp_with_header.csv | 2 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_sinr_with_header.csv | 2 |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/input_throughput_with_header.csv | 1 |
| 18 | LTE_4G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf1_cellid_with_header.csv | 2 |
| 18 | LTE_4G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf1_rsrp_with_header.csv | 2 |
| 18 | LTE_4G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf1_sinr_with_header.csv | 2 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_cellid_with_header.csv | 2 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_cqi_with_header.csv | 2 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_mcs_with_header.csv | 2 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_ri_with_header.csv | 2 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_rsrp_with_header.csv | 2 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_sinr_with_header.csv | 2 |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/input_throughput_with_header.csv | 1 |
| 22 | LTE_4G | 4G LTE Flight | Logs/lte.csv | 18 |
| 24 | NR_5G | Nemo 5G 30m Flight 1 | Logs/nemo_5G_NR_altitude30m_flight1.csv | 15 |
| 24 | NR_5G | Nemo 5G 30m Flight 2 | Logs/nemo_5G_NR_altitude30m_flight2.csv | 15 |
| 24 | LTE_4G | PawPrints 4G 30m Flight 1 | Logs/pawprints_4G_LTE_altitude30m_flight1.csv | 23 |
| 24 | LTE_4G | PawPrints 4G 30m Flight 2 | Logs/pawprints_4G_LTE_altitude30m_flight2.csv | 23 |
| 24 | LTE_4G | PawPrints 4G 50m Flight | Logs/pawprints_4G_LTE_altitude50m.csv | 23 |
| 24 | NR_5G | PawPrints 5G 30m Flight 1 | Logs/pawprints_5G_NR_altitude30m_flight1.csv | 13 |
| 24 | NR_5G | PawPrints 5G 30m Flight 2 | Logs/pawprints_5G_NR_altitude30m_flight2.csv | 16 |
| 24 | NR_5G | PawPrints 5G 50m Flight | Logs/pawprints_5G_NR_altitude50m.csv | 13 |

- **Total standardized columns (sample)**: 6
- **Columns (sample)**: `ID, Timestamp, Longitude, Latitude, Altitude, Cell_ID`


## Step: remove_columns

- Unwanted columns (not defined in CONFIG) removed.

### Columns Removed per Dataset

| Dataset | Tech | Flight | Filepath | Cols Removed | Remaining Columns |
| --- | --- | --- | --- | --- | --- |
| 18 | LTE_4G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf1_cellid_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, Cell_ID` |
| 18 | LTE_4G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf1_rsrp_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, RSRP` |
| 18 | LTE_4G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf1_sinr_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, SINR` |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_cellid_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, Cell_ID` |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_cqi_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, CQI` |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_mcs_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, MCS` |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_ri_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, RI` |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_rsrp_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, RSRP` |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/inputf2_sinr_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, SINR` |
| 18 | NR_5G | Yaw 315 Flight | Ericsson_Amir/Dryad/yaw315/input_throughput_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, Throughput` |
| 18 | LTE_4G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf1_cellid_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, Cell_ID` |
| 18 | LTE_4G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf1_rsrp_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, RSRP` |
| 18 | LTE_4G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf1_sinr_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, SINR` |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_cellid_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, Cell_ID` |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_cqi_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, CQI` |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_mcs_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, MCS` |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_ri_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, RI` |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_rsrp_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, RSRP` |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/inputf2_sinr_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, SINR` |
| 18 | NR_5G | Yaw 45 Flight | Ericsson_Amir/Dryad/yaw45/input_throughput_with_header.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, Throughput` |
| 22 | LTE_4G | 4G LTE Flight | Logs/lte.csv | 5 | `RSRP, RSRQ, RSSI, ASU, EARFCN, PCI, TA, CI, TAC, Bands, Modes, MCC, MNC, Is_Connected, Timestamp, Longitude, Latitude, Altitude` |
| 24 | NR_5G | Nemo 5G 30m Flight 1 | Logs/nemo_5G_NR_altitude30m_flight1.csv | 267 | `Timestamp, Longitude, Latitude, Altitude, Bands, MCC, MNC, RSRP, RSRQ, RSSI, SINR, TA, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation` |
| 24 | NR_5G | Nemo 5G 30m Flight 2 | Logs/nemo_5G_NR_altitude30m_flight2.csv | 268 | `Timestamp, Longitude, Latitude, Altitude, Bands, MCC, MNC, RSRP, RSRQ, RSSI, SINR, TA, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation` |
| 24 | LTE_4G | PawPrints 4G 30m Flight 1 | Logs/pawprints_4G_LTE_altitude30m_flight1.csv | 6 | `Timestamp, Longitude, Latitude, Altitude, RSSNR, RSRP, RSRQ, RSSI, ASU, EARFCN, PCI, TA, CI, TAC, Level, Bands, Modes, MCC, MNC, Is_Connected, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation` |
| 24 | LTE_4G | PawPrints 4G 30m Flight 2 | Logs/pawprints_4G_LTE_altitude30m_flight2.csv | 7 | `Timestamp, Longitude, Latitude, Altitude, RSSNR, RSRP, RSRQ, RSSI, ASU, EARFCN, PCI, TA, CI, TAC, Level, Bands, Modes, MCC, MNC, Is_Connected, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation` |
| 24 | LTE_4G | PawPrints 4G 50m Flight | Logs/pawprints_4G_LTE_altitude50m.csv | 6 | `Timestamp, Longitude, Latitude, Altitude, RSSNR, RSRP, RSRQ, RSSI, ASU, EARFCN, PCI, TA, CI, TAC, Level, Bands, Modes, MCC, MNC, Is_Connected, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation` |
| 24 | NR_5G | PawPrints 5G 30m Flight 1 | Logs/pawprints_5G_NR_altitude30m_flight1.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, NR_Type, RSRP, RSRQ, SINR, Level, ASU, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation` |
| 24 | NR_5G | PawPrints 5G 30m Flight 2 | Logs/pawprints_5G_NR_altitude30m_flight2.csv | 2 | `Timestamp, Longitude, Latitude, Altitude, NR_Type, CSI_RSRP, CSI_RSRQ, CSI_SINR, RSRP, RSRQ, SINR, Level, ASU, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation` |
| 24 | NR_5G | PawPrints 5G 50m Flight | Logs/pawprints_5G_NR_altitude50m.csv | 1 | `Timestamp, Longitude, Latitude, Altitude, NR_Type, RSRP, RSRQ, SINR, Level, ASU, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation` |


## Step: combine_tech_files

- Multiple KPI files combined into 13 technology-specific datasets per flight.

### Combined Datasets

| Dataset | Tech | Flight | Rows | Cols | Prev Rows | Columns |
| --- | --- | --- | --- | --- | --- | --- |
| 18 | LTE_4G | Yaw 315 Flight | 2,898 | 7 | 2,898, 2,898, 2,898 | `Timestamp, Longitude, Latitude, Altitude, Cell_ID, RSRP, SINR`
| 18 | NR_5G | Yaw 315 Flight | 6,582 | 11 | 2,898, 2,898, 2,898, 2,898, 2,898, 2,898, 786 | `Timestamp, Longitude, Latitude, Altitude, Cell_ID, CQI, MCS, RI, RSRP, SINR, Throughput`
| 18 | LTE_4G | Yaw 45 Flight | 2,949 | 7 | 2,949, 2,949, 2,949 | `Timestamp, Longitude, Latitude, Altitude, Cell_ID, RSRP, SINR`
| 18 | NR_5G | Yaw 45 Flight | 6,716 | 11 | 2,949, 2,949, 2,949, 2,949, 2,949, 2,949, 818 | `Timestamp, Longitude, Latitude, Altitude, Cell_ID, CQI, MCS, RI, RSRP, SINR, Throughput`
| 22 | LTE_4G | 4G LTE Flight | 2,021 | 18 | 2,021 | `RSRP, RSRQ, RSSI, ASU, EARFCN, PCI, TA, CI, TAC, Bands, Modes, MCC, MNC, Is_Connected, Timestamp, Longitude, Latitude, Altitude`
| 24 | NR_5G | Nemo 5G 30m Flight 1 | 30,730 | 15 | 30,730 | `Timestamp, Longitude, Latitude, Altitude, Bands, MCC, MNC, RSRP, RSRQ, RSSI, SINR, TA, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | NR_5G | Nemo 5G 30m Flight 2 | 30,981 | 15 | 30,981 | `Timestamp, Longitude, Latitude, Altitude, Bands, MCC, MNC, RSRP, RSRQ, RSSI, SINR, TA, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | LTE_4G | PawPrints 4G 30m Flight 1 | 2,421 | 23 | 2,421 | `Timestamp, Longitude, Latitude, Altitude, RSSNR, RSRP, RSRQ, RSSI, ASU, EARFCN, PCI, TA, CI, TAC, Level, Bands, Modes, MCC, MNC, Is_Connected, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | LTE_4G | PawPrints 4G 30m Flight 2 | 2,078 | 23 | 2,078 | `Timestamp, Longitude, Latitude, Altitude, RSSNR, RSRP, RSRQ, RSSI, ASU, EARFCN, PCI, TA, CI, TAC, Level, Bands, Modes, MCC, MNC, Is_Connected, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | LTE_4G | PawPrints 4G 50m Flight | 1,533 | 23 | 1,533 | `Timestamp, Longitude, Latitude, Altitude, RSSNR, RSRP, RSRQ, RSSI, ASU, EARFCN, PCI, TA, CI, TAC, Level, Bands, Modes, MCC, MNC, Is_Connected, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | NR_5G | PawPrints 5G 30m Flight 1 | 791 | 13 | 791 | `Timestamp, Longitude, Latitude, Altitude, NR_Type, RSRP, RSRQ, SINR, Level, ASU, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | NR_5G | PawPrints 5G 30m Flight 2 | 731 | 16 | 731 | `Timestamp, Longitude, Latitude, Altitude, NR_Type, CSI_RSRP, CSI_RSRQ, CSI_SINR, RSRP, RSRQ, SINR, Level, ASU, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | NR_5G | PawPrints 5G 50m Flight | 709 | 13 | 709 | `Timestamp, Longitude, Latitude, Altitude, NR_Type, RSRP, RSRQ, SINR, Level, ASU, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`


## Step: combine_flight_techs

- Multiple technologies combined into 11 single flight datasets.

### Combined Flight Datasets

| Dataset | Flight | Rows | Cols | Techs Combined | Columns |
| --- | --- | --- | --- | --- | --- |
| 18 | Yaw 315 Flight | 6,582 | 14 | LTE_4G, NR_5G | `Timestamp, Longitude, Latitude, Altitude, Cell_ID_LTE_4G, RSRP_LTE_4G, SINR_LTE_4G, Cell_ID_NR_5G, CQI_NR_5G, MCS_NR_5G, RI_NR_5G, RSRP_NR_5G, SINR_NR_5G, Throughput_NR_5G`
| 18 | Yaw 45 Flight | 6,716 | 14 | LTE_4G, NR_5G | `Timestamp, Longitude, Latitude, Altitude, Cell_ID_LTE_4G, RSRP_LTE_4G, SINR_LTE_4G, Cell_ID_NR_5G, CQI_NR_5G, MCS_NR_5G, RI_NR_5G, RSRP_NR_5G, SINR_NR_5G, Throughput_NR_5G`
| 22 | 4G LTE Flight | 2,021 | 18 | LTE_4G | `RSRP_LTE_4G, RSRQ_LTE_4G, RSSI_LTE_4G, ASU_LTE_4G, EARFCN_LTE_4G, PCI_LTE_4G, TA_LTE_4G, CI_LTE_4G, TAC_LTE_4G, Bands_LTE_4G, Modes_LTE_4G, MCC_LTE_4G, MNC_LTE_4G, Is_Connected_LTE_4G, Timestamp, Longitude, Latitude, Altitude`
| 24 | Nemo 5G 30m Flight 1 | 30,730 | 15 | NR_5G | `Timestamp, Longitude, Latitude, Altitude, Bands_NR_5G, MCC_NR_5G, MNC_NR_5G, RSRP_NR_5G, RSRQ_NR_5G, RSSI_NR_5G, SINR_NR_5G, TA_NR_5G, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | Nemo 5G 30m Flight 2 | 30,981 | 15 | NR_5G | `Timestamp, Longitude, Latitude, Altitude, Bands_NR_5G, MCC_NR_5G, MNC_NR_5G, RSRP_NR_5G, RSRQ_NR_5G, RSSI_NR_5G, SINR_NR_5G, TA_NR_5G, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | PawPrints 4G 30m Flight 1 | 2,421 | 23 | LTE_4G | `Timestamp, Longitude, Latitude, Altitude, RSSNR_LTE_4G, RSRP_LTE_4G, RSRQ_LTE_4G, RSSI_LTE_4G, ASU_LTE_4G, EARFCN_LTE_4G, PCI_LTE_4G, TA_LTE_4G, CI_LTE_4G, TAC_LTE_4G, Level_LTE_4G, Bands_LTE_4G, Modes_LTE_4G, MCC_LTE_4G, MNC_LTE_4G, Is_Connected_LTE_4G, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | PawPrints 4G 30m Flight 2 | 2,078 | 23 | LTE_4G | `Timestamp, Longitude, Latitude, Altitude, RSSNR_LTE_4G, RSRP_LTE_4G, RSRQ_LTE_4G, RSSI_LTE_4G, ASU_LTE_4G, EARFCN_LTE_4G, PCI_LTE_4G, TA_LTE_4G, CI_LTE_4G, TAC_LTE_4G, Level_LTE_4G, Bands_LTE_4G, Modes_LTE_4G, MCC_LTE_4G, MNC_LTE_4G, Is_Connected_LTE_4G, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | PawPrints 4G 50m Flight | 1,533 | 23 | LTE_4G | `Timestamp, Longitude, Latitude, Altitude, RSSNR_LTE_4G, RSRP_LTE_4G, RSRQ_LTE_4G, RSSI_LTE_4G, ASU_LTE_4G, EARFCN_LTE_4G, PCI_LTE_4G, TA_LTE_4G, CI_LTE_4G, TAC_LTE_4G, Level_LTE_4G, Bands_LTE_4G, Modes_LTE_4G, MCC_LTE_4G, MNC_LTE_4G, Is_Connected_LTE_4G, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | PawPrints 5G 30m Flight 1 | 791 | 13 | NR_5G | `Timestamp, Longitude, Latitude, Altitude, NR_Type_NR_5G, RSRP_NR_5G, RSRQ_NR_5G, SINR_NR_5G, Level_NR_5G, ASU_NR_5G, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | PawPrints 5G 30m Flight 2 | 731 | 16 | NR_5G | `Timestamp, Longitude, Latitude, Altitude, NR_Type_NR_5G, CSI_RSRP_NR_5G, CSI_RSRQ_NR_5G, CSI_SINR_NR_5G, RSRP_NR_5G, RSRQ_NR_5G, SINR_NR_5G, Level_NR_5G, ASU_NR_5G, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`
| 24 | PawPrints 5G 50m Flight | 709 | 13 | NR_5G | `Timestamp, Longitude, Latitude, Altitude, NR_Type_NR_5G, RSRP_NR_5G, RSRQ_NR_5G, SINR_NR_5G, Level_NR_5G, ASU_NR_5G, Base_Station_Distance, Base_Station_Bearing, Base_Station_Elevation`


## Step: interpolate_to_label

- Interpolated and standardized data around label columns.

### Interpolated Datasets

| Dataset | Flight | Rows | Cols | Prev Rows |
| --- | --- | --- | --- | --- |
| 18 | Yaw 315 Flight | 2,898 | 14 | 6,582 |
| 18 | Yaw 45 Flight | 2,949 | 14 | 6,716 |
| 22 | 4G LTE Flight | 2,021 | 18 | 2,021 |
| 24 | Nemo 5G 30m Flight 1 | 1,598 | 15 | 30,730 |
| 24 | Nemo 5G 30m Flight 2 | 1,599 | 15 | 30,981 |
| 24 | PawPrints 4G 30m Flight 1 | 2,421 | 23 | 2,421 |
| 24 | PawPrints 4G 30m Flight 2 | 2,078 | 23 | 2,078 |
| 24 | PawPrints 4G 50m Flight | 1,533 | 23 | 1,533 |
| 24 | PawPrints 5G 30m Flight 1 | 791 | 13 | 791 |
| 24 | PawPrints 5G 30m Flight 2 | 731 | 16 | 731 |
| 24 | PawPrints 5G 50m Flight | 709 | 13 | 709 |


## Step: project_coordinates

- Projected Latitude/Longitude to x/y coordinates using AEQD projection.
- Base Tower (Origin): LW1 (35.727451, -78.695974)

### Projected 3D Trajectories (Colored by RSRP)

#### Flight: Yaw 315 Flight

[![Yaw 315 Flight](report_assets/dataset_18_Yaw_315_Flight_3d.png)](report_assets/dataset_18_Yaw_315_Flight_3d.html)

#### Flight: Yaw 45 Flight

[![Yaw 45 Flight](report_assets/dataset_18_Yaw_45_Flight_3d.png)](report_assets/dataset_18_Yaw_45_Flight_3d.html)

#### Flight: 4G LTE Flight

[![4G LTE Flight](report_assets/dataset_22_4G_LTE_Flight_3d.png)](report_assets/dataset_22_4G_LTE_Flight_3d.html)

#### Flight: Nemo 5G 30m Flight 1

[![Nemo 5G 30m Flight 1](report_assets/dataset_24_Nemo_5G_30m_Flight_1_3d.png)](report_assets/dataset_24_Nemo_5G_30m_Flight_1_3d.html)

#### Flight: Nemo 5G 30m Flight 2

[![Nemo 5G 30m Flight 2](report_assets/dataset_24_Nemo_5G_30m_Flight_2_3d.png)](report_assets/dataset_24_Nemo_5G_30m_Flight_2_3d.html)

#### Flight: PawPrints 4G 30m Flight 1

[![PawPrints 4G 30m Flight 1](report_assets/dataset_24_PawPrints_4G_30m_Flight_1_3d.png)](report_assets/dataset_24_PawPrints_4G_30m_Flight_1_3d.html)

#### Flight: PawPrints 4G 30m Flight 2

[![PawPrints 4G 30m Flight 2](report_assets/dataset_24_PawPrints_4G_30m_Flight_2_3d.png)](report_assets/dataset_24_PawPrints_4G_30m_Flight_2_3d.html)

#### Flight: PawPrints 4G 50m Flight

[![PawPrints 4G 50m Flight](report_assets/dataset_24_PawPrints_4G_50m_Flight_3d.png)](report_assets/dataset_24_PawPrints_4G_50m_Flight_3d.html)

#### Flight: PawPrints 5G 30m Flight 1

[![PawPrints 5G 30m Flight 1](report_assets/dataset_24_PawPrints_5G_30m_Flight_1_3d.png)](report_assets/dataset_24_PawPrints_5G_30m_Flight_1_3d.html)

#### Flight: PawPrints 5G 30m Flight 2

[![PawPrints 5G 30m Flight 2](report_assets/dataset_24_PawPrints_5G_30m_Flight_2_3d.png)](report_assets/dataset_24_PawPrints_5G_30m_Flight_2_3d.html)

#### Flight: PawPrints 5G 50m Flight

[![PawPrints 5G 50m Flight](report_assets/dataset_24_PawPrints_5G_50m_Flight_3d.png)](report_assets/dataset_24_PawPrints_5G_50m_Flight_3d.html)



## Step: calculate_bin

- Partitioned the 3D data points into a spherical grid centered at the Base Station with 0.05 rad bins.

### Projected 3D Trajectories (Colored by Angular Bin)

#### Flight: Yaw 315 Flight

[![Yaw 315 Flight](report_assets/dataset_18_Yaw_315_Flight_bin_3d.png)](report_assets/dataset_18_Yaw_315_Flight_bin_3d.html)

#### Flight: Yaw 45 Flight

[![Yaw 45 Flight](report_assets/dataset_18_Yaw_45_Flight_bin_3d.png)](report_assets/dataset_18_Yaw_45_Flight_bin_3d.html)

#### Flight: 4G LTE Flight

[![4G LTE Flight](report_assets/dataset_22_4G_LTE_Flight_bin_3d.png)](report_assets/dataset_22_4G_LTE_Flight_bin_3d.html)

#### Flight: Nemo 5G 30m Flight 1

[![Nemo 5G 30m Flight 1](report_assets/dataset_24_Nemo_5G_30m_Flight_1_bin_3d.png)](report_assets/dataset_24_Nemo_5G_30m_Flight_1_bin_3d.html)

#### Flight: Nemo 5G 30m Flight 2

[![Nemo 5G 30m Flight 2](report_assets/dataset_24_Nemo_5G_30m_Flight_2_bin_3d.png)](report_assets/dataset_24_Nemo_5G_30m_Flight_2_bin_3d.html)

#### Flight: PawPrints 4G 30m Flight 1

[![PawPrints 4G 30m Flight 1](report_assets/dataset_24_PawPrints_4G_30m_Flight_1_bin_3d.png)](report_assets/dataset_24_PawPrints_4G_30m_Flight_1_bin_3d.html)

#### Flight: PawPrints 4G 30m Flight 2

[![PawPrints 4G 30m Flight 2](report_assets/dataset_24_PawPrints_4G_30m_Flight_2_bin_3d.png)](report_assets/dataset_24_PawPrints_4G_30m_Flight_2_bin_3d.html)

#### Flight: PawPrints 4G 50m Flight

[![PawPrints 4G 50m Flight](report_assets/dataset_24_PawPrints_4G_50m_Flight_bin_3d.png)](report_assets/dataset_24_PawPrints_4G_50m_Flight_bin_3d.html)

#### Flight: PawPrints 5G 30m Flight 1

[![PawPrints 5G 30m Flight 1](report_assets/dataset_24_PawPrints_5G_30m_Flight_1_bin_3d.png)](report_assets/dataset_24_PawPrints_5G_30m_Flight_1_bin_3d.html)

#### Flight: PawPrints 5G 30m Flight 2

[![PawPrints 5G 30m Flight 2](report_assets/dataset_24_PawPrints_5G_30m_Flight_2_bin_3d.png)](report_assets/dataset_24_PawPrints_5G_30m_Flight_2_bin_3d.html)

#### Flight: PawPrints 5G 50m Flight

[![PawPrints 5G 50m Flight](report_assets/dataset_24_PawPrints_5G_50m_Flight_bin_3d.png)](report_assets/dataset_24_PawPrints_5G_50m_Flight_bin_3d.html)



## Step: correlation_computation

- Computed pairwise spatial correlation of RSRP for each angular bin and aggregated by 5m radial separation distance bins.

### Spatial Correlation Profiles

#### Flight: Yaw 315 Flight

[![Yaw 315 Flight](report_assets/dataset_18_Yaw_315_Flight_correlation.png)](report_assets/dataset_18_Yaw_315_Flight_correlation.html)

#### Flight: Yaw 45 Flight

[![Yaw 45 Flight](report_assets/dataset_18_Yaw_45_Flight_correlation.png)](report_assets/dataset_18_Yaw_45_Flight_correlation.html)

#### Flight: 4G LTE Flight

[![4G LTE Flight](report_assets/dataset_22_4G_LTE_Flight_correlation.png)](report_assets/dataset_22_4G_LTE_Flight_correlation.html)

#### Flight: Nemo 5G 30m Flight 1

[![Nemo 5G 30m Flight 1](report_assets/dataset_24_Nemo_5G_30m_Flight_1_correlation.png)](report_assets/dataset_24_Nemo_5G_30m_Flight_1_correlation.html)

#### Flight: Nemo 5G 30m Flight 2

[![Nemo 5G 30m Flight 2](report_assets/dataset_24_Nemo_5G_30m_Flight_2_correlation.png)](report_assets/dataset_24_Nemo_5G_30m_Flight_2_correlation.html)

#### Flight: PawPrints 4G 30m Flight 1

[![PawPrints 4G 30m Flight 1](report_assets/dataset_24_PawPrints_4G_30m_Flight_1_correlation.png)](report_assets/dataset_24_PawPrints_4G_30m_Flight_1_correlation.html)

#### Flight: PawPrints 4G 30m Flight 2

[![PawPrints 4G 30m Flight 2](report_assets/dataset_24_PawPrints_4G_30m_Flight_2_correlation.png)](report_assets/dataset_24_PawPrints_4G_30m_Flight_2_correlation.html)

#### Flight: PawPrints 4G 50m Flight

[![PawPrints 4G 50m Flight](report_assets/dataset_24_PawPrints_4G_50m_Flight_correlation.png)](report_assets/dataset_24_PawPrints_4G_50m_Flight_correlation.html)

#### Flight: PawPrints 5G 30m Flight 1

[![PawPrints 5G 30m Flight 1](report_assets/dataset_24_PawPrints_5G_30m_Flight_1_correlation.png)](report_assets/dataset_24_PawPrints_5G_30m_Flight_1_correlation.html)

#### Flight: PawPrints 5G 30m Flight 2

[![PawPrints 5G 30m Flight 2](report_assets/dataset_24_PawPrints_5G_30m_Flight_2_correlation.png)](report_assets/dataset_24_PawPrints_5G_30m_Flight_2_correlation.html)

#### Flight: PawPrints 5G 50m Flight

[![PawPrints 5G 50m Flight](report_assets/dataset_24_PawPrints_5G_50m_Flight_correlation.png)](report_assets/dataset_24_PawPrints_5G_50m_Flight_correlation.html)

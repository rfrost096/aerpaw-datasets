#!/bin/bash

INIT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$INIT_DIR/init.sh"

flights=("Dataset_18_Yaw_315_Flight" "Dataset_18_Yaw_45_Flight" "Dataset_22_4G_LTE_Flight" "Dataset_24_Nemo_5G_30m_Flight_1" "Dataset_24_Nemo_5G_30m_Flight_2" "Dataset_24_PawPrints_4G_30m_Flight_1" "Dataset_24_PawPrints_4G_30m_Flight_2" "Dataset_24_PawPrints_4G_50m_Flight" "Dataset_24_PawPrints_5G_30m_Flight_1" "Dataset_24_PawPrints_5G_30m_Flight_2" "Dataset_24_PawPrints_5G_50m_Flight")

# for flight in "${flights[@]}"; do
#   graph_spatial_rsrp_correlation --flight-id "$flight" --label RSRP --save-path /home/ryan/VT/Spring26/GradProject/DatasetWorkspace/aerpaw-datasets/src/aerpaw_processing/graph/radial_correlation/$flight
# done

for flight in "${flights[@]}"; do
  graph_fast_fading_correlation --flight-id "$flight" --label RSRP --save-path /home/ryan/VT/Spring26/GradProject/DatasetWorkspace/aerpaw-datasets/src/aerpaw_processing/graph/fast_fading_correlation/$flight
done

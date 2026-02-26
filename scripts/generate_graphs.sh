#!/bin/bash

# generate_graphs.sh
#
# Example invocations for the four graph entry points defined in pyproject.toml.
#
# Flight IDs are formatted as  <dataset_num>_<Flight_Name_With_Underscores>
# and must match the values printed by get_all_flight_ids().  Run any graph
# command with --help to see the full list of valid IDs for your environment.
#
# Optional: export SAVE_DIR to write PNGs/HTMLs instead of opening windows.
#   export SAVE_DIR=/tmp/graphs
# ---------------------------------------------------------------------------

INIT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$INIT_DIR/init.sh"

SAVE_DIR="${SAVE_DIR:-}"   # empty → display interactively

_save() {
    # Usage: _save "filename.png"  →  returns --save-path /tmp/graphs/filename.png
    #                                  or nothing if SAVE_DIR is unset
    if [ -n "$SAVE_DIR" ]; then
        mkdir -p "$SAVE_DIR"
        echo "--save-path $SAVE_DIR/$1"
    fi
}

# ---------------------------------------------------------------------------
# 1. Mutual correlation for a single flight (RSRP as the target label)
#    Uses Dataset 18 – Yaw 315 Flight as a representative example.
# ---------------------------------------------------------------------------
echo "==> graph-mutual  (RSRP, Dataset 18 – Yaw 315 Flight)"
graph-mutual \
    --flight-id  "Dataset_18_Yaw_315_Flight" \
    --label      RSRP \
    $(_save "mutual_rsrp_18_yaw315.png")

# ---------------------------------------------------------------------------
# 2. Average mutual correlation across ALL flights (RSRP only)
#    No flight selection needed – the command iterates every loaded flight.
# ---------------------------------------------------------------------------
echo "==> graph-avg-mutual  (RSRP, all flights)"
graph-avg-mutual \
    $(_save "avg_mutual_rsrp_all_flights.png")

# ---------------------------------------------------------------------------
# 3a. PCI spatial distribution – Dataset 22, 4G LTE Flight
#     PCI is categorical by default; filter-features is off by default, so
#     no extra flags are required beyond specifying the flight and label.
# ---------------------------------------------------------------------------
echo "==> graph-label  (PCI, Dataset 22 – 4G LTE Flight)"
graph-label \
    --flight-ids "Dataset_22_4G_LTE_Flight" \
    --label      PCI \
    $(_save "label_pci_22_4g_lte.html")

# ---------------------------------------------------------------------------
# 3b. RSRP spatial distribution – every dedicated 4G LTE flight
#     Includes:
#       • Dataset 22  – 4G LTE Flight
#       • Dataset 24  – PawPrints 4G 30m Flight 1
#       • Dataset 24  – PawPrints 4G 30m Flight 2
#       • Dataset 24  – PawPrints 4G 50m Flight
#
#     Dataset 18 flights carry merged LTE_4G + NR_5G data and are therefore
#     excluded from this "4G LTE only" comparison.
#     --no-categorical treats RSRP as a continuous signal-strength value.
# ---------------------------------------------------------------------------
echo "==> graph-label  (RSRP, all 4G LTE flights)"
graph-label \
    --flight-ids "Dataset_22_4G_LTE_Flight,Dataset_24_PawPrints_4G_30m_Flight_1,Dataset_24_PawPrints_4G_30m_Flight_2,Dataset_24_PawPrints_4G_50m_Flight" \
    --label          RSRP \
    --no-categorical \
    $(_save "label_rsrp_all_4g_lte.html")

# ---------------------------------------------------------------------------
# 4. Temporal RSRP animation – Dataset 24, Nemo 5G 30m Flight 1
#    Altitude MAD filtering is enabled by default in graph-label-temporal,
#    which keeps the z-axis meaningful at a fixed 30 m flight altitude.
# ---------------------------------------------------------------------------
echo "==> graph-label-temporal  (RSRP, Dataset 24 – Nemo 5G 30m Flight 1)"
graph-label-temporal \
    --flight-ids "Dataset_24_Nemo_5G_30m_Flight_1" \
    --label      RSRP \
    $(_save "temporal_rsrp_24_nemo5g_flight1.html")
"""Tower location data for AERPAW Lake Wheeler location"""

from dataclasses import dataclass

CALCULATE_ALTITUDE_WITH_DEM = False


@dataclass
class Tower:
    name: str
    lat: float
    lon: float
    alt: float


towers = [
    Tower("LW1", 35.727451, -78.695974, 10.0),
    Tower("LW2", 35.728210, -78.700930, 5.5499954),
    Tower("LW3", 35.724933, -78.691912, -0.23000336),
    Tower("LW4", 35.73320368161924, -78.69838096280522, 16.529999),
    Tower("LW5", 35.74292132108924, -78.69961868795892, 25.029991),
]

if CALCULATE_ALTITUDE_WITH_DEM:

    import rasterio  # type: ignore
    from rasterio.warp import transform  # type: ignore
    from dotenv import load_dotenv, find_dotenv
    import os
    from typing import cast

    load_dotenv(find_dotenv("config.env"))

    dem_path = str(os.getenv("DEM_PATH"))

    def get_elevation(lat: float, lon: float) -> float | None:
        with rasterio.open(dem_path) as src:  # type: ignore
            crs_str: str = src.crs.to_string()  # type: ignore
            if crs_str != "EPSG:4326":
                xs, ys = cast(tuple[list[float], list[float]], transform("EPSG:4326", src.crs, [lon], [lat]))  # type: ignore
                x, y = xs[0], ys[0]
            else:
                x, y = lon, lat
            row, col = cast(tuple[int, int], src.index(x, y))  # type: ignore
            data = src.read(1)  # type: ignore
            if 0 <= row < src.height and 0 <= col < src.width:  # type: ignore
                elevation = cast(float, data[row, col])
                return elevation
            else:
                return None

    LW1_alt = get_elevation(towers[0].lat, towers[0].lon)  # Base station altitude

    if LW1_alt is None:
        ValueError("Coords not in DEM")

    LW1_alt = cast(float, LW1_alt)

    for tower in towers[1:]:
        elevation = get_elevation(tower.lat, tower.lon)
        if elevation is None:
            ValueError("Coords not in DEM")
        elevation = cast(float, elevation)
        tower.alt = elevation - LW1_alt  # Alt relative to base station

    for tower in towers:
        tower.alt += (
            10  # README.md has sources for radio equipment being 10m up each tower
        )
        print(tower.alt)

# Output:
# 10.0
# 5.5499954
# -0.23000336
# 16.529999
# 25.029991


# Dataset 12 locations.txt:
# LW1
#         origin_y=35.727451;
#         origin_x=-78.695974;
# ======================================
# LW2
#         origin_y=35.728210;
#         origin_x=-78.700930;
# ======================================
# LW3
#         origin_y=35.724933;
#         origin_x=-78.691912;
# ======================================
# LW4
#         origin_y=35.73320368161924;
#         origin_x=-78.69838096280522;
# ======================================
# LW5
#         origin_y=35.74292132108924;
#         origin_x=-78.69961868795892;
# ======================================

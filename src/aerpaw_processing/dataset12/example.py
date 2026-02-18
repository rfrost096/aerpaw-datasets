# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 09:52:58 2024

@author: afahimr

Febuary 11th 2026:
Modified by Ryan Frost
- Modified data paths to use .env file in
    this python project.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os
from dotenv import load_dotenv

script_dir = Path(__file__).resolve().parent

load_dotenv(script_dir / "../config.env")

data_path = (
    str(os.getenv("DATASET_12_HOME"))
    + "/"
    + str(os.getenv("DATASET_12_SAMPLE_DIRECTORY"))
    + "/"
    + str(os.getenv("DATASET_12_MEASUREMENT_NAME"))
)

data_file = data_path + ".sigmf-data"
meta_file = data_path + ".sigmf-meta"

gps_path = (
    str(os.getenv("DATASET_12_HOME"))
    + "/"
    + str(os.getenv("DATASET_12_SAMPLE_DIRECTORY"))
    + "/"
    + str(os.getenv("DATASET_12_GPS_NAME"))
)

GPS_data_file = gps_path + ".sigmf-data"
GPS_meta_file = gps_path + ".sigmf-meta"

# Load your data from SigMF files using appropriate libraries
with open(GPS_meta_file, "r") as f:
    metadata_GPS = json.load(f)

captures_GPS = metadata_GPS["captures"]
timestamp_gps = captures_GPS[0]["gps:time_stamp"]
position = captures_GPS[0]["gps:position"]
GPSx = np.array(position["x"])
GPSy = np.array(position["y"])
GPSz = np.array(position["z"])


# Load your data from SigMF files using appropriate libraries
with open(meta_file, "r") as f:
    metadata = json.load(f)

captures = metadata["captures"]
timestamp_meas = captures[0]["measurement:time_stamp"]
values = captures[0]["measurement:values"]
mX = np.array(values["mX"])
mY = np.array(values["mY"])
mZ = np.array(values["mZ"])
RSS1 = np.array(values["RSS1"])
RSS2 = np.array(values["RSS2"])


# Setting origin for LW1 node
origin_x, origin_y = -78.695974, 35.727451

# Calculate the distance from the origin for each point
mdistance = np.sqrt((mX - origin_x) ** 2 + (mY - origin_y) ** 2) * 1.113195e5

# Plotting GPS Altitude
plt.figure(1)
plt.grid(True)
plt.plot(timestamp_gps, GPSz, "x")
plt.xlabel("time (s)")
plt.ylabel("Altitude (m)")
plt.title("GPS points")

# Plotting Measurement Altitude
plt.figure(2)
plt.grid(True)
plt.plot(timestamp_meas, mZ, "x")
plt.xlabel("time (s)")
plt.ylabel("Altitude (m)")
plt.title("Measurement points")

# Plotting RSS1 vs Distance
plt.figure(3)
plt.grid(True)
plt.plot(mdistance, RSS1, "x")
plt.xlabel("distance (m)")
plt.ylabel("RSS (dB)")
plt.xlim([0, 2500])
plt.ylim([-60, 0])

# Plotting RSS2 vs Distance
plt.figure(4)
plt.grid(True)
plt.plot(mdistance, RSS2, "x")
plt.xlabel("distance (m)")
plt.ylabel("RSS (dB)")
plt.xlim([0, 2500])
plt.ylim([-60, 0])

# Plotting Distance and RSS1 over time
plt.figure(5)
plt.grid(True)
plt.plot(timestamp_meas, mdistance, label="Distance")
plt.xlabel("time (s)")
plt.ylabel("Distance (m)")
plt.ylim([0, 2500])
plt.twinx()
plt.plot(timestamp_meas, RSS1, label="RSS1", color="orange")
plt.ylabel("RSS (dB)")
plt.ylim([-60, 0])

# Plotting Distance and RSS2 over time
plt.figure(6)
plt.grid(True)
plt.plot(timestamp_meas, mdistance, label="Distance")
plt.xlabel("time (s)")
plt.ylabel("Distance (m)")
plt.ylim([0, 2500])
plt.twinx()
plt.plot(timestamp_meas, RSS2, label="RSS2", color="orange")
plt.ylabel("RSS (dB)")
plt.ylim([-60, 0])

# 3D Scatter Plot of RSS1
fig = plt.figure(7)
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(mX, mY, mZ, c=RSS1, cmap="jet", s=10)  # type: ignore
plt.colorbar(sc)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Altitude")
plt.xlim([-78.702, -78.69])
plt.ylim([35.724, 35.735])
ax.set_zlim([0, 115])
plt.title("RSS1 (dB)")

# 3D Scatter Plot of RSS2
fig = plt.figure(8)
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(mX, mY, mZ, c=RSS2, cmap="jet", s=10)  # type: ignore
plt.colorbar(sc)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Altitude")
plt.xlim([-78.702, -78.69])
plt.ylim([35.724, 35.735])
ax.set_zlim([0, 115])
plt.title("RSS2 (dB)")

# 2D Scatter Plot of RSS1
plt.figure(9)
plt.grid(True)
plt.scatter(mX, mY, c=RSS1, cmap="jet", s=10)
plt.colorbar()
plt.xlim([-78.702, -78.69])
plt.ylim([35.724, 35.735])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("RSS1 (dB)")

# 2D Scatter Plot of RSS2
plt.figure(10)
plt.grid(True)
plt.scatter(mX, mY, c=RSS2, cmap="jet", s=10)
plt.colorbar()
plt.xlim([-78.702, -78.69])
plt.ylim([35.724, 35.735])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("RSS2 (dB)")

# Show all plots
plt.show()

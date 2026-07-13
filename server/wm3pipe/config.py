"""Static configuration for the WeatherMesh-3 real-time inference pipeline."""
import os

REGION = os.environ.get("AWS_REGION", "us-east-1")
OUTPUT_BUCKET = os.environ.get("WM3_OUTPUT_BUCKET", "wm3-forecasts-194290773983")
WEIGHTS_S3 = os.environ.get("WM3_WEIGHTS_S3", "s3://wm3-gpu-194290773983/model/WeatherMesh3.pt")

GFS_BUCKET = "noaa-gfs-bdp-pds"
GFS_RESOLUTION = "0p25"
CW_NAMESPACE = "WeatherMesh3"

NI, NJ = 1440, 721

CORE_PRESSURE_VARS = ["129_z", "130_t", "131_u", "132_v", "133_q"]
CORE_SFC_VARS = ["165_10u", "166_10v", "167_2t", "151_msl"]
EXTRA_SFC_INPUT = ["45_tcc", "168_2d", "246_100u", "247_100v"]

# model var id -> (GFS .idx variable token, unit scale to model units)
GFS_PRESSURE = {
    "129_z": ("HGT", 9.80665),
    "130_t": ("TMP", 1.0),
    "131_u": ("UGRD", 1.0),
    "132_v": ("VGRD", 1.0),
    "133_q": ("SPFH", 1.0),
}
# model var id -> (GFS token, .idx level string)
GFS_SURFACE = {
    "165_10u": ("UGRD", "10 m above ground"),
    "166_10v": ("VGRD", "10 m above ground"),
    "167_2t": ("TMP", "2 m above ground"),
    "151_msl": ("PRMSL", "mean sea level"),
}
# model var id -> (GFS token, .idx level string, representation)
GFS_EXTRA = {
    "45_tcc": ("TCDC", "entire atmosphere", "fraction"),
    "168_2d": ("DPT", "2 m above ground", "norm"),
    "246_100u": ("UGRD", "100 m above ground", "norm"),
    "247_100v": ("VGRD", "100 m above ground", "norm"),
}

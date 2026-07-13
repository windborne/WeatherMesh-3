# WeatherMesh-3
The purpose of this repository is to define the architecture of [WindBorne](https://www.windbornesystems.com)'s WeatherMesh3 as outlined in our ICLR submission [WeatherMesh-3](https://arxiv.org/abs/2503.22235). 
We are not currently providing weights or any data processing used to run the model.

> **This fork adds a real-time inference pipeline** under [`server/`](server/): it fetches
> live GFS, runs WM-3 on a cloud GPU, and publishes API-shaped netCDF forecasts to S3 on a
> 6-hourly schedule with CloudWatch monitoring. Weights (~1.2 GB) and data stay out of git
> and are fetched from S3 at runtime. Start at [`server/README.md`](server/README.md).

# Installation
Install the required packages using: 
```
./install.sh
```

# To view the WeatherMesh3 architecture
Run our sample script at 
```
python3 model.py
```

Or import the script and interact with the model yourself 

```
from model import get_WeatherMesh3

model = get_WeatherMesh3()
print(model)
```

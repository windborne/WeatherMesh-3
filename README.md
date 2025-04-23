# WeatherMesh-3
The public repo for the source code of [WindBorne](https://www.windbornesystems.com)'s [WeatherMesh-3](https://arxiv.org/abs/2503.22235).

Install the weights at 
#JACK_CHANGE_LATER
needs to install:

/huge/deep/evaluation/joanrealtimesucks3/weights/model_epoch27_iter50394_step8399_loss0.079.pt
<-- install this to WeatherMesh3.pt

and

/huge/deep/evaluation/joanrealtimesucks3/weights/state_dict_epoch27_iter50394_step8399_loss0.079.pt # WE DON'T ACTUALLY NEED THIS CAUSE IT'S STORED IN THE WEIGHTS

to 

DOWNLOAD THE DATA TO the DATA FOLDER
#JACK_CHANGE_LATER

# Model Weights:
Install the 
```
from model import get_wm3

model = get_wm3()
```

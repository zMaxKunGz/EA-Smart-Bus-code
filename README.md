# Nvidia Jetson Setup step
1. Setup Jectpack OS 
2. Setup python environment
```
python venv -m env
source env/bin/activate
``` 
3. Install torch version 2.1.0 by downloading wheel
Torch: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 
```
export TORCH_INSTALL=path/to/wheel_file
pip install --no-cache $TORCH_INSTALL
```
4. Install ffmpeg libjpeg etc
```
sudo apt update
sudo apt install ffmpeg libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libswresample-dev libswresample-dev libpostproc-dev libjpeg-dev libpng-dev
pip3 install ffmpeg av
```
5. Switch to root to install Torchvision
The build gonna take long time.
```
sudo -s
source env/bin/activate
git clone https://github.com/pytorch/vision
python setup.py install
```
Covid Guidelines Detection is an open source repository that provides object tracking, social distance detection, and mask detection.

## Results

## Installation
To use the covid guidelines detection software is simple. All you need to do is clone this repository and download the required modules:
```
git clone https://github.com/HectorENevarez/guidelines_detection.git
```
Once the repository is cloned, navigate to the main directory and install the required modules:
```
pip install -r requirements.txt
pip install --upgrade tensorflow
```
The installation of tensorflow and the requirements text might be different based on your os. The following references for different operating systems are listed below:
- [tensorflow installation](https://www.tensorflow.org/install/pip)
- [Requirements.txt](https://pip.pypa.io/en/stable/reference/pip_install/#requirements-file-format)

## Use
Once all the required modules are installed properly, the model files need to be downloaded. YOLOV3 was used for the object detection and a custom model was used 
for the face mask detection. All the models need to be downloaded and placed in the ```models``` directory. The following models must be downloaded and added to the directory:
- [YOLOV3](https://pjreddie.com/media/files/yolov3.weights) **Clicking the link will download the weights**
- [Face Mask Detection](https://www.dropbox.com/sh/ufidzbijm0drzsf/AAArVo4sx4s1Ola1hcjvgIwCa?dl=0)

to use the software you'll need video's to test the software on. If the user has no video available, they can download
a traditional test video for object detection:
```
pip install youtube_dl
youtube-dl https://www.youtube.com/watch?v=pk96gqasGBQ -o streets.mp4
```
This will download [this](https://www.youtube.com/watch?v=pk96gqasGBQ) youtube video that works for testing. <br>

In order to use the software, a video must be specified. Using that recently downloaded video, the software can be quickly accessed:
```
python covid_tracker.py -v street.mp4
```

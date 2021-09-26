# OpenCV Learnings

In this repository i will be sharing my learning with OpenCV library for object detections. 

Code was prepared on following versions of libraries
```bash
OpenCV- 4.5.3

Numpy- 1.21.2

Python- 3.7.11
```


Here in **OpenCv_Modules** i have uploaded 9 chapters as follows.

## Chapter List
```
1) How to read Image, Video and Webcam 
2) How to convert colored images into Gray scaled image, different blur and edge detection techniques like GaussianBlur, Canny edge detector
3) Cropping and resize images
4) Draw different shapes and write text on image
5) Change Prespective of image
6) stacking images vertically and Horizontally
7) Trackbar- for color detection
8) Edge detection and edge Highlight
9) Face detection with *"HarCascadeClassifier"* pre trained model 
```
## Simple Project of detecting pedestrian with Pretrained HaarCascades Model for Body Classifier

![pedestrian detection with haarcascades](https://github.com/TheAshpak/Learning-OpenCV/blob/main/Output_img.png)

```
* Using OpenCV to read video frame by frame
* converting coloured images to Gray Scale image and passing it to haarcascades human body detection model
* Model returns four co-ordinates ,using those co-ordinates to plot bonding box around human body
```
### Advantages
```
* Haarcascades models are fast
* Requires low computational power
```
### Limitations
```
It is hard for haarcascades models to detect multiple objects simultaneously 
```
### Applications
```
* Real time human detection at traffic signals to avoid accidents
* Security cameras to detect human presence
```

# object-detection-by-using-mobileapp-IPweb camera
* Recently I’ve been doing a few projects in computer vision. And naturally I did one of the most popular projects, drum-rolls please… The Object Detection Using SSD MobileNet project! The project itself wasn’t too long, but it raises a lot of questions in a beginner’s mind. By the end of the project, I had gone over all the websites that promised me the explanation of codes in its entirety, What a scam! Well, although it wasn’t a waste of time, it sure wasn’t very efficient. 
Every website added on to what I had learnt from the previous page but none of the pages cleared all my doubts at once. In the end I was exhausted but hey, isn’t this a good blog idea? This brings me to the reason that I am writing today — To address general questions that might arise in a beginner’s mind while doing this project or while referring someone else’s code like we all do (What’s that? yeah, me neither). So, let’s go!
This particular project detects objects using the mobileNet SSD method. Therefore before proceeding, three files are a pre-requisite — ‘coco.names’, ‘ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt’ and ‘frozen_inference_graph.pb’.


Aim: To detect object Which is infrount of web camera with the help of opencv2 with Python programing.
In this project, I will take you through the task of object-detection-by-mobile app-IP web camera with Computer Vision Opencv2 using data with weights and configuration along with coco names to detect objects with YOLO algorithm. 

Lets Start Project:

* packages we need  
* List of objects that can be detected using this model
* Creating dnn Detection Model
* locating the trained model class path
* Lets' test on an image
* importing image from assets directory
* This image kinda big, let's Resize it
* resize a bit 
* Lets' declear some variables
* Threshold to detect object
* Threshold to detect object
* 0-1 higer value means lower suppress
* Formating for non-maxima suppression
* it removes overlap bounding boxes & keep the most confident ones. 
   - make sure bounding boxex and confident are List of floats, it shouldn't associate with numpy
* Take a look on actual image
* ind returns list we need index only
* set the rectangle around object
* Lets show the result 
* Let's test it on webcam
* we can also make a funtion just for image processing 
* Formating data
* ind returns list we need index only
* set the rectangle around object
* show accuracy level
* putting name of object
* esc to quit

* 







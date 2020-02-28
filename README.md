# 1. Introduction

Image classification is tricky and it becomes even trickier when I don't know exactly where in an image my objects of interest will appear or  what size they'll be or even how many of them I might find. 

In this lesion I will focus on the task of detecting vehicles in images taken from a camera mounted on the front of a car but the same principles apply to pedestrian detection, or traffic sign detection, or identifying any object I might be looking for in an image. 

Object detection and tracking is a central theme in computer vision and in this lesson I will be using what you might call traditional computer vision techniques to tackle this problem. 

I will first explore what kind of visual features I can extract from images in order to reliably classify vehicles. Next, I will look into searching an image for detections and then I will track those detections from frame to frame in a video stream. In the end of this lesson, I am going to implement a pipeline to detect and track vehicles in a video stream.


# 2. Manual Vehicle Detection
<p align="right">
 <img src="./img/1.png" width="600" height="300" />
 </p>
Assume, I will have an algorithm that's outputting bounding box positions and I'll want an easy way to plot them up over my images (like above). So, now is a good time to get familiar with the cv2.rectangle() function ([documentation](http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html)) that makes it easy to draw boxes of different size, shape and color.

    cv2.rectangle(image_to_draw_on, (x1, y1), (x2, y2), color, thick)
 
In this call to cv2.rectangle() my image_to_draw_on should be the copy of your image, then (x1, y1) and (x2, y2) are the x and y coordinates of any two opposing corners of the bounding box I want to draw. color is a 3-tuple, for example, (0, 0, 255) for blue, and thick is an optional integer parameter to define the box thickness.

If you want to investigate this function more closely, take a look at the bounding boxes exercise [code](https://github.com/A2Amir/Object-Detection/blob/master/code/the%20bounding%20boxes%20exercise.ipynb)

# 3. Features

Color, Apparent size, shape , position within image  are useful characteristics for identifying cars in an  image. All of these potential characteristics are features that I can use.  Features describe the characteristics of an object and with images, it really all comes down to intensity and gradients of intensity, and how these features capture the color and shape of an object.

Different features (like intensity and gradients of intensity) capture  the characteristic(s) about an object in an image. For example by **raw pixel intensity** (I mean essentially taking the image itself as features) give us color and shape information or **Histogram of pixel intensity** (With a histogram, I've discarded spatial information) give us only color information or **Gradients of pixel intensity** give us information about shaps only.

What features are more important may depend upon the appearance of  the objects in question. In most applications, I'll end up using a combination of features that give me the best results. 

<p align="right">
 <img src="./img/2.png" width="600" height="300" />
 </p>
 
 # 4. Color Features and Template matching
 
 The simplest feature I can get from images consists of raw color values. For instance, below is an image of a car. Let's say I want to find out whether some region (yellow box) of a new test image contains a car or not.  Well, using my known car image as is (red car), I can simply find the image difference between the car image and the test region (yellow box), and see the difference is small. 
This basically means subtracting the corresponding color values, aggregating the differences and comparing it with the threshold. 

<p align="right">
 <img src="./img/3.png" width="600" height="400" />
 </p>
 
Note: alternatively, I could compute the correlation between the car image and test region and check if that is high.

This general approach is known as template matching. My known image is the template and I try to match it with regions of the test image. Template matching does work in limited circumstances but isn't very helpful for my case. 

To figure out when template matching works and when it doesn't, let's play around with the OpenCV cv2.matchTemplate() function! In [the bounding boxes exercise](https://github.com/A2Amir/Object-Detection/blob/master/code/Template%20Matching.ipynb), I found six cars in the image above. This time, I am going to play the opposite game. Assuming I know these six cars are what I am looking for, I can use them as templates and search the image for matches.

<p align="right">
 <img src="./img/4.png" width="600" height="300" />
 </p>
 
As seen in [the Template Matching code](https://github.com/A2Amir/Object-Detection/blob/master/code/Template%20Matching.ipynb) with template matching we can only find very close matches, and changes in size or orientation of a car make it impossible to match with a template.

 

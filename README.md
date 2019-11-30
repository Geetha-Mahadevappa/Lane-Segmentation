# Lane-Segmentation
Opencv, Python, Pycharm

In this project, I used Python and OpenCV to find lane lines in the road images.
Following techniques are implemented
1. Color Selection:
  -> Convert RGB image into HSL color space
  -> Applied filter to select white and yellow lines
2. Canny Edge Detection
  -> Convert images into gray scale
  -> Gaussian Smoothing applyeid to smooth out rough lines
  -> Edge detector applied to detect edges in image
3. Region of Interest Selection
  -> selecting interested area where we find lane lines
4. Hough Transform Line Detection
  -> Detects all lines in edge image
  -> Averaging and Extrapolating multiple detected Lines
  -> 

 



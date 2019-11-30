# import
import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse


# select white and yellow lines in converted image (RGB to HSL)
def select_white_yellow(converted_img):
    # if white line exists in image: white color mask
    white_line_mask = cv2.inRange(converted_img, np.uint8([0, 200, 0]), np.uint8([255, 255, 255]))
    # if yellow line exists in image: yellow color mask
    yellow_line_mask = cv2.inRange(converted_img, np.uint8([10, 0, 100]), np.uint8([40, 255, 255]))
    # combine the mask
    mask = cv2.bitwise_or(white_line_mask, yellow_line_mask)
    return cv2.bitwise_and(image, image, mask=mask)

def select_ROI(image):
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    selected_region = region_filter(image, vertices)
    return selected_region

# filter region other than lines in ROI
def region_filter(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        # if input image has a channel dimension
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])
    return cv2.bitwise_and(image, mask)

# draw lines on actutal image
def draw_lines(image, lines, color=[0, 0, 255], thickness=3, copy=True):
    if copy:
        # make copy of original image
        image = np.copy(image)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

# veraged multiple lines detected for a lane line
def avg_slope_intercept(lines):
    # list all lines lays in left and reíghts side of image and their weights
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    if len(left_weights) > 0:
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights)
    else:
        left_lane = None
    if len(right_weights) > 0:
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights)
    else:
        right_lane = None
    # return slope and intercept of lines
    return left_lane, right_lane

#   Convert a line represented in slope and intercept into pixel points

def line_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    # convert every points to integer to draw lane line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def gt_lane_lines(image, lines):
    left_lane, right_lane = avg_slope_intercept(lines)

    # bottom of the image
    y1 = image.shape[0]
    # slightly lower than the middle
    y2 = y1 * 0.6

    # get all points of lift and right lines
    left_line = line_points(y1, y2, left_lane)
    right_line = line_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[0, 0, 255], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    # image and line_image must be the same shape
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

# display images
def show_image(image, name, show=True):
    if show:
        plt.figure(figsize=(5, 5))
    # use gray scale color map if there is only one channel
    if len(image.shape) == 2:
        cmap = 'gray'
    else:
        cmap = None
    plt.imshow(image, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()
    plt.savefig(name + ".jpg")

# main function
#
if __name__ == '__main__':

    print("start script {}".format(__file__))
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # required: source image that should be annotated
    ap.add_argument("-src", "--src", required=True, help="path to images")
    # parse arguments
    args = ap.parse_args()

    # required argument
    image_path = args.src

    # change debug to True, to show and save image after each operation
    debug = False

    # get all images for testing from folder
    for root, dir, files in (os.walk(image_path)):
        for file in files:
            # check whether file contains ".jpg"
            if file.endswith(".jpg"):
                file = os.path.join(root, file)
                # read image
                image = cv2.imread(file)
                # convert image to HLS colorspace to all detect lines
                converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                if debug:
                    show_image(converted_image, name="rgb2hls")

                # select yellow and white lines in image
                line_selected_img = select_white_yellow(converted_image)
                if debug:
                    show_image(line_selected_img, name="line_selected")

                # convert image to gray scale
                gray_image = cv2.cvtColor(line_selected_img, cv2.COLOR_RGB2GRAY)
                if debug:
                    show_image(gray_image, name="gray_image")

                # apply gaussian smoothing to make edges smoother, kernel size = (15, 15)
                blur_image = cv2.GaussianBlur(gray_image, (15, 15), 0)
                if debug:
                    show_image(blur_image, name="blur_image")

                # apply canny edge detector, low_threshold=50 high_threshold=200
                edge_image = cv2.Canny(blur_image, 50, 150)
                if debug:
                    show_image(edge_image, name="edge_image")

                # select region of interest
                selected_region = select_ROI(edge_image)
                if debug:
                    show_image(selected_region, name="selected_region")

                # get all lines
                lines = cv2.HoughLinesP(selected_region, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
                # draw line from list of lines obtained by hough transform
                line_img = draw_lines(image, lines, color=[0, 0, 255], thickness=2, copy=True)
                if debug:
                    show_image(line_img, name="line_img")

                # draw lane line, i,e. extend the detected lines
                laned_images = draw_lane_lines(image, gt_lane_lines(image, lines))
                show_image(laned_images, name="laned_images")

    # done
    print("finished script {}".format(__file__))
    sys.exit(0)

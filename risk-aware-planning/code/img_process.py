import numpy as np
import cv2

##################################### Image Perspective Warp ##############################################
def perspective_warp(img, points, map_w, map_h):
    # These 4 points are used to perspective correct the image
    # they represent the 4 corners of the map
    # used code from here as reference: https://stackoverflow.com/questions/22656698
    src = np.float32(points)
    dest = np.float32([[0, 0], [0, map_h], [map_w, map_h], [map_w, 0]])

    # Use those corners to create the transformation matrix
    # then apply that matrix to the image
    transmtx = cv2.getPerspectiveTransform(src, dest)
    wpcc_img = cv2.warpPerspective(img, transmtx, (map_w, map_h)) # processed

    # Show the perspective corrected image
    # plt.imshow(cv2.cvtColor(wpcc_img, cv2.COLOR_BGR2RGB)); plt.show()

    return wpcc_img


##################################### HSV Channel Segmentation ##############################################
def color_segment_image(img):
    # Split the image into the seperate HSV vhannels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_channel, sat_channel, _ = cv2.split(img)

    # plt.imshow(hue_channel); plt.show()
    # plt.imshow(sat_channel); plt.show()

    # Extract the bright colors from the image
    # the useful values are in the Hue and Saturation channel
    # hue will allow us to pinpoint the color shade and the saturation channel will allow us
    # to choose the bright colored spots (remove the gray areas and keep the bright colored areas)

    # EXAMPLE: for the red channel, the hue values are around zero (the hue channel loops back around to 180)
    # so for the hue channel we get the values in range of 175 and 180 and the values in range of 0 to 5
    # this will create a mask of all values that are 175-180 and 0-5. This mask will also include gray colors
    # that are slightly red, we then use the saturation channel to remove the slightly red gray values and only
    # keep the deep red colors
    # red = (hue_red_low or hue_red_high) and sat_high
    # the and distributes and creates 
    # red = (hue_red_low and sat_high) or (hue_red_high and sat_high)
    # This same can be used for all the colors that we want to extract
    red_low_channel = cv2.bitwise_and(cv2.inRange(hue_channel, 0, 5), cv2.inRange(sat_channel, 100, 255))
    red_high_channel = cv2.bitwise_and(cv2.inRange(hue_channel, 175, 180), cv2.inRange(sat_channel, 100, 255))
    red_channel = cv2.bitwise_or(red_low_channel, red_high_channel)
    green_channel = cv2.bitwise_and(cv2.inRange(hue_channel, 40, 50), cv2.inRange(sat_channel, 100, 255))
    blue_channel = cv2.bitwise_and(cv2.inRange(hue_channel, 100, 110), cv2.inRange(sat_channel, 100, 255))
    yellow_channel = cv2.bitwise_and(cv2.inRange(hue_channel, 20, 30), cv2.inRange(sat_channel, 100, 255))

    # plt.imshow(red_channel, cmap='gray'); plt.show()
    # plt.imshow(green_channel, cmap='gray'); plt.show()
    # plt.imshow(blue_channel, cmap='gray'); plt.show()
    # plt.imshow(yellow_channel, cmap='gray'); plt.show()

    return (red_channel, green_channel, blue_channel, yellow_channel)
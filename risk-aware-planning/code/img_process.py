import numpy as np
import cv2
import math

import cell_process
import matplotlib.pyplot as plt


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

##################################### Final Image Construction ##############################################
def merge_colors(red_channel, green_channel, blue_channel, yellow_channel):

    # We want to convert the different color channels into an RGB image and since yellow is Red and Green
    # we want add the yellow channel into the red and green channels 
    red_channel = cv2.bitwise_or(red_channel, yellow_channel)
    green_channel = cv2.bitwise_or(green_channel, yellow_channel)

    # red_channel = cv2.bitwise_or(red_channel, purple_channel)
    # blue_channel = cv2.bitwise_or(blue_channel, purple_channel)

    # plt.imshow(red_channel, cmap='gray'); plt.show()
    # plt.imshow(green_channel, cmap='gray'); plt.show()

    # Merge the channels into one RGB image
    processed_img = cv2.merge([red_channel,green_channel,blue_channel])
    # plt.imshow(processed_img); plt.show()

    return processed_img


def apply_edge_blur(img, reward_size):
    map_h, map_w = img.shape
    # goal_reward_image = cv2.bitwise_or(goal_reward_image, orig_goal_reward_image)

    # To do a guassian distribution around the edges, we first dialate the mask the same amount
    # as much as we want to do the gaussian blur 
    reward_size = 128
    dilate_kernel = np.ones((reward_size,reward_size), np.uint8)
    gaussian_kernel_size = reward_size + 1

    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    new_img = np.zeros((map_h, map_w, 1), dtype = "uint8")

    num_cnt = len(contours)
    for cnts in contours:
        mask = np.zeros((map_h, map_w, 1), dtype = "uint8")
        # (x,y),radius = cv2.minEnclosingCircle(cnts)
        # center = (int(x),int(y))
        # radius = int((reward_size/2) + math.sqrt(2) * radius)
        # print(radius)
        # cv2.circle(mask,center,radius,(255),-1)
        mask = cv2.drawContours(mask, [cnts], -1, (255), -1)
        mask = cv2.dilate(mask, dilate_kernel, 0)
        # plt.imshow(mask, cmap='gray'); plt.show()
        # mask = cv2.bilateralFilter(mask, reward_size*16, 1000, 1000)
        # mask = cv2.blur(mask, (gaussian_kernel_size, gaussian_kernel_size))
        mask = cv2.GaussianBlur(mask, (gaussian_kernel_size, gaussian_kernel_size), reward_size)
        new_img = cv2.scaleAdd(mask, (1/num_cnt), new_img)
        # plt.imshow(new_img, cmap='gray'); plt.show()
    
    return cv2.normalize(new_img, None, 255, 0, norm_type = cv2.NORM_MINMAX)


def create_risk_img(img, risk_size):
    # Wall risk image
    dilate_kernel = np.ones((risk_size,risk_size), np.uint8)
    gaussian_kernel_size = risk_size + 1
    wall_risk_image = cv2.dilate(img, dilate_kernel, 0)
    wall_risk_image = cv2.GaussianBlur(wall_risk_image, (gaussian_kernel_size, gaussian_kernel_size), 0)
    wall_risk_image = cv2.bitwise_or(wall_risk_image, img)
    # plt.imshow(wall_risk_image, cmap='gray'); plt.show()

    risk_image = wall_risk_image
    # purple_channel = risk_image

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # xx, yy = np.mgrid[0:risk_image.shape[0], 0:risk_image.shape[1]]
    # ax.plot_surface(xx, yy, risk_image, rstride=1, cstride=1, linewidth=0)
    # plt.show()

    return risk_image

def get_reward_images(cell_type, img, cell_size):
    # plt.imshow(img); plt.show()
    map_h, map_w = img.shape
    reward_graphs = {}
    types = cell_process.get_cell_types(cell_type)
    for goal in types:
        empty_image = np.zeros((map_h, map_w, 1), dtype = "uint8")
        for col_num in range(len(cell_type)):
            for row_num in range(len(cell_type[col_num])):
                if goal == cell_type[col_num][row_num]:
                    
                    for px_y in range(col_num * cell_size, (col_num+1) * cell_size):
                        for px_x in range(row_num * cell_size, (row_num+1) * cell_size):
                            # print(len(cell_type), col_num, px_y, (col_num * cell_size), (col_num * (cell_size+1) - 1))
                            # print(len(cell_type[col_num]), row_num, px_x, (row_num * cell_size), (row_num * (cell_size+1) - 1))
                            if img[px_y][px_x] > 250:
                                empty_image[px_y][px_x] = 254

        reward_graphs[goal] = empty_image
        print(goal)
        plt.imshow(empty_image); plt.show()

    return reward_graphs
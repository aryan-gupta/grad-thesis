import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

import cell
import global_vars as gv

# read in a image
def read_image(filename, show=False):
    img = cv2.imread(filename)
    if show: plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.show()
    return img


# perspective warp an image
def perspective_warp(img, points, map_w, map_h, show=False):
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
    if show: plt.imshow(cv2.cvtColor(wpcc_img, cv2.COLOR_BGR2RGB)); plt.show()

    return wpcc_img


# split the image into the color segments
# this is more complicated than just RGB channel splitting. The
# function uses HSV to find the hue and sat values that are in
# the appropriate bounds
def color_segment_image(img, show=False):
    # Split the image into the seperate HSV vhannels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_channel, sat_channel, _ = cv2.split(img)

    if show: plt.imshow(hue_channel); plt.show()
    if show: plt.imshow(sat_channel); plt.show()

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

    if show: plt.imshow(red_channel, cmap='gray'); plt.show()
    if show: plt.imshow(green_channel, cmap='gray'); plt.show()
    if show: plt.imshow(blue_channel, cmap='gray'); plt.show()
    if show: plt.imshow(yellow_channel, cmap='gray'); plt.show()

    return (red_channel, green_channel, blue_channel, yellow_channel)


# merge back the colors
# essentially does the opposite of color_segment_image. However,
# since color_segment_image is a lossy algorithm, running it then
# running merge_colors will result in a different but clean image
def merge_colors(red_channel, green_channel, blue_channel, yellow_channel, show=False):

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
    if show: plt.imshow(processed_img); plt.show()

    return processed_img


# applys the edge gaussian blur to a risk image
# @TODO move this to the risk module
def apply_edge_blur(img, reward_size, show=False):
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

    img = cv2.normalize(new_img, None, 255, 0, norm_type = cv2.NORM_MINMAX)
    if show: plt.imshow(img, cmap='gray'); plt.show()
    return img


# applys the edge gaussian blur to a risk image
# @TODO move this to the risk module
def create_risk_img(img, risk_size, show=False):
    # Wall risk image
    dilate_factor = 1
    dilate_kernel = np.ones((risk_size//dilate_factor,risk_size//dilate_factor), np.uint8)
    gaussian_kernel_size = risk_size + 1
    wall_risk_image = cv2.dilate(img, dilate_kernel, 0)
    wall_risk_image = cv2.GaussianBlur(wall_risk_image, (gaussian_kernel_size, gaussian_kernel_size), 0)
    wall_risk_image = cv2.bitwise_or(wall_risk_image, img)
    if show: plt.imshow(wall_risk_image, cmap='gray'); plt.show()

    risk_image = wall_risk_image

    risk_image = cv2.normalize(risk_image, None, 254, 0, norm_type=cv2.NORM_MINMAX)
    # purple_channel = risk_image

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # xx, yy = np.mgrid[0:risk_image.shape[0], 0:risk_image.shape[1]]
    # ax.plot_surface(xx, yy, risk_image, rstride=1, cstride=1, linewidth=0)
    # plt.show()

    return risk_image


# creates all the reward images for each axiom
# @TODO move to task/mission class
def get_reward_images(cell_type, img, CELLS_SIZE, show=False):
    if show: plt.imshow(img); plt.show()

    reward_graphs = {}
    reward_locations = {}

    # we are searching the cell_type for all the goals and where they are
    # for each goal we are searching
    # find the cell that its at
    # copy the pixels of that cell into the image
    # keep a log of the cells that have that goal
    for goal in cell.get_cell_types(cell_type):
        goal_locations = []
        empty_image = np.zeros((gv.map_h, gv.map_w, 1), dtype = "uint8")

        for col_num in range(len(cell_type)):
            for row_num in range(len(cell_type[col_num])):

                if goal == cell_type[col_num][row_num]:
                    goal_locations.append((row_num, col_num))

                    for px_y in range(col_num * CELLS_SIZE, (col_num+1) * CELLS_SIZE):
                        for px_x in range(row_num * CELLS_SIZE, (row_num+1) * CELLS_SIZE):
                            # print(len(cell_type), col_num, px_y, (col_num * CELLS_SIZE), (col_num * (CELLS_SIZE+1) - 1))
                            # print(len(cell_type[col_num]), row_num, px_x, (row_num * CELLS_SIZE), (row_num * (CELLS_SIZE+1) - 1))
                            if img[px_y][px_x] > 0:
                                empty_image[px_y][px_x] = 250

        reward_graphs[goal] = empty_image
        reward_locations[goal] = goal_locations

        if show: print(goal)
        if show: plt.imshow(empty_image); plt.show()
        if show:
            for loc in reward_locations[goal]:
                print(loc)
                print(cell_type[loc[0]][loc[1]])

    return reward_graphs, reward_locations


# copy pixels from src to dest cell by cell
# this implies the robot "seeing" the enviroment
def copy_pixels_risk(dest, src, current_phys_loc, cells_updated, CELLS_SIZE, VIEW_CELLS_SIZE, xop, yop):
    map_h = dest.shape[0]
    map_w = dest.shape[1]
    total_diff = 0

    # iterate over one quadrant of the range
    # xop and yop decide which quadrant
    for dy in range(VIEW_CELLS_SIZE):
        for dx in range(VIEW_CELLS_SIZE):
            # constrain the view cells size to a circle
            if math.sqrt(dx**2 + dy**2) > VIEW_CELLS_SIZE:
                continue

            cell_diff = 0

            # calculate out x and y values
            ycell = yop(current_phys_loc[1], dy)
            y = ycell * CELLS_SIZE
            if ycell > (map_h / CELLS_SIZE) or ycell < 0:
                continue

            xcell = xop(current_phys_loc[0], dx)
            x = xcell * CELLS_SIZE
            if xcell > (map_w / CELLS_SIZE) or xcell < 0:
                continue

            # iterate into the cell
            for u in range(y, y + CELLS_SIZE, 1):
                if u >= map_h:
                    break
                for v in range(x, x + CELLS_SIZE, 1):
                    if v >= map_w:
                        break

                    # update risk and calculate the difference
                    cell_diff += abs(int(dest[u,v]) - int(src[u,v]))
                    dest[u,v] = src[u,v]

            if cell_diff != 0:
                cells_updated.append((xcell, ycell))

            total_diff += cell_diff

    # return the total changes made to the risk in this cell
    return total_diff


# copy pixels from src to dest pixel by pixel. In one cell
# dest and src are rgb pixels
def copy_pixels_cells_img(cell_loc, dest, src, current_phys_loc, CELLS_SIZE):
    xcell, ycell = cell_loc
    y = ycell * CELLS_SIZE
    x = xcell * CELLS_SIZE
    map_h = dest.shape[0]
    map_w = dest.shape[1]

    for u in range(y, y + CELLS_SIZE, 1):
        if u >= map_h:
            break
        for v in range(x, x + CELLS_SIZE, 1):
            if v >= map_w:
                break

            dest[u,v] = src[u,v]


# update a local risk image using the global risk image
def update_local_risk_image(risk_image_local, raw_risk_image, current_phys_loc, CELLS_SIZE, VIEW_CELLS_SIZE, UPDATE_WEIGHT):
    total_diff = 0
    cells_updated = []
    add_lambda = lambda a,b: a+b
    sub_lambda = lambda a,b: a-b

    # +x+y
    total_diff += copy_pixels_risk(risk_image_local, raw_risk_image, current_phys_loc, cells_updated, CELLS_SIZE, VIEW_CELLS_SIZE, add_lambda, add_lambda)

    # -x+y
    total_diff += copy_pixels_risk(risk_image_local, raw_risk_image, current_phys_loc, cells_updated, CELLS_SIZE, VIEW_CELLS_SIZE, sub_lambda, add_lambda)

    # +x-y
    total_diff += copy_pixels_risk(risk_image_local, raw_risk_image, current_phys_loc, cells_updated, CELLS_SIZE, VIEW_CELLS_SIZE, add_lambda, sub_lambda)

    # -x-y
    total_diff += copy_pixels_risk(risk_image_local, raw_risk_image, current_phys_loc, cells_updated, CELLS_SIZE, VIEW_CELLS_SIZE, sub_lambda, sub_lambda)

    return risk_image_local, total_diff, cells_updated


# saves an image that only has one or two channel
# creates the other channels using a blank image the
# same size as the param channel
def save_channel_image(name, r=None, g=None, b=None):
    if r is None:
        r = np.zeros((gv.map_h, gv.map_w), np.uint8)

    if g is None:
        g = np.zeros((gv.map_h, gv.map_w), np.uint8)

    if b is None:
        b = np.zeros((gv.map_h, gv.map_w), np.uint8)

    img = cv2.merge([r, g, b])
    cv2.imwrite(name, img)

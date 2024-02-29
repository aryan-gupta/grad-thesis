import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

import img
import global_vars as gv

# I honestly thought there was a built in map function, but I guess Im wrong
# https://stackoverflow.com/questions/1969240
# This function returns a function that will map the image pixel values (in the
# range of 0 to 255) to another set of numbers in the range of 0.0 to 1.0
# it will make calculating Dj's algo alot easier
def make_interpolater(left_min, left_max, right_min, right_max):
    # Figure out how 'wide' each range is
    leftSpan = left_max - left_min
    rightSpan = right_max - right_min

    # Compute the scale factor between left and right values
    scaleFactor = float(rightSpan) / float(leftSpan)

    # create interpolation function using pre-calculated scaleFactor
    def interp_fn(value):
        return right_min + (value-left_min)*scaleFactor

    return interp_fn


# Creates the cells based off an risk/reward image based off of the CELLS_SIZE
# create cells on @r img_cells, extract cell_type s, and cell_cost s using
# ltl_target_image to __, risk_image to __, and img_cells to __
# we want to use the function name, input and output parameters. each input and output
def create_cells(ltl_target_image, risk_image, img_cells):
    # get image size
    map_h, map_w = ltl_target_image.shape

    # Variable to store the unique colors in the image
    colors = []

    # Go through each cell and each pixel of the cell to decide what type of cell it is
    # use this info to construct a cell type map of the area
    cell_type = []
    cell_cost = []
    xcell = 0
    ycell = 0
    max_cost = -1
    for y in range(0, map_h, gv.CELLS_SIZE):
        cell_type.append([])
        cell_cost.append([])
        xcell = 0

        for x in range(0, map_w, gv.CELLS_SIZE):
            cell_type[ycell].append(gv.EMPTY_CELL_CHAR)
            cell_cost[ycell].append(0.0)

            cell_known, cell_sum = update_a_cell((xcell, ycell), ltl_target_image, cell_type, cell_cost, img_cells, risk_image)

            # if we dont know the cell type, mark it as a clean cell
            # and add the weights to the cell
            if not cell_known:
                cost = handle_unknown_cell((x, y), (xcell, ycell), img_cells, cell_sum, cell_cost)
                # record the max cost for debugging purposes
                if cost > max_cost:
                    max_cost = cost

            xcell += 1
        ycell += 1

    if gv.DEBUG >= 2:
        # Print the different colors in the image
        print(colors)
        print()
        print()

        # Print the max cost
        print(max_cost)
        print()
        print()

        # Print the cell type map for debugging
        for y in cell_type:
            for ch in y:
                if ch =='Y': ch = 'C'
                if ch =='#': ch = ' '
                print(ch,' ', end="")
            print("")
        print()
        print()

        # Print the cell cost map for debugging
        for y in cell_cost:
            for cost in y:
                print("{:.2f}".format(cost), end=", ")
            print()
        print()
        print()

        # Show the images with the cell type and cell boundries
        plt.imshow(img_cells); plt.show()

    return img_cells, cell_type, cell_cost


# Get the start and finish locations of the enviroment using the cell types
def get_start_finish_locations(cell_type):
    # loops thorugh the 2D cell_type array to find the CELL_CHARS for start and finish
    start = ()
    finish = ()
    for y in range(len(cell_type)):
        for x in range(len(cell_type[0])):
            if cell_type[y][x] == gv.START_CELL_CHAR:
                start = (x, y)
            if cell_type[y][x] == gv.END_CELL_CHAR:
                finish = (x, y)
            if cell_type[y][x] == gv.LTL_TARGET_CELL_CHAR:
                finish = (x, y)

    # print(start)
    # print(finish)

    return start, finish


# Return all the possible cell types that are present in the enviroment
# Needs to be improved
def get_cell_types(cell_type):
    # types = []
    # for col_num in range(len(cell_type)):
    #     for row_num in range(len(cell_type[col_num])):
    #         if cell_type[col_num][row_num] not in types:
    #             # print(cell_type[col_num][row_num])
    #             types.append(cell_type[col_num][row_num])
    # https://stackoverflow.com/questions/16228248
    return list(gv.CHAR_COLOR_MAP.values())


# handles an unknown cell during the cell creation or cell updating period
def handle_unknown_cell(points, cell_loc, img_cells, cell_sum, cell_cost):
    x, y = points
    xcell, ycell = cell_loc

    # function to map pixel values to cost values
    image_value_to_cost_value = make_interpolater(0, 255, 0, 1.0)

    # draw rectangles
    cost = cell_sum / (gv.CELLS_SIZE**2)
    # print(xcell, "--", ycell)
    cell_cost[ycell][xcell] = image_value_to_cost_value(cost)
    # print(f"{x}-{y} :: {cost}")

    # if cell_cost[ycell][xcell] < 0.9999999:
    #     cell_cost[ycell][xcell] = 0.9999999

    if cost == 0:
        img_cells = cv2.rectangle(img_cells, (x,y), (x + gv.CELLS_SIZE - 1,y + gv.CELLS_SIZE - 1), (50,50,50), 1)
    else:
        img_cells = cv2.rectangle(img_cells, (x,y), (x + gv.CELLS_SIZE - 1,y + gv.CELLS_SIZE - 1), (cost,0,cost), 1)

    return cost


# creates or updates a cell
# if the variables input is being created, its creating a cell (IE full replan needed)
# if the variables input are already created, its overwriting the values/updating the cell
def update_a_cell(cell_loc, ltl_target_image, cell_type, cell_cost, img_cells, risk_image):
    xcell, ycell = cell_loc
    y = ycell * gv.CELLS_SIZE
    x = xcell * gv.CELLS_SIZE

    # get image size
    assert ltl_target_image.shape == risk_image.shape, f"{ltl_target_image.shape} == {risk_image.shape}"
    map_h, map_w = ltl_target_image.shape

    # determine what the cell is
    cell_known = False
    cell_sum = 0
    cell_pxl_count = 0
    for u in range(y, y + gv.CELLS_SIZE, 1):
        if u >= map_h:
            break
        for v in range(x, x + gv.CELLS_SIZE, 1):
            if v >= map_w:
                break

            # If the cell is a Goal Cell, give it 0.0 weight
            # If the cell is a Objective Cell, give it a 0.0 weight
            # if the cell is a Hazard Cell, give it the same weight as the average value of the cell

            cell_sum += risk_image[u,v]

            # mark the cells if its corosponding color exists in the cell
            # these values are dont care values
            r = ltl_target_image[u,v]
            g = risk_image[u,v]
            b = None


            # Traceback (most recent call last):
            # File "/life/1/grad/thesis/thesis/fast-risk-aware-ltl-planning/code/main.py", line 110, in <module>
            #     main()
            # File "/life/1/grad/thesis/thesis/fast-risk-aware-ltl-planning/code/main.py", line 107, in main
            #     e.create_final_image(final_image, p.get_filled_assumed_risk(), p.get_total_shortest_path())
            # File "/life/1/grad/thesis/thesis/fast-risk-aware-ltl-planning/code/env.py", line 220, in create_final_image
            #     dj_path_image, _, _ = cell.create_cells(dj_path_image, assumed_risk_image_filled, CELLS_SIZE, show=False)
            # File "/life/1/grad/thesis/thesis/fast-risk-aware-ltl-planning/code/cell.py", line 59, in create_cells
            #     cell_known, cell_sum = update_a_cell((xcell, ycell), processed_img, cell_type, cell_cost, img_cells, risk_image, CELLS_SIZE)
            # File "/life/1/grad/thesis/thesis/fast-risk-aware-ltl-planning/code/cell.py", line 202, in update_a_cell
            #     assert g == risk_image[u,v]
            # AssertionError
            #
            # For some reason
            #
            # try:
            #     assert g == risk_image[u,v]
            # except AssertionError:
            #     print(g)
            #     print(risk_image[u,v])
            #     plt.imshow(processed_img); plt.show()
            #     plt.imshow(risk_image); plt.show()

            if g == 255: # Hazard Cells
                cell_known = True
                img_cells = cv2.rectangle(img_cells, (x,y), (x + gv.CELLS_SIZE - 1,y + gv.CELLS_SIZE - 1), (0,255,0), 1)
                cell_type[ycell][xcell] = gv.HAZARD_CELL_CHAR
                break
            if r == 250: # LTL Current Target
                cell_known = True
                img_cells = cv2.rectangle(img_cells, (x,y), (x + gv.CELLS_SIZE - 1,y + gv.CELLS_SIZE - 1), (250,0,0), 1)
                cell_type[ycell][xcell] = gv.LTL_TARGET_CELL_CHAR
                break
            if r == 225: # Mission Start Cell
                cell_known = True
                img_cells = cv2.rectangle(img_cells, (x,y), (x + gv.CELLS_SIZE - 1,y + gv.CELLS_SIZE - 1), (225,0,0), 1)
                cell_type[ycell][xcell] = gv.START_CELL_CHAR
                break
            if r == 200: # Mission Finish Cell
                cell_known = True
                img_cells = cv2.rectangle(img_cells, (x,y), (x + gv.CELLS_SIZE - 1,y + gv.CELLS_SIZE - 1), (200,0,0), 1)
                cell_type[ycell][xcell] = gv.END_CELL_CHAR
                break

            for color in gv.CHAR_COLOR_MAP.keys():
                if r == color: # K Target
                    cell_known = True
                    img_cells = cv2.rectangle(img_cells, (x,y), (x + gv.CELLS_SIZE - 1,y + gv.CELLS_SIZE - 1), (color,0,0), 1)
                    cell_type[ycell][xcell] = gv.CHAR_COLOR_MAP[color]
                    break

            if cell_known:
                break

        # Exit loop if we know the cell type, if its a hazard cell mark it as 1.0 cost
        if cell_known:
            if cell_type[ycell][xcell] == gv.HAZARD_CELL_CHAR:
                cell_cost[ycell][xcell] = float("inf")
            else:
                cell_cost[ycell][xcell] = 0.0
            break

    return cell_known, cell_sum


# Updates the cells surrounding the current location, speeds up calc
# so we dont have to call create_cells everytime
def update_cells(cells_updated, ltl_target_image, risk_reward_cell_type, risk_reward_cell_cost, risk_reward_img_cells, current_phys_loc, assumed_risk_image_filled):
    # get image size
    map_h, map_w, _ = risk_reward_img_cells.shape

    # create copy of the processed image so we cn draw the cells on it
    img_cells = risk_reward_img_cells.copy()

    # go through each updated cell and each pixel of the cell to decide what type of cell it is
    # use this info to update the cell type map of the area
    max_cost = -1
    for (xcell, ycell) in cells_updated:
        x = xcell * gv.CELLS_SIZE
        y = ycell * gv.CELLS_SIZE

        cell_known, cell_sum = update_a_cell((xcell, ycell), ltl_target_image, risk_reward_cell_type, risk_reward_cell_cost, img_cells, assumed_risk_image_filled)

        # if we dont know the cell type, mark it as a clean cell
        # and add the weights to the cell
        if not cell_known: handle_unknown_cell((x, y), (xcell, ycell), img_cells, cell_sum, risk_reward_cell_cost)

        # only copy the cells that have changed
        img.copy_pixels_cells_img((xcell, ycell), risk_reward_img_cells, img_cells, current_phys_loc, gv.CELLS_SIZE)

    return risk_reward_img_cells, risk_reward_cell_type, risk_reward_cell_cost

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

import img_process

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
def create_cells(processed_img, risk_image, CELLS_SIZE, show=False):
    # get image size
    map_h, map_w, _ = processed_img.shape

    # create copy of the processed image so we cn draw the cells on it
    img_cells = processed_img.copy()

    # Variable to store the unique colors in the image
    colors = []

    # Go through each cell and each pixel of the cell to decide what type of cell it is
    # use this info to construct a cell type map of the area
    cell_type = []
    cell_cost = []
    xcell = 0
    ycell = 0
    max_cost = -1
    for y in range(0, map_h, CELLS_SIZE):
        cell_type.append([])
        cell_cost.append([])
        xcell = 0

        for x in range(0, map_w, CELLS_SIZE):
            cell_type[ycell].append('C')
            cell_cost[ycell].append(0.0)

            cell_known, cell_sum = update_a_cell((xcell, ycell), processed_img, cell_type, cell_cost, img_cells, risk_image, CELLS_SIZE)

            # if we dont know the cell type, mark it as a clean cell
            # and add the weights to the cell
            if not cell_known:
                cost = handle_unknown_cell((x, y), (xcell, ycell), img_cells, cell_sum, cell_cost, CELLS_SIZE)
                # record the max cost for debugging purposes
                if cost > max_cost:
                    max_cost = cost

            xcell += 1
        ycell += 1

    if show:
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
            print(y)
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


# Convert connected cells with the \p orig_value to \p new_value
# this allows us to mark areas from Goals to Start and Finish Cells
def convert_cells_recurse(cell_type, y, x, orig_value, new_value):
    cell_type[y][x] = new_value
    if y > 0 and cell_type[y - 1][x] == orig_value:
        convert_cells_recurse(cell_type, y - 1, x, orig_value, new_value)
    if x > 0 and cell_type[y][x - 1] == orig_value:
        convert_cells_recurse(cell_type, y, x - 1, orig_value, new_value)
    if y < (len(cell_type)-1) and cell_type[y + 1][x] == orig_value:
        convert_cells_recurse(cell_type, y + 1, x, orig_value, new_value)
    if x < (len(cell_type[y])-1) and cell_type[y][x + 1] == orig_value:
        convert_cells_recurse(cell_type, y, x + 1, orig_value, new_value)


# Converts objective and goal cells into the specific objective and goals
def convert_cells(cell_type, objectives, goals):
    objectives_idx = 0
    goals_idx = 0
    # Convert Goal Cells into start and finish cells
    for y in range(len(cell_type)):
        for x in range(len(cell_type[y])):
            if cell_type[y][x] == "O":
                convert_cells_recurse(cell_type, y, x, "O", objectives[objectives_idx])
                objectives_idx += 1
            if cell_type[y][x] == "G":
                convert_cells_recurse(cell_type, y, x, "G", goals[goals_idx])
                goals_idx += 1

    # Print the converted cell types
    # for y in cell_type:
    #     print(y)
    # print()
    # print()

    return cell_type


# Get the start and finish locations of the enviroment using the cell types
def get_start_finish_locations(cell_type):
    # find the start node
    start = ()
    finish = ()
    for y in range(len(cell_type)):
        for x in range(len(cell_type[0])):
            if cell_type[y][x] == 'S':
                start = (x, y)
            if cell_type[y][x] == 'F':
                finish = (x, y)
            if cell_type[y][x] == 'T':
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
    types = ["A","B","F","S","R"]
    return types


# Convert the cell type map into a state diagram
# the algo pretty much checks the 4 sides (North, South, Eeast, West) to see
# if the block is a travelable block and creates a valid edge with weight of 1.0
# if it is
def cells_to_state_diagram(cell_type, cell_cost, show=False):
    state_diagram = []
    state_dict = {}
    for y in range(len(cell_type)):
        state_diagram.append([])
        for x in range(len(cell_type[0])):
            state_diagram[y].append([float("inf"), float("inf"), float("inf"), float("inf"), cell_type[y][x], f"{x}-{y}"])
            state_dict[f"{x}-{y}"] = []
            if cell_type[y][x] == 'H':
                continue
            # check up left
            # NOT IMPL

            # check up
            if y > 0:
                state_diagram[y][x][0] = cell_cost[y][x]
                state_dict[f"{x}-{y}"].append(('u', f"{x}-{y-1}"))
            # check up right
            # NOT IMPL

            # check left
            if x > 0:
                state_diagram[y][x][1] = cell_cost[y][x]
                state_dict[f"{x}-{y}"].append(('l', f"{x-1}-{y}"))

            # check right
            if x < (len(cell_type[0]) - 1):
                state_diagram[y][x][2] = cell_cost[y][x]
                state_dict[f"{x}-{y}"].append(('r', f"{x+1}-{y}"))

            # check down left
            # NOT IMPL

            # check down
            if y < (len(cell_type) - 1):
                state_diagram[y][x][3] = cell_cost[y][x]
                state_dict[f"{x}-{y}"].append(('d', f"{x}-{y+1}"))
            # check down right
            # NOT IMPL

    if show: cell_process.pretty_print_state_dd(state_diagram, state_dict)

    return state_diagram, state_dict


# Pretty prints the state diagram and the state dictionary
def pretty_print_state_dd(state_diagram, state_dict):
    # pretty print state diagram
    for row in state_diagram:
        # Up arrows
        for col in row:
            print(" ", end="") # space for the left arrow
            weight_single = 9 if col[0] >= 1.0 else int(col[0] * 10)
            print(weight_single, end="")
            print(" ", end="") # space for the right arrow
        print()
        # left/right and center char
        for col in row:
            weight_single = 9 if col[1] >= 1.0 else int(col[1] * 10)
            print(weight_single, end="")
            print(col[4], end="")
            weight_single = 9 if col[2] >= 1.0 else int(col[2] * 10)
            print(weight_single, end="")
        print()
        # Down arrows
        for col in row:
            print(" ", end="") # space for the left arrow
            weight_single = 9 if col[3] >= 1.0 else int(col[3] * 10)
            print(weight_single, end="")
            print(" ", end="") # space for the right arrow
        print()

    print(state_dict)


# handles an unknown cell during the cell creation or cell updating period
def handle_unknown_cell(points, cell_loc, img_cells, cell_sum, cell_cost, CELLS_SIZE):
    x, y = points
    xcell, ycell = cell_loc

    # function to map pixel values to cost values
    image_value_to_cost_value = make_interpolater(0, 255, 0, 1.0)

    # draw rectangles
    cost = cell_sum / (CELLS_SIZE**2)
    # print(xcell, "--", ycell)
    cell_cost[ycell][xcell] = image_value_to_cost_value(cost)
    # print(f"{x}-{y} :: {cost}")

    # if cell_cost[ycell][xcell] < 0.9999999:
    #     cell_cost[ycell][xcell] = 0.9999999

    if cost == 0:
        img_cells = cv2.rectangle(img_cells, (x,y), (x + CELLS_SIZE - 1,y + CELLS_SIZE - 1), (50,50,50), 1)
    else:
        img_cells = cv2.rectangle(img_cells, (x,y), (x + CELLS_SIZE - 1,y + CELLS_SIZE - 1), (cost,0,cost), 1)

    return cost


# creates or updates a cell
# if the variables input is being created, its creating a cell (IE full replan needed)
# if the variables input are already created, its overwriting the values/updating the cell
def update_a_cell(cell_loc, processed_img, cell_type, cell_cost, img_cells, risk_image, CELLS_SIZE):
    xcell, ycell = cell_loc
    y = ycell * CELLS_SIZE
    x = xcell * CELLS_SIZE

    # get image size
    map_h, map_w, _ = processed_img.shape

    # determine what the cell is
    cell_known = False
    cell_sum = 0
    cell_pxl_count = 0
    for u in range(y, y + CELLS_SIZE, 1):
        if u >= map_h:
            break
        for v in range(x, x + CELLS_SIZE, 1):
            if v >= map_w:
                break

            # If the cell is a Goal Cell, give it 0.0 weight
            # If the cell is a Objective Cell, give it a 0.0 weight
            # if the cell is a Hazard Cell, give it the same weight as the average value of the cell

            cell_sum += risk_image[u,v]

            # mark the cells if its corosponding color exists in the cell
            # these values are dont care values
            r = tuple(processed_img[u,v])[0]
            g = tuple(processed_img[u,v])[1]
            b = tuple(processed_img[u,v])[2]

            if tuple(processed_img[u,v]) == (0,255,0): # Hazard Cells
                cell_known = True
                img_cells = cv2.rectangle(img_cells, (x,y), (x + CELLS_SIZE - 1,y + CELLS_SIZE - 1), (0,255,0), 1)
                cell_type[ycell][xcell] = 'H'
                break
            if tuple(processed_img[u,v]) == (255, 0, 0): # Goal Cells
                cell_known = True
                img_cells = cv2.rectangle(img_cells, (x,y), (x + CELLS_SIZE - 1,y + CELLS_SIZE - 1), (255,0,0), 1)
                cell_type[ycell][xcell] = 'G'
                break
            if tuple(processed_img[u,v]) == (255, 255, 0): # Objective Cells
                cell_known = True
                img_cells = cv2.rectangle(img_cells, (x,y), (x + CELLS_SIZE - 1,y + CELLS_SIZE - 1), (255,255,0), 1)
                cell_type[ycell][xcell] = 'O'
                break
            if tuple(processed_img[u,v]) == (0, 0, 255): # Refuel Cells
                cell_known = True
                img_cells = cv2.rectangle(img_cells, (x,y), (x + CELLS_SIZE - 1,y + CELLS_SIZE - 1), (0,0,255), 1)
                cell_type[ycell][xcell] = 'R'
                break
            if tuple(processed_img[u,v]) == (254, 0, 254): # LTL Current Target
                cell_known = True
                img_cells = cv2.rectangle(img_cells, (x,y), (x + CELLS_SIZE - 1,y + CELLS_SIZE - 1), (255,0,255), 1)
                cell_type[ycell][xcell] = 'T'
                break

        # Exit loop if we know the cell type, if its a hazard cell mark it as 1.0 cost
        if cell_known:
            if cell_type[ycell][xcell] == 'H':
                cell_cost[ycell][xcell] = float("inf")
            else:
                cell_cost[ycell][xcell] = 0.0
            break

    return cell_known, cell_sum


# Updates the cells surrounding the current location, speeds up calc
# so we dont have to call create_cells everytime
def update_cells(cells_updated, risk_reward_image, risk_reward_cell_type, risk_reward_cell_cost, risk_reward_img_cells, current_phys_loc, assumed_risk_image_filled, CELLS_SIZE, VIEW_CELLS_SIZE):
    # get image size
    map_h, map_w, _ = risk_reward_image.shape

    # create copy of the processed image so we cn draw the cells on it
    img_cells = risk_reward_image.copy()

    # go through each updated cell and each pixel of the cell to decide what type of cell it is
    # use this info to update the cell type map of the area
    max_cost = -1
    for (xcell, ycell) in cells_updated:
        x = xcell * CELLS_SIZE
        y = ycell * CELLS_SIZE

        cell_known, cell_sum = update_a_cell((xcell, ycell), risk_reward_image, risk_reward_cell_type, risk_reward_cell_cost, img_cells, assumed_risk_image_filled, CELLS_SIZE)

        # if we dont know the cell type, mark it as a clean cell
        # and add the weights to the cell
        if not cell_known: handle_unknown_cell((x, y), (xcell, ycell), img_cells, cell_sum, risk_reward_cell_cost, CELLS_SIZE)

        # only copy the cells that have changed
        img_process.copy_pixels_cells_img((xcell, ycell), risk_reward_img_cells, img_cells, current_phys_loc, CELLS_SIZE)

    return risk_reward_img_cells, risk_reward_cell_type, risk_reward_cell_cost

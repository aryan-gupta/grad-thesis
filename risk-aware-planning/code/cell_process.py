import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


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


# Creates the cells based off an risk/reward image based off of the cell_size
def create_cells(processed_img, risk_image, cell_size, show=False):
    # get image size
    map_h, map_w, _ = processed_img.shape

    # create copy of the processed image so we cn draw the cells on it
    img_cells = processed_img.copy()

    # function to map pixel values to cost values
    image_value_to_cost_value = make_interpolater(0, 255, 0, 1.0)

    # Variable to store the unique colors in the image
    colors = []

    # Go through each cell and each pixel of the cell to decide what type of cell it is
    # use this info to construct a cell type map of the area
    cell_type = []
    cell_cost = []
    cell_num_height = 0
    cell_num_width = 0
    max_cost = -1
    for y in range(0, map_h, cell_size):
        cell_type.append([])
        cell_cost.append([])
        cell_num_height = 0

        for x in range(0, map_w, cell_size):
            cell_type[cell_num_width].append('C')
            cell_cost[cell_num_width].append(0.0)

            # determine what the cell is
            cell_known = False
            cell_sum = 0
            cell_pxl_count = 0
            for u in range(y, y + cell_size, 1):
                if u >= map_h:
                    break

                for v in range(x, x + cell_size, 1):
                    if v >= map_w:
                        break

                    # If the cell is a Goal Cell, give it 0.0 weight
                    # If the cell is a Objective Cell, give it a 0.0 weight
                    # if the cell is a Hazard Cell, give it the same weight as the average value of the cell

                    cell_sum += risk_image[u,v]

                    # keep a record of all the different colors
                    if tuple(processed_img[u,v]) not in colors:
                        colors.append(tuple(processed_img[u,v]))

                    # mark the cells if its corosponding color exists in the cell
                    if tuple(processed_img[u,v]) == (0,255,0): # Hazard Cells
                        cell_known = True
                        img_cells = cv2.rectangle(img_cells, (x+1,y+1), (x + cell_size,y + cell_size), (0,255,0), 1)
                        cell_type[cell_num_width][cell_num_height] = 'H'
                    if tuple(processed_img[u,v]) == (255, 0, 0): # Goal Cells
                        cell_known = True
                        img_cells = cv2.rectangle(img_cells, (x+1,y+1), (x + cell_size,y + cell_size), (255,0,0), 1)
                        cell_type[cell_num_width][cell_num_height] = 'G'
                    if tuple(processed_img[u,v]) == (255, 255, 0): # Objective Cells
                        cell_known = True
                        img_cells = cv2.rectangle(img_cells, (x+1,y+1), (x + cell_size,y + cell_size), (255,255,0), 1)
                        cell_type[cell_num_width][cell_num_height] = 'O'
                    if tuple(processed_img[u,v]) == (0, 0, 255): # Refuel Cells
                        cell_known = True
                        img_cells = cv2.rectangle(img_cells, (x+1,y+1), (x + cell_size,y + cell_size), (0,0,255), 1)
                        cell_type[cell_num_width][cell_num_height] = 'R'
                    if tuple(processed_img[u,v]) == (254, 0, 254): # LTL Current Target
                        cell_known = True
                        img_cells = cv2.rectangle(img_cells, (x+1,y+1), (x + cell_size,y + cell_size), (255,0,255), 1)
                        cell_type[cell_num_width][cell_num_height] = 'T'


                # Exit loop if we know the cell type, if its a hazard cell mark it as 1.0 cost
                if cell_known:
                    if cell_type[cell_num_width][cell_num_height] == 'H':
                        cell_cost[cell_num_width][cell_num_height] = float("inf")
                    else:
                        cell_cost[cell_num_width][cell_num_height] = 0.0
                    break


            # if we dont know the cell type, mark it as a clean cell
            # and add the weights to the cell
            if not cell_known:
                # draw rectangles
                cost = cell_sum / (cell_size**2)
                cell_cost[cell_num_width][cell_num_height] = image_value_to_cost_value(cost)
                # record the max cost for debugging purposes
                if cost > max_cost:
                    max_cost = cost
                # print(f"{x}-{y} :: {cost}")

                # if cell_cost[cell_num_width][cell_num_height] < 0.9999999:
                #     cell_cost[cell_num_width][cell_num_height] = 0.9999999

                if cost == 0:
                    img_cells = cv2.rectangle(img_cells, (x+1,y+1), (x + cell_size,y + cell_size), (50,50,50), 1)
                else:
                    img_cells = cv2.rectangle(img_cells, (x+1,y+1), (x + cell_size,y + cell_size), (cost,0,cost), 1)


            cell_num_height += 1
        cell_num_width += 1

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

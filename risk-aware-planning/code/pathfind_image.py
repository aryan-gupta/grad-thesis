import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import bisect


import img_process
import cell_process
import ltl_process
import dijkstra

# GLOBAL VARS
CELLS_SIZE = 8 # 32 pixels
VIEW_CELLS_SIZE = 8

# final image dimensions (must be divisiable by CELL_SIZE)
map_h = 640
map_w = 576

# read image and show it
img = img_process.read_image('./sample.jpg', show=True)

# perspective warp the image so its a top down view
points = [[1025, 132], [855, 2702], [3337, 2722], [2974, 165]]
wpcc_img = img_process.perspective_warp(img, points, map_w, map_h, show=True)

# seperate the image into the color segments to create binary images of each color
(red_channel, green_channel, blue_channel, yellow_channel) = img_process.color_segment_image(wpcc_img, show=False)

# merge the color channels into one fully saturated and colorified image
processed_img = img_process.merge_colors(red_channel, green_channel, blue_channel, yellow_channel, show=True)

# create images of all the goals and objectives
orig_goal_reward_image = cv2.add(cv2.add(red_channel, blue_channel), yellow_channel)

# apply blur to edges of goals (deprecated and must be removed)
goal_reward_image = img_process.apply_edge_blur(orig_goal_reward_image, 128, show=False)

# create image of risk gradient
raw_risk_image = green_channel
risk_image = img_process.create_risk_img(green_channel, 64, show=False)

# create cells based off of map and risk and assign costs to cells
img_cells, cell_type, cell_cost = cell_process.create_cells(processed_img, risk_image, CELLS_SIZE); # plt.imshow(risk_reward_image); plt.show()

# convert cells to seperate objectives and goals (@TODO find a better way to do this)
cell_type = cell_process.convert_cells(cell_type, objectives=["A", "B"], goals=["S", "F"])

# get start and finish locations from cell graph
start, finish = cell_process.get_start_finish_locations(cell_type)

# get reward map for each objectives and goals
reward_graphs = img_process.get_reward_images(cell_type, orig_goal_reward_image, CELLS_SIZE, show=False)

# parse through LTL automata
ltl_state_diag, aps, start_state, final_state = ltl_process.parse_ltl_hoa("ltl.hoa.txt")
current_state_reward_graph = ltl_process.get_reward_img_state(ltl_state_diag, start_state, reward_graphs, (map_h, map_w))
reward_current = img_process.apply_edge_blur(current_state_reward_graph, 128)
# plt.imshow(reward_current, cmap="gray"); plt.show()

risk_reward_image = cv2.merge([current_state_reward_graph, risk_image, current_state_reward_graph])
plt.imshow(current_state_reward_graph, cmap="gray"); plt.show()
plt.imshow(risk_image, cmap="gray"); plt.show()
plt.imshow(risk_reward_image); plt.show()


# Convert risk_reward_image into cells
risk_reward_img_cells, risk_reward_cell_type, risk_reward_cell_cost = cell_process.create_cells(risk_reward_image, risk_image, CELLS_SIZE)
plt.imshow(risk_reward_img_cells); plt.show()
# Print the cell type map for debugging
for y in risk_reward_cell_type:
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

state_diagram, state_dict = cell_process.cells_to_state_diagram(risk_reward_cell_type, risk_reward_cell_cost, MAX_WEIGHT)
cell_process.pretty_print_state_dd(state_diagram, state_dict)
_, finish = cell_process.get_start_finish_locations(risk_reward_cell_type)

# state_diagram, state_dict = cell_process.cells_to_state_diagram(cell_type, cell_cost, MAX_WEIGHT)
# cell_process.pretty_print_state_dd(state_diagram, state_dict)

# exit()
##################################### State Diagram Conversion ##############################################

# start pathfinding
current_ltl_state = start_state
start_phys_loc = start
next_phys_loc = finish
dj_path_image = img_cells.copy()

while current_ltl_state != final_state:
    # == do a preliminary dj's algo
    # get reward map of current LTL state
    current_state_reward_graph = ltl_process.get_reward_img_state(ltl_state_diag, current_ltl_state, reward_graphs, (map_h, map_w))
    reward_current = img_process.apply_edge_blur(current_state_reward_graph, 128, show=False)

    # merge risk, reward into a single image map
    risk_reward_image = cv2.merge([current_state_reward_graph, risk_image, current_state_reward_graph])


    # Convert risk_reward_image into cells
    risk_reward_img_cells, risk_reward_cell_type, risk_reward_cell_cost = cell_process.create_cells(risk_reward_image, risk_image, CELLS_SIZE, show=False)

    # convert cells to state diagram so we can apply dj's algo to it
    state_diagram, state_dict = cell_process.cells_to_state_diagram(risk_reward_cell_type, risk_reward_cell_cost, show=False)

    # get start and finish locations for this ltl node
    _, next_phys_loc = cell_process.get_start_finish_locations(risk_reward_cell_type)

    # state_diagram, state_dict = cell_process.cells_to_state_diagram(cell_type, cell_cost)
    # cell_process.pretty_print_state_dd(state_diagram, state_dict)

    # apply dj's algo
    shortest_path = dijkstra.dj_algo(img_cells, cell_type, (start_phys_loc, next_phys_loc), state_diagram, CELLS_SIZE)
    _ = dijkstra.draw_shortest_path(shortest_path, risk_reward_img_cells, reward_graphs, (start_phys_loc, next_phys_loc), CELLS_SIZE)
    dijkstra.draw_path_global(shortest_path, dj_path_image, (start_phys_loc, next_phys_loc), CELLS_SIZE)


    # == do our mini dj's algo
    # create copy of the current risk map
    risk_image_local = risk_image.copy()
    current_phys_loc = start_phys_loc
    dj_path_idx = 0
    plt.imshow(dj_path_image); plt.savefig(f"/tmp/thesis/pic000orig.png")
    dj_path_image_adhoc = img_cells.copy()


    img_tmp_idx=0
    while current_phys_loc != next_phys_loc:
        # update risk map everytime we move
        risk_image_local = img_process.update_local_risk_image(risk_image_local, raw_risk_image, current_phys_loc, CELLS_SIZE, VIEW_CELLS_SIZE)

        # reapply DJ's algo using new start key
        risk_reward_image_local = cv2.merge([current_state_reward_graph, risk_image_local, current_state_reward_graph])
        risk_reward_img_cells_local, risk_reward_cell_type_local, risk_reward_cell_cost_local = cell_process.create_cells(risk_reward_image_local, risk_image_local, CELLS_SIZE, show=False)
        state_diagram_local, state_dict_local = cell_process.cells_to_state_diagram(risk_reward_cell_type_local, risk_reward_cell_cost_local, show=False)
        shortest_path = dijkstra.dj_algo(risk_reward_img_cells_local, risk_reward_cell_type_local, (current_phys_loc, next_phys_loc), state_diagram_local, CELLS_SIZE)

        dj_path_image_local = risk_reward_img_cells_local.copy()
        dijkstra.draw_path_global(shortest_path, dj_path_image_local, (current_phys_loc, next_phys_loc), CELLS_SIZE)

        half_cell = math.ceil((CELLS_SIZE/2))
        center = (current_phys_loc[0]*CELLS_SIZE+half_cell, current_phys_loc[1]*CELLS_SIZE+half_cell)
        dj_path_image_local = cv2.circle(dj_path_image_local, center, 4, (255, 255, 255), 1)

        plt.imshow(dj_path_image_local); plt.savefig(f"/tmp/thesis/pic{ img_tmp_idx }.png")
        img_tmp_idx+=1

        current_phys_loc = dijkstra.get_next_cell_shortest_path(shortest_path, current_phys_loc)
        print(current_phys_loc)

        center = (current_phys_loc[0]*CELLS_SIZE+half_cell, current_phys_loc[1]*CELLS_SIZE+half_cell)
        dj_path_image_adhoc = cv2.circle(dj_path_image_adhoc, center, 4, (255, 255, 255), 1)


    plt.imshow(dj_path_image_adhoc); plt.savefig(f"/tmp/thesis/pic000adhoc.png")
    exit()




    # find next state that we should go to
    next_ltl_state = ltl_process.get_next_state(ltl_state_diag, cell_type, current_ltl_state, next_phys_loc)

    # print(next_phys_loc)
    # print(start_state)
    # print(cell_type[next_phys_loc[1]][next_phys_loc[0]])
    # print(aps)
    # print(ltl_state_diag)

    # setup next interation
    current_ltl_state = next_ltl_state
    start_phys_loc = next_phys_loc

# show final path
plt.imshow(dj_path_image); plt.show()

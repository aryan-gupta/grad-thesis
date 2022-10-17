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
MAX_WEIGHT = float("inf")

map_h = 640
map_w = 576

# Read image and show it
img = cv2.imread('./sample.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.show()

points = [[1025, 132], [855, 2702], [3337, 2722], [2974, 165]]
wpcc_img = img_process.perspective_warp(img, points, map_w, map_h)

(red_channel, green_channel, blue_channel, yellow_channel) = img_process.color_segment_image(wpcc_img)

processed_img = img_process.merge_colors(red_channel, green_channel, blue_channel, yellow_channel)
# plt.imshow(processed_img); plt.show()

orig_goal_reward_image = cv2.add(cv2.add(red_channel, blue_channel), yellow_channel)
goal_reward_image = img_process.apply_edge_blur(orig_goal_reward_image, 128)

risk_image = img_process.create_risk_img(green_channel, 64)

img_cells, cell_type, cell_cost = cell_process.create_cells(processed_img, risk_image, CELLS_SIZE)

cell_type = cell_process.convert_cells(cell_type, objectives=["A", "B"], goals=["S", "F"])
start, finish = cell_process.get_start_finish_locations(cell_type)

reward_graphs = img_process.get_reward_images(cell_type, orig_goal_reward_image, CELLS_SIZE)

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


current_ltl_state = start_state
start_phys_loc = start
next_phys_loc = finish
dj_path_image = img_cells.copy()

while current_ltl_state != final_state:
    current_state_reward_graph = ltl_process.get_reward_img_state(ltl_state_diag, current_ltl_state, reward_graphs, (map_h, map_w))
    reward_current = img_process.apply_edge_blur(current_state_reward_graph, 128)
    # plt.imshow(reward_current, cmap="gray"); plt.show()

    risk_reward_image = cv2.merge([current_state_reward_graph, risk_image, current_state_reward_graph])
    # plt.imshow(current_state_reward_graph, cmap="gray"); plt.show()
    # plt.imshow(risk_image, cmap="gray"); plt.show()
    # plt.imshow(risk_reward_image); plt.show()


    # Convert risk_reward_image into cells
    risk_reward_img_cells, risk_reward_cell_type, risk_reward_cell_cost = cell_process.create_cells(risk_reward_image, risk_image, CELLS_SIZE)
    # plt.imshow(risk_reward_img_cells); plt.show()
    # Print the cell type map for debugging
    # for y in risk_reward_cell_type:
    #     print(y)
    # print()
    # print()

    # # Print the cell cost map for debugging
    # for y in cell_cost:
    #     for cost in y:
    #         print("{:.2f}".format(cost), end=", ")
    #     print()
    # print()
    # print()

    state_diagram, state_dict = cell_process.cells_to_state_diagram(risk_reward_cell_type, risk_reward_cell_cost, MAX_WEIGHT)
    # cell_process.pretty_print_state_dd(state_diagram, state_dict)
    _, next_phys_loc = cell_process.get_start_finish_locations(risk_reward_cell_type)

    # state_diagram, state_dict = cell_process.cells_to_state_diagram(cell_type, cell_cost, MAX_WEIGHT)
    # cell_process.pretty_print_state_dd(state_diagram, state_dict)

    shortest_path = dijkstra.dj_algo(img_cells, cell_type, (start_phys_loc, next_phys_loc), state_diagram, CELLS_SIZE)
    _ = dijkstra.draw_shortest_path(shortest_path, risk_reward_img_cells, reward_graphs, (start_phys_loc, next_phys_loc), CELLS_SIZE)
    dijkstra.draw_path_global(shortest_path, dj_path_image, (start_phys_loc, next_phys_loc), CELLS_SIZE)

    next_ltl_state = ltl_process.get_next_state(ltl_state_diag, cell_type, current_ltl_state, next_phys_loc)

    # print(next_phys_loc)
    # print(start_state)
    # print(cell_type[next_phys_loc[1]][next_phys_loc[0]])
    # print(aps)
    # print(ltl_state_diag)

    current_ltl_state = next_ltl_state
    start_phys_loc = next_phys_loc

plt.imshow(dj_path_image); plt.show()


shortest_path = dijkstra.dj_algo(img_cells, cell_type, (start, finish), state_diagram, CELLS_SIZE)
_ = dijkstra.draw_shortest_path(shortest_path, risk_reward_img_cells, reward_graphs, (start, finish), CELLS_SIZE)
exit()

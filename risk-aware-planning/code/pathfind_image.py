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
plt.imshow(processed_img); plt.show()

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
plt.imshow(reward_current, cmap="gray"); plt.show()

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


exit()

##################################### LTL Formula Conversion ##############################################


# This the LTL ormula converted to a buchii automata
ltl_auto = ["0", "1", "2", "3"]
def ltl_auto_valid(src, dest, ops):
    if src == "0": return True

    if src == "1" and dest =="0": return True if "B" in ops else False
    if src == "1" and dest =="1": return True if "B" not in ops else False

    if src == "2" and dest =="0": return True if "A" in ops and "B" in ops else False
    if src == "2" and dest =="1": return True if "A" in ops and "B" not in ops else False
    if src == "2" and dest =="2": return True if "A" not in ops and "B" not in ops and "R" not in ops else False
    if src == "2" and dest =="3": return True if "A" not in ops and "B" in ops and "R" not in ops else False

    if src == "3" and dest =="3": return True if "A" not in ops and "R" not in ops else False
    if src == "3" and dest =="0": return True if "A" in ops else False

    return False

# This is the map FSM wrapper func, allows us to write the algoithm easier
def phys_map_fsm_valid(src, dest, ops):
    valid_paths = state_dict[src]
    for direction in valid_paths:
        if dest == direction[1]:
            return True
    return False


##################################### Product Automata Construction ##############################################


# PRODUCT AUTOMATA Code
# see the other folders on how this works
# waws copied from there
auto_final = {}
key_f = []

for key_1 in state_dict.keys():
    for key_2 in ltl_auto:
        key_f.append((key_1, key_2))

print(key_f)
print(len(key_f))

for src_1, src_2 in key_f:
    for dest_1, dest_2 in key_f:
        state_diagram_cord = dest_1.split("-")
        state_diagram_cord_x = int(state_diagram_cord[0])
        state_diagram_cord_y = int(state_diagram_cord[1])
        state_at_cord = state_diagram[state_diagram_cord_y][state_diagram_cord_x][4]
        if phys_map_fsm_valid(src_1, dest_1, []) and ltl_auto_valid(src_2, dest_2, [state_at_cord]):
            key_f_src_str = src_1 + ',' + src_2
            key_f_dest_str = dest_1 + ',' + dest_2
            if key_f_src_str not in auto_final.keys(): auto_final[key_f_src_str] = []
            auto_final[key_f_src_str].append(key_f_dest_str)

for key in auto_final:
    print(key, end=" : ") 
    print(auto_final[key]) 

auto_final_start = f"{start[0]}-{start[1]},2"
auto_final_end = f"{finish[0]}-{finish[1]},0"

print()
print(auto_final_start)
print(auto_final_end)

##################################### D's Algo on Product Automata ##############################################


## MOST OF THE CODE BELOW IS COPIED FROM ABOVE and does the same thing so Im not going to detail
## comment it. 
# pretty much its just D's algo, video making, and making the images

# Dijkstras algo
# When I wrote this code, only god and I knew how it works. Now, only god knows
queue = [] # queue is an array of (weight, (x, y))
visited_nodes = [] # create bool false array same size as state_diagram
distances = {}
prev = {}

queue.append((0,auto_final_start))
distances[auto_final_start] = 0

visited_image = cv2.cvtColor(img_cells.copy(), cv2.COLOR_BGR2RGB)
video_out = cv2.VideoWriter('project_final.mkv',cv2.VideoWriter_fourcc('M','P','4','V'), 15, (visited_image.shape[1], visited_image.shape[0]))

while len(queue) != 0:
    # get first element
    current = queue[0]
    queue = queue[1:]

    # unpack element
    dist = current[0]
    node = current[1]

    # if weve already been to this node, skip it
    if (node in visited_nodes): continue
    # mark node as visited
    visited_nodes.append(node)
    # get directions we can travel
    valid_paths = auto_final[node]

    node_x = int(node.split(',')[0].split('-')[0])
    node_y = int(node.split(',')[0].split('-')[1])
    half_cell = math.ceil((CELLS_SIZE/2))
    center = (node_x*CELLS_SIZE+half_cell, node_y*CELLS_SIZE+half_cell)
    visited_image = cv2.circle(visited_image, center, 4, (0, 255, 255), 5)
    # plt.imshow(visited_image)
    # plt.show()
    video_out.write(visited_image)
    visited_image = cv2.circle(visited_image, center, 4, (100 + (dist*10), 0, 100 + (dist*10)), 5)

    for path in valid_paths:
        if path not in distances.keys(): distances[path] = MAX_WEIGHT
        
        old_distance = distances[path]
        new_distance = dist + 1
        
        if new_distance <= old_distance:
            distances[path] = new_distance
            prev[path] = node
        
        bisect.insort(queue, (distances[path], path), key=lambda a: a[0])

for key in distances.keys():
    print(key, end=" : ") 
    print(distances[key]) 

print() 
print() 
for key in prev.keys():
    print(key, end=" : ") 
    print(prev[key]) 

# calculate the shortest path
shortest_path = []
current_node = auto_final_end
visited_image_b4_backtrace = visited_image.copy()
while current_node != auto_final_start:
    node_x = int(current_node.split(',')[0].split('-')[0])
    node_y = int(current_node.split(',')[0].split('-')[1])
    half_cell = math.ceil((CELLS_SIZE/2))
    center = (node_x*CELLS_SIZE+half_cell, node_y*CELLS_SIZE+half_cell)
    visited_image = cv2.circle(visited_image, center, 4, (255, 255, 255), 5)
    if state_diagram[node_y][node_x][4] == "A" or state_diagram[node_y][node_x][4] == "B":
        visited_image = visited_image_b4_backtrace.copy()
    for i in range(3):
        video_out.write(visited_image)

    shortest_path.append(current_node)
    current_node = prev[current_node]
shortest_path.append(auto_final_start)

print(shortest_path)

shortest_path_phys = []
for path in shortest_path:
    sp = path.split(",")
    shortest_path_phys.append(sp[0])

print(shortest_path_phys)


# draw the shortest path
img_final_djk = cv2.cvtColor(img_cells.copy(), cv2.COLOR_BGR2RGB)
for i in range(len(shortest_path_phys) - 1):
    half_cell = math.ceil((CELLS_SIZE/2))
    
    if shortest_path_phys[i] == start: break
    
    node_str = shortest_path_phys[i]
    next_node_str = shortest_path_phys[i+1]
    
    node = (int(node_str.split("-")[0]), int(node_str.split("-")[1]))
    next_node = (int(next_node_str.split("-")[0]), int(next_node_str.split("-")[1]))

    center = (node[0]*CELLS_SIZE+half_cell, node[1]*CELLS_SIZE+half_cell)
    next_center = (next_node[0]*CELLS_SIZE+half_cell, next_node[1]*CELLS_SIZE+half_cell)
    
    img_final_djk = cv2.line(img_final_djk, center, next_center, (255,255,255), 8)

for i in range(60):
    video_out.write(img_final_djk)
video_out and video_out.release()

# plt.imshow(img_final_djk); plt.show()

print(len(key_f))
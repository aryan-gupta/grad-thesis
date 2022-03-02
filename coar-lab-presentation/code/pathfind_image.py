import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import bisect

CELLS_SIZE = 32 # 32 pixels
MAX_WEIGHT = 999

img = cv2.imread('./sample1.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue_channel, sat_channel, _ = cv2.split(img)

plt.imshow(hue_channel)
plt.show()
plt.imshow(sat_channel)
plt.show()

red_low_channel = cv2.bitwise_and(cv2.inRange(hue_channel, 0, 5), cv2.inRange(sat_channel, 100, 255))
red_high_channel = cv2.bitwise_and(cv2.inRange(hue_channel, 175, 180), cv2.inRange(sat_channel, 100, 255))
red_channel = cv2.bitwise_or(red_low_channel, red_high_channel)
green_channel = cv2.bitwise_and(cv2.inRange(hue_channel, 60, 67), cv2.inRange(sat_channel, 100, 255))
blue_channel = cv2.bitwise_and(cv2.inRange(hue_channel, 100, 110), cv2.inRange(sat_channel, 100, 255))
yellow_channel = cv2.bitwise_and(cv2.inRange(hue_channel, 20, 30), cv2.inRange(sat_channel, 100, 255))
red_channel = cv2.bitwise_or(red_channel, yellow_channel)
green_channel = cv2.bitwise_or(green_channel, yellow_channel)

processed_img = cv2.merge([red_channel,green_channel,blue_channel])
plt.imshow(processed_img)
plt.show()

dim = img.shape
print(dim)

cells_height = math.floor(dim[0] / CELLS_SIZE)
cells_width  = math.floor(dim[1] / CELLS_SIZE)

print((cells_height, cells_width))

colors = []

cell_type = []
cell_num_height = 0
cell_num_width = 0
for y in range(0, dim[0], CELLS_SIZE):
    cell_type.append([])
    cell_num_height = 0

    for x in range(0, dim[1], CELLS_SIZE):
        cell_type[cell_num_width].append('C')

        # draw rectangles
        img_cells = cv2.rectangle(processed_img, (x+1,y+1), (x + CELLS_SIZE,y + CELLS_SIZE), (255,255,255), 1)
        
        # determine what the cell is
        cell_known = False
        for u in range(y, y + CELLS_SIZE, 1):
            if u >= dim[0]:
                break
            
            for v in range(x, x + CELLS_SIZE, 1):
                if v >= dim[1]:
                    break

                # keep a record of all the different colors
                if tuple(processed_img[u,v]) not in colors:
                    colors.append(tuple(processed_img[u,v]))
                
                # mark the cells if its corosponding color exists in the cell
                if tuple(processed_img[u,v]) == (0,255,0): # Hazard Cells
                    cell_known = True
                    img_cells = cv2.putText(img_cells, 'H',(x+3, y+CELLS_SIZE-3), 2, 1, (255,255,255),1)
                    cell_type[cell_num_width][cell_num_height] = 'H'
                if tuple(processed_img[u,v]) == (255, 0, 0): # Goal Cells
                    cell_known = True
                    img_cells = cv2.putText(img_cells, 'G',(x+3, y+CELLS_SIZE-3), 2, 1, (255,255,255),1)
                    cell_type[cell_num_width][cell_num_height] = 'G'
                if tuple(processed_img[u,v]) == (255, 255, 0): # Objective Cells
                    cell_known = True
                    img_cells = cv2.putText(img_cells, 'O',(x+3, y+CELLS_SIZE-3), 2, 1, (255,255,255),1)
                    cell_type[cell_num_width][cell_num_height] = 'O'              
                if tuple(processed_img[u,v]) == (0, 0, 255): # Refuel Cells
                    cell_known = True
                    img_cells = cv2.putText(img_cells, 'R',(x+3, y+CELLS_SIZE-3), 2, 1, (255,255,255),1)
                    cell_type[cell_num_width][cell_num_height] = 'R'
                
            # Exit loop if we know the cell type
            if cell_known:
                break

        # if we dont know the cell type (its all white), mark it as a clean cell
        if not cell_known:    
            img_cells = cv2.putText(img_cells, 'C',(x+3, y+CELLS_SIZE-3), 2, 1, (255,255,255),1)
        cell_num_height += 1
    cell_num_width += 1

print(colors)
print()
print()

for y in cell_type:
    print(y)
print()
print()

plt.imshow(img_cells)
plt.show()

def convert_cells(cell_type, y, x, orig_value, new_value):
    cell_type[y][x] = new_value
    if y > 0 and cell_type[y - 1][x] == orig_value:
        convert_cells(cell_type, y - 1, x, orig_value, new_value)
    if x > 0 and cell_type[y][x - 1] == orig_value:
        convert_cells(cell_type, y, x - 1, orig_value, new_value)
    if y < (len(cell_type)-1) and cell_type[y + 1][x] == orig_value:
        convert_cells(cell_type, y + 1, x, orig_value, new_value)
    if x < (len(cell_type[y])-1) and cell_type[y][x + 1] == orig_value:
        convert_cells(cell_type, y, x + 1, orig_value, new_value)

objectives = ["A", "B"]
objectives_idx = 0
goals = ["S", "F"]
goals_idx = 0
# Convert Goal Cells into start and finish cells
for y in range(len(cell_type)):
    for x in range(len(cell_type[y])):
        if cell_type[y][x] == "O":
            convert_cells(cell_type, y, x, "O", objectives[objectives_idx])
            objectives_idx += 1
        if cell_type[y][x] == "G":
            convert_cells(cell_type, y, x, "G", goals[goals_idx])
            goals_idx += 1

for y in cell_type:
    print(y)
print()
print()

def is_valid_travel_cell(c):
    if c in ["A", "B", "C", "S", "F"]:
        return True
    return False

state_diagram = []
for y in range(len(cell_type)):
    state_diagram.append([])
    for x in range(len(cell_type[0])):
        state_diagram[y].append([MAX_WEIGHT, MAX_WEIGHT, MAX_WEIGHT, MAX_WEIGHT, cell_type[y][x]])
        if cell_type[y][x] == 'H':
            continue
        # check up left
        # NOT IMPL
        
        # check up
        if y > 0 and is_valid_travel_cell(cell_type[y - 1][x]):
            state_diagram[y][x][0] = 1
        # check up right
        # NOT IMPL
        
        # check left
        if x > 0 and is_valid_travel_cell(cell_type[y][x - 1]):
            state_diagram[y][x][1] = 1
        # check right
        if x < (len(cell_type[0]) - 1) and is_valid_travel_cell(cell_type[y][x + 1]):
            state_diagram[y][x][2] = 1
        # check down left
        # NOT IMPL

        # check down
        if y < (len(cell_type) - 1) and is_valid_travel_cell(cell_type[y + 1][x]):
            state_diagram[y][x][3] = 1
        # check down right
        # NOT IMPL

# pretty print state diagram
for row in state_diagram:
    # Up arrows
    for col in row:
        print(" ", end="") # space for the left arrow
        if col[0] != MAX_WEIGHT:
            print("↑", end="")
        else:
            print(" ", end="")
        print(" ", end="") # space for the right arrow
    print()
    # left/right and center char
    for col in row:
        if col[1] != MAX_WEIGHT:
            print("←", end="")
        else:
            print(" ", end="")
        print(col[4], end="")
        if col[2] != MAX_WEIGHT:
            print("→", end="")
        else:
            print(" ", end="")
    print()
    # Down arrows
    for col in row:
        print(" ", end="") # space for the left arrow
        if col[3] != MAX_WEIGHT:
            print("↓", end="")
        else:
            print(" ", end="")
        print(" ", end="") # space for the right arrow
    print()
    

start = ()
finish = ()
# find the start node
for y in range(len(cell_type)):
    for x in range(len(cell_type[0])):
        if cell_type[y][x] == 'S':
            start = (x, y)
        if cell_type[y][x] == 'F':
            finish = (x, y)

print(start)
print(finish)


# Dijkstras algo
# When I wrote this code, only god and I knew how it works. Now, only god knows
queue = [] # queue is an array of (weight, (x, y))
visited_nodes = [ [False] * len(cell_type[0]) for _ in range(len(cell_type))] # create bool false array same size as state_diagram
distances = [ [MAX_WEIGHT] * len(cell_type[0]) for _ in range(len(cell_type))]
prev = [ [(0,0)] * len(cell_type[0]) for _ in range(len(cell_type))]

queue.append((0,start))
distances[start[1]][start[0]] = 0

while len(queue) != 0:
    # get first element
    current = queue[0]
    queue = queue[1:]

    # unpack element
    x = current[1][0]
    y = current[1][1]
    dist = current[0]

    # if weve already been to this node, skip it
    if (visited_nodes[y][x]): continue
    # mark node as visited
    visited_nodes[y][x] = True
    # get directions we can travel
    valid_paths = state_diagram[y][x]

    # check each direction we can travel
    if valid_paths[0] == 1 and not visited_nodes[y-1][x]: # UP
        old_distance = distances[y - 1][x]
        new_distance = dist + state_diagram[y][x][0]
        if new_distance <= old_distance:
            distances[y - 1][x] = new_distance
            prev[y - 1][x] = (x,y)
        bisect.insort(queue, (distances[y - 1][x], (x,y-1)), key=lambda a: a[0])
    if valid_paths[1] == 1 and not visited_nodes[y][x-1]: # LEFT
        old_distance = distances[y][x - 1]
        new_distance = dist + state_diagram[y][x][1]
        if new_distance <= old_distance:
            distances[y][x - 1] = new_distance
            prev[y][x - 1] = (x,y)
        bisect.insort(queue, (distances[y][x - 1], (x-1,y)), key=lambda a: a[0])
    if valid_paths[2] == 1 and not visited_nodes[y][x+1]: # RIGHT
        old_distance = distances[y][x + 1]
        new_distance = dist + state_diagram[y][x][2]
        if new_distance <= old_distance:
            distances[y][x + 1] = new_distance
            prev[y][x + 1] = (x,y)
        bisect.insort(queue, (distances[y][x + 1], (x+1,y)), key=lambda a: a[0])
    if valid_paths[3] == 1 and not visited_nodes[y+1][x]: # DOWN
        old_distance = distances[y + 1][x]
        new_distance = dist + state_diagram[y][x][3]
        if new_distance <= old_distance:
            distances[y + 1][x] = new_distance
            prev[y + 1][x] = (x,y)
        bisect.insort(queue, (distances[y + 1][x], (x,y+1)), key=lambda a: a[0])

for y in distances:
    print(y)

for y in prev:
    print(y)

# calculate the shortest path
shortest_path = []
current_node = finish
while current_node != start:
    shortest_path.append(current_node)
    current_node = prev[current_node[1]][current_node[0]]
shortest_path.append(start)
    
print(shortest_path)

# draw the shortest path
for i in range(len(shortest_path)):
    half_cell = math.ceil((CELLS_SIZE/2))
    
    if shortest_path[i] == start: break
    
    node = shortest_path[i]
    next_node = shortest_path[i+1]
    
    center = (node[0]*CELLS_SIZE+half_cell, node[1]*CELLS_SIZE+half_cell)
    next_center = (next_node[0]*CELLS_SIZE+half_cell, next_node[1]*CELLS_SIZE+half_cell)
    
    img_cells = cv2.line(img_cells, center, next_center, (255,0,0), 2)


plt.imshow(img_cells)
plt.show()

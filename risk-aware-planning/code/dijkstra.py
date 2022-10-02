import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import bisect

def dj_algo(img_cells, cell_type, points, state_diagram, CELLS_SIZE):
    # Start creating a video of the D's algo in working
    # visited_image = cv2.cvtColor(img_cells.copy(), cv2.COLOR_BGR2RGB)
    # video_out = cv2.VideoWriter('project_phys_only.mkv',cv2.VideoWriter_fourcc('M','P','4','V'), 15, (visited_image.shape[1], visited_image.shape[0]))

    start, finish = points
    map_w, map_h = (img_cells.shape[1], img_cells.shape[0])

    # Dijkstras algo
    # When I wrote this code, only god and I knew how it works. Now, only god knows
    queue = [] # queue is an array of (weight, (x, y))
    visited_nodes = [ [False] * len(cell_type[0]) for _ in range(len(cell_type))] # create bool false array same size as state_diagram
    distances = [ [float("inf")] * len(cell_type[0]) for _ in range(len(cell_type))]
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

        half_cell = math.ceil((CELLS_SIZE/2))
        center = (x*CELLS_SIZE+half_cell, y*CELLS_SIZE+half_cell)
        # visited_image = cv2.circle(visited_image, center, 4, (0, 255, 255), 1)
        # plt.imshow(visited_image)
        # plt.show()

        # write the current state as an image into the video
        # video_out.write(visited_image)
        # visited_image = cv2.circle(visited_image, center, 4, (100 + (dist*10), 0, 100 + (dist*10)), 1)

        # check each direction we can travel
        if y > 0: # UP
            old_distance = distances[y - 1][x]
            new_distance = dist + state_diagram[y][x][0]
            if new_distance < old_distance:
                distances[y - 1][x] = new_distance
                prev[y - 1][x] = (x,y)
                bisect.insort(queue, (distances[y - 1][x], (x,y-1)), key=lambda a: a[0])
        if x > 0: # LEFT
            old_distance = distances[y][x - 1]
            new_distance = dist + state_diagram[y][x][1]
            if new_distance < old_distance:
                distances[y][x - 1] = new_distance
                prev[y][x - 1] = (x,y)
                bisect.insort(queue, (distances[y][x - 1], (x-1,y)), key=lambda a: a[0])
        if x < (len(cell_type[0]) - 1): # RIGHT
            old_distance = distances[y][x + 1]
            new_distance = dist + state_diagram[y][x][2]
            if new_distance < old_distance:
                distances[y][x + 1] = new_distance
                prev[y][x + 1] = (x,y)
                bisect.insort(queue, (distances[y][x + 1], (x+1,y)), key=lambda a: a[0])
        if y < (len(cell_type) - 1): # DOWN
            old_distance = distances[y + 1][x]
            new_distance = dist + state_diagram[y][x][3]
            if new_distance < old_distance:
                distances[y + 1][x] = new_distance
                prev[y + 1][x] = (x,y)
                bisect.insort(queue, (distances[y + 1][x], (x,y+1)), key=lambda a: a[0])

    # # Print the distances map
    # for y in distances:
    #     for dist in y:
    #         print("{:.2f}".format(dist), end=", ")
    #     print()

    # # Print the previous cell map
    # for y in prev:
    #     print(y)

    # calculate the shortest path and create a video while 
    shortest_path = []
    current_node = finish
    while current_node != start:
        # write the current back trace state into the video
        half_cell = math.ceil((CELLS_SIZE/2))
        center = (current_node[0]*CELLS_SIZE+half_cell, current_node[1]*CELLS_SIZE+half_cell)
        # visited_image = cv2.circle(visited_image, center, 4, (255, 255, 255), 1)
        # for i in range(3):
        #     video_out.write(visited_image)

        shortest_path.append(current_node)
        current_node = prev[current_node[1]][current_node[0]]
    shortest_path.append(start)

    # pause for two seconds on the final frame
    # for i in range(60):
    #     video_out.write(visited_image)
    # video_out and video_out.release()

    # print(shortest_path)

    return shortest_path


def draw_shortest_path(shortest_path, risk_reward_img_cells, reward_graphs, points, CELLS_SIZE):
    start, finish = points

    # draw the shortest path
    map_w, map_h = (risk_reward_img_cells.shape[1], risk_reward_img_cells.shape[0])
    empty_image = np.zeros((map_h, map_w, 1), dtype = "uint8")
    img_plain_djk = risk_reward_img_cells.copy()
    img_plain_djk = cv2.add(img_plain_djk, cv2.merge([empty_image, empty_image, reward_graphs['S']]))
    for i in range(len(shortest_path)):
        half_cell = math.ceil((CELLS_SIZE/2))
        
        if shortest_path[i] == start: break
        
        node = shortest_path[i]
        next_node = shortest_path[i+1]
        
        center = (node[0]*CELLS_SIZE+half_cell, node[1]*CELLS_SIZE+half_cell)
        next_center = (next_node[0]*CELLS_SIZE+half_cell, next_node[1]*CELLS_SIZE+half_cell)
        
        img_plain_djk = cv2.line(img_plain_djk, center, next_center, (0,255,255), 1)

    # Show the path found image from D's algo
    # plt.imshow(img_plain_djk)
    # plt.show()


def draw_path_global(shortest_path, img_cells, points, CELLS_SIZE):
    start, finish = points

    for i in range(len(shortest_path)):
        half_cell = math.ceil((CELLS_SIZE/2))
        
        if shortest_path[i] == start: break
        
        node = shortest_path[i]
        next_node = shortest_path[i+1]
        
        center = (node[0]*CELLS_SIZE+half_cell, node[1]*CELLS_SIZE+half_cell)
        next_center = (next_node[0]*CELLS_SIZE+half_cell, next_node[1]*CELLS_SIZE+half_cell)
        
        img_cells = cv2.line(img_cells, center, next_center, (0,255,255), 1)

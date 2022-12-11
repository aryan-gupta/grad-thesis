import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import bisect


# returns the cost function
# the cost function returns the correct cost function for the cell_cost data structure
# this can be checked by testing if x/y elements is an array, because if it isnt, its
# the raw cost value and not the state_diagram
def get_cfunc(ds):
    return cell_cost_cfunc


# the cost function for the cell_cost raw method
def cell_cost_cfunc(points, cell_cost, direction):
    x, y = points

    if direction == 0:
        if y > 0:
            return cell_cost[y - 1][x]

    if direction == 1:
        if x > 0:
            return cell_cost[y][x - 1]

    if direction == 2:
        if x < (len(cell_cost[0]) - 1):
            return cell_cost[y][x + 1]

    if direction == 3:
        if y < (len(cell_cost) - 1):
            return cell_cost[y + 1][x]


# creates a function that uses the euclidean distance to the next_phys_loc
# is used when doing a partial a* algo
def create_astar_partial_hfunc(next_phys_loc):
    # for the a* algo, the heuristic function is the euclidean distance
    # from the current pos to the final pos
    def astar_algo_default_hfunc(current_phys_loc, target_phy_loc):
        dx = current_phys_loc[0] - next_phys_loc[0]
        dy = current_phys_loc[1] - next_phys_loc[1]
        euclidean_distance = math.sqrt((dx**2) + (dy**2))
        return 0.0005 * euclidean_distance

    return astar_algo_default_hfunc


# runs the a* algo with the points[1] being a partial target and the hfunc uses next_phys_loc as
# the euclidean distance heuristic
def astar_algo_partial_target(img_cells, cell_type, points, next_phys_loc, cell_cost, CELLS_SIZE):
    astar_algo_hfunc = create_astar_partial_hfunc(next_phys_loc)
    cfunc = get_cfunc(cell_cost)
    return dj_algo_cfunc_hfunc(img_cells, cell_type, points, cell_cost, cfunc, CELLS_SIZE, astar_algo_hfunc)


# this is only here for legacy reasons, will be removed later
# img_cells used to be a parameter as it was used to create a video
# now this feature is done by calling function
def astar_algo(img_cells, cell_type, points, cell_cost, CELLS_SIZE):
    # for the a* algo, the heuristic function is the euclidean distance
    # from the current pos to the final pos
    def astar_algo_hfunc(current_phys_loc, next_phy_loc, direction):
        dx = current_phys_loc[0] - next_phy_loc[0]
        dy = current_phys_loc[1] - next_phy_loc[1]
        euclidean_distance = math.sqrt((dx**2) + (dy**2))
        return 0.0005 * euclidean_distance

    cfunc = get_cfunc(cell_cost)
    return dj_algo_cfunc_hfunc(img_cells, cell_type, points, cell_cost, cfunc, CELLS_SIZE, astar_algo_hfunc)


def astar_opt(img_cells, cell_type, points, cell_cost, CELLS_SIZE, optimizer):
    def astar_algo_hfunc(current_phys_loc, next_phy_loc, direction, optimizer=optimizer):
        if not optimizer.is_valid_direction(current_phys_loc, direction):
            return float("inf")

        if optimizer.is_direction_forbidden_by_task(current_phys_loc, direction):
            return float("inf")

        dx = current_phys_loc[0] - next_phy_loc[0]
        dy = current_phys_loc[1] - next_phy_loc[1]
        euclidean_distance = math.sqrt((dx**2) + (dy**2))
        return 0.0005 * euclidean_distance

    cfunc = get_cfunc(cell_cost)
    return dj_algo_cfunc_hfunc(img_cells, cell_type, points, cell_cost, cfunc, CELLS_SIZE, astar_algo_hfunc)

# for djk's algo, the heuristic function always return 0 since we dont use
# a heuristic
def dj_algo_default_hfunc(*args, **kwargs):
    return 0


# to prevent breaking changes, this helper function will fix legacy
# functions that call djk algo without an hfunc
def dj_algo(img_cells, cell_type, points, cell_cost, CELLS_SIZE):
    cfunc = get_cfunc(cell_cost)
    return dj_algo_cfunc_hfunc(img_cells, cell_type, points, cell_cost, cfunc, CELLS_SIZE, dj_algo_default_hfunc)


# Runs a dijkstra's algorithm on cell_type from the start and end locations
# from points. The img_cells are used to make the video
# @TODO remove img_cells parameter
def dj_algo_cfunc_hfunc(img_cells, cell_type, points, cell_cost, cfunc, CELLS_SIZE, hfunc):
    # Start creating a video of the D's algo in working
    # visited_image = cv2.cvtColor(img_cells.copy(), cv2.COLOR_BGR2RGB)
    # video_out = cv2.VideoWriter('project_phys_only.mkv',cv2.VideoWriter_fourcc('M','P','4','V'), 15, (visited_image.shape[1], visited_image.shape[0]))

    start, finish = points
    map_w, map_h = (img_cells.shape[1], img_cells.shape[0])

    # Dijkstras algo
    # When I wrote this code, only god and I knew how it works. Now, only god knows
    queue = [] # queue is an array of (weight, (x, y))
    visited_nodes = [ [False] * len(cell_type[0]) for _ in range(len(cell_type))] # create bool false array same size as cell_type
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
            new_distance = dist + cfunc((x, y), cell_cost, 0) + hfunc(current[1], finish, 0)
            if new_distance < old_distance:
                distances[y - 1][x] = new_distance
                prev[y - 1][x] = (x,y)
                bisect.insort(queue, (distances[y - 1][x], (x,y-1)), key=lambda a: a[0])
        if x > 0: # LEFT
            old_distance = distances[y][x - 1]
            new_distance = dist + cfunc((x, y), cell_cost, 1) + hfunc(current[1], finish, 1)
            if new_distance < old_distance:
                distances[y][x - 1] = new_distance
                prev[y][x - 1] = (x,y)
                bisect.insort(queue, (distances[y][x - 1], (x-1,y)), key=lambda a: a[0])
        if x < (len(cell_type[0]) - 1): # RIGHT
            old_distance = distances[y][x + 1]
            new_distance = dist + cfunc((x, y), cell_cost, 2) + hfunc(current[1], finish, 2)
            if new_distance < old_distance:
                distances[y][x + 1] = new_distance
                prev[y][x + 1] = (x,y)
                bisect.insort(queue, (distances[y][x + 1], (x+1,y)), key=lambda a: a[0])
        if y < (len(cell_type) - 1): # DOWN
            old_distance = distances[y + 1][x]
            new_distance = dist + cfunc((x, y), cell_cost, 3) + hfunc(current[1], finish, 3)
            if new_distance < old_distance:
                distances[y + 1][x] = new_distance
                prev[y + 1][x] = (x,y)
                bisect.insort(queue, (distances[y + 1][x], (x,y+1)), key=lambda a: a[0])

        if current[1] == finish:
            break
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


# Draws the shortest path for all LTL transitions
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


# Draws the path for one transition
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


# Gets the next cell in the shortest path
def get_next_cell_shortest_path(shortest_path, current_phys_loc):
    if current_phys_loc == shortest_path[0]:
        return current_phys_loc
    for i in range(len(shortest_path)):
        if current_phys_loc == shortest_path[i]:
            return shortest_path[i - 1]


# this function gets the target of the DJK/astar algo
# based off the previous path and the
def get_astar_target(current_phys_loc, shortest_path, distance):
    for idx in range(len(shortest_path)):
        dx = shortest_path[idx][0] - current_phys_loc[0]
        dy = shortest_path[idx][1] - current_phys_loc[1]

        if math.sqrt(dx**2 + dy**2) < distance:
            return shortest_path[idx], idx

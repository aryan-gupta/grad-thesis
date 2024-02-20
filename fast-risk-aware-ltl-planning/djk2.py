def _dj_algo_cfunc_hfunc(cell_type, points, cell_cost, cfunc, hfunc):

    start, finish = points

    queue = []
    visited_nodes = set()
    distances = {}
    prev = {}

    queue.append((0,start))
    distances[(start[1], start[0])] = 0

    while len(queue) != 0:
        # get first element
        current = queue[0]
        queue = queue[1:]

        # unpack element
        x = current[1][0]
        y = current[1][1]
        dist = current[0]

        # if weve already been to this node, skip it
        if (y, x) in visited_nodes: continue
        # mark node as visited
        visited_nodes.add((y, x))

        half_cell = math.ceil((gv.CELLS_SIZE/2))
        center = (x*gv.CELLS_SIZE+half_cell, y*gv.CELLS_SIZE+half_cell)

        # check each direction we can travel
        if y > 0: # UP
            up = y - 1
            old_distance = distances.get((up, x), float("inf"))
            new_distance = dist + cfunc((x, y), cell_cost, 0) + hfunc(current[1], finish, 0)
            if new_distance < old_distance:
                distances[(up, x)] = new_distance
                prev[(up, x)] = (x,y)
                bisect.insort(queue, (new_distance, (x,y-1)), key=lambda a: a[0])
        if x > 0: # LEFT
            left = x - 1
            old_distance = distances.get((y, left), float("inf"))
            new_distance = dist + cfunc((x, y), cell_cost, 1) + hfunc(current[1], finish, 1)
            if new_distance < old_distance:
                distances[(y, left)] = new_distance
                prev[(y, left)] = (x,y)
                bisect.insort(queue, (new_distance, (x-1,y)), key=lambda a: a[0])
        if x < (len(cell_type[0]) - 1): # RIGHT
            right = x + 1
            old_distance = distances.get((y, right), float("inf"))
            new_distance = dist + cfunc((x, y), cell_cost, 2) + hfunc(current[1], finish, 2)
            if new_distance < old_distance:
                distances[(y, right)] = new_distance
                prev[(y, right)] = (x,y)
                bisect.insort(queue, (new_distance, (x+1,y)), key=lambda a: a[0])
        if y < (len(cell_type) - 1): # DOWN
            down = y + 1
            old_distance = distances.get((down, x), float("inf"))
            new_distance = dist + cfunc((x, y), cell_cost, 3) + hfunc(current[1], finish, 3)
            if new_distance < old_distance:
                distances[(down, x)] = new_distance
                prev[(down, x)] = (x,y)
                bisect.insort(queue, (new_distance, (x,y+1)), key=lambda a: a[0])

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
        half_cell = math.ceil((gv.CELLS_SIZE/2))
        center = (current_node[0]*gv.CELLS_SIZE+half_cell, current_node[1]*gv.CELLS_SIZE+half_cell)
        # visited_image = cv2.circle(visited_image, center, 4, (255, 255, 255), 1)
        # for i in range(3):
        #     video_out.write(visited_image)

        shortest_path.append(current_node)
        current_node = prev[(current_node[1], current_node[0])]
    shortest_path.append(start)

    # pause for two seconds on the final frame
    # for i in range(60):
    #     video_out.write(visited_image)
    # video_out and video_out.release()

    # print(shortest_path)

    return shortest_path

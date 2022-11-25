import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import bisect
import PIL as pil


import img
import cell
import ltl
import dijkstra
import env
import random

# GLOBAL VARS
CELLS_SIZE = 8 # 32 pixels
VIEW_CELLS_SIZE = 8
UPDATE_WEIGHT = 0 #5

# final image dimensions (must be divisiable by CELLS_SIZE)
map_h = 800
map_w = 800

output_images_dir = '../../../tmp'
ltl_hoa_file = 'ltl.hoa.txt'


# reads in an image but doesnt pre process it
def get_env(input_image_file, show=False):
    global map_h
    global map_w

    red_channel = green_channel = blue_channel = None

    if input_image_file is None:
        # read image and show it
        img = env.create_env(2, (map_w, map_h))
    else:
        img = img.read_image(input_image_file, show=False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        map_h, map_w , _= img.shape

    red_channel, green_channel, blue_channel = cv2.split(img)

    if show: plt.imshow(green_channel); plt.show()
    if show: plt.imshow(red_channel); plt.show()

    return img, red_channel, green_channel


# creates all the reward graphs for each axiom
def create_reward_graphs(processed_img, raw_reward_image, raw_risk_image):
    # create cells based off of map and risk and assign costs to cells
    img_cells, cell_type, cell_cost = cell.create_cells(processed_img, raw_risk_image, CELLS_SIZE, show=False)

    # get start and finish locations from cell graph
    global_start, global_finish = cell.get_start_finish_locations(cell_type)

    # get reward map for each objectives and goals
    reward_graphs = img.get_reward_images(cell_type, raw_reward_image, CELLS_SIZE, show=False)

    return img_cells, reward_graphs, (global_start, global_finish)


# parses the ltl hoa file
def parse_ltl_hoa_file():
    # parse through LTL automata
    return ltl.parse_ltl_hoa(ltl_hoa_file)


# creates tha unknown/assumed risk image
# pretty much blurs the risk image but will need to find better way to do this
def get_assumed_risk(raw_risk_image):
    # create blurred risk image
    return img.create_risk_img(raw_risk_image, 16, show=False)

# pathfinds using a view range that updates the risk live
def pathfind(reward_graphs, raw_risk_image, assumed_risk_image, ltl_state_diag, ltl_heuristic, ltl_state_bounds, mission_phys_bounds, show=True):
    # get the start conditions
    current_ltl_state = ltl_state_bounds[0]
    current_phys_loc = mission_phys_bounds[0]

    # the shortest path of the entire mission
    total_shortest_path = []

    # the min possible path len if there was no risk
    min_path_len = 0

    # our local copy of risk from using our viewing range
    assumed_risk_image_filled = assumed_risk_image.copy()

    # any risk of 0 has min risk 5
    # assumed_risk_image_filled = cv2.normalize(assumed_risk_image_filled, None, 254, 10, norm_type = cv2.NORM_MINMAX)

    #  the index for ltl steps for the image output
    img_tmp_idx_ltl = 0

    # the loop to traverse the LTL formula
    while current_ltl_state != ltl_state_bounds[1]:
        # get reward map of current LTL state
        current_ltl_state_reward_graph = ltl.get_reward_img_state(ltl_state_diag, current_ltl_state, reward_graphs, (map_h, map_w))

        # we wont know the final_phys_loc or the dj's target location until we run our algo
        # store a dummy result in the meantime
        final_phys_loc = (-1,-1)

        # the index for phys steps for the image output
        img_tmp_idx_phys=0

        # This is needed so we can do partial replans
        current_planned_path = []
        risk_reward_image_local = None
        risk_reward_img_cells_local = None
        risk_reward_cell_type_local = None
        risk_reward_cell_cost_local = None

        # calculate min path len
        min_path_len += abs(current_phys_loc[0] - final_phys_loc[0]) + abs(current_phys_loc[1] - final_phys_loc[1])

        # the loop to traverse the phys enviroment
        while current_phys_loc != final_phys_loc:
            # add current node to path
            total_shortest_path.insert(0, current_phys_loc)

            # update risk map everytime we move
            assumed_risk_image_filled, amount_risk_updated, cells_updated = img.update_local_risk_image(assumed_risk_image_filled, raw_risk_image, current_phys_loc, CELLS_SIZE, VIEW_CELLS_SIZE, UPDATE_WEIGHT)

            # the temp target for partial astar algorithm
            astar_target = None

            # only do a full replan if its our first run
            # if we havent, then update the local data structures and do a partial replan if the risk values have changed
            if risk_reward_image_local is None:
                if show: print("full cells replanning")
                # create required data structures
                risk_reward_image_local = cv2.merge([current_ltl_state_reward_graph, assumed_risk_image_filled, np.zeros((map_h,map_w), np.uint8)])
                risk_reward_img_cells_local, risk_reward_cell_type_local, risk_reward_cell_cost_local = cell.create_cells(risk_reward_image_local, assumed_risk_image_filled, CELLS_SIZE, show=False)

                # get next phys loc based off the LTL state diag
                final_phys_loc = ltl.get_finish_location(risk_reward_cell_type_local, ltl_state_diag, ltl_heuristic, reward_graphs, current_ltl_state, CELLS_SIZE)
                # print(final_phys_loc)

                # apply dj's algo
                current_planned_path = dijkstra.astar_algo(risk_reward_img_cells_local, risk_reward_cell_type_local, (current_phys_loc, final_phys_loc), risk_reward_cell_cost_local, CELLS_SIZE)
                if show: print(len(current_planned_path))
            else:
                # instead of recreating out required data structures, just update the ones we "saw"
                # these are the same calls as full_replan except update_cells instead of create_cells
                risk_reward_image_local = cv2.merge([current_ltl_state_reward_graph, assumed_risk_image_filled, np.zeros((map_h,map_w), np.uint8)])
                risk_reward_img_cells_local, risk_reward_cell_type_local, risk_reward_cell_cost_local = cell.update_cells(cells_updated, risk_reward_image_local, risk_reward_cell_type_local, risk_reward_cell_cost_local, risk_reward_img_cells_local, current_phys_loc, assumed_risk_image_filled, CELLS_SIZE, VIEW_CELLS_SIZE)

                if show: print(amount_risk_updated)

                if amount_risk_updated > 10_000:
                    if show: print("full astar replanning")
                    current_planned_path = dijkstra.astar_algo(risk_reward_img_cells_local, risk_reward_cell_type_local, (current_phys_loc, final_phys_loc), risk_reward_cell_cost_local, CELLS_SIZE)
                elif amount_risk_updated > 0:
                    if show: print("part astar replanning")

                    # get astar's target cell
                    # this target cell will be somewhere on the current_planned_path line
                    # idx is the index of the astar_target cell
                    astar_target, idx = dijkstra.get_astar_target(current_phys_loc, current_planned_path, VIEW_CELLS_SIZE * 2)

                    # get new path from current loc to astar_target
                    shortest_path_astar_target = dijkstra.astar_algo_partial_target(risk_reward_img_cells_local, risk_reward_cell_type_local, (current_phys_loc, astar_target), final_phys_loc, risk_reward_cell_cost_local, CELLS_SIZE)

                    # splice our two shortest_paths together
                    current_planned_path = current_planned_path[0:idx]
                    current_planned_path = current_planned_path + shortest_path_astar_target
                    if show: print(len(current_planned_path))

            if show:
                # draw our taken path and future path on an image
                dj_path_image_local = risk_reward_img_cells_local.copy()
                dijkstra.draw_path_global(total_shortest_path, dj_path_image_local, (total_shortest_path[-1], total_shortest_path[0]), CELLS_SIZE)
                dijkstra.draw_path_global(current_planned_path, dj_path_image_local, (current_phys_loc, final_phys_loc), CELLS_SIZE)

                # draw the agent as a circle
                half_cell = math.ceil((CELLS_SIZE/2))
                center = (current_phys_loc[0]*CELLS_SIZE+half_cell, current_phys_loc[1]*CELLS_SIZE+half_cell)
                dj_path_image_local = cv2.circle(dj_path_image_local, center, 4, (255, 255, 255), 1)

                # draw the partial target is there is one
                if astar_target != None:
                    center = (astar_target[0]*CELLS_SIZE+half_cell, astar_target[1]*CELLS_SIZE+half_cell)
                    dj_path_image_local = cv2.circle(dj_path_image_local, center, 4, (255, 255, 255), 1)

                    # reset out astar_target so if we fully plan next loop, the circle doesnt get added
                    astar_target = None

                # save the image
                # print(img_tmp_idx_ltl, "-", img_tmp_idx_phys, " :: ", current_phys_loc, "(", amount_risk_updated, ")")
                img_tmp_idx_ltl_str  = str(img_tmp_idx_ltl).zfill(2)
                img_tmp_idx_phys_str = str(img_tmp_idx_phys).zfill(3)
                cv2.imwrite(f"{ output_images_dir }/pic{ img_tmp_idx_ltl_str }-{ img_tmp_idx_phys_str }.bmp", cv2.cvtColor(dj_path_image_local, cv2.COLOR_RGB2BGR) )

                # increment our image file counter
                img_tmp_idx_phys += 1

            # get the next location in the shortest path
            current_phys_loc = dijkstra.get_next_cell_shortest_path(current_planned_path, current_phys_loc)

        # find next state that we should go to and setup next interation
        current_ltl_state = ltl.get_next_state(ltl_state_diag, reward_graphs, current_ltl_state, final_phys_loc, CELLS_SIZE)
        img_tmp_idx_ltl += 1

    return total_shortest_path, min_path_len, assumed_risk_image_filled


# creates the final image to output
def create_final_image(processed_img, raw_risk_image, assumed_risk_image_filled, path, mission_phys_bounds):
    # seperate the image into RGB channels
    red_channel, green_channel, blue_channel = cv2.split(processed_img)

    # add our filled out assumed risk
    green_channel = cv2.add(green_channel, assumed_risk_image_filled)

    # make the actual walls white so its easy to tell apart from the green assumed risk surroundings
    red_channel = cv2.add(red_channel, raw_risk_image)
    green_channel = cv2.add(green_channel, raw_risk_image)
    blue_channel = cv2.add(blue_channel, raw_risk_image)

    # merge back our image into a single image
    dj_path_image = cv2.merge([red_channel, green_channel, blue_channel])

    # create our img_cell
    dj_path_image, _, _ = cell.create_cells(dj_path_image, assumed_risk_image_filled, CELLS_SIZE, show=False)

    # draw the path on img_cell
    dijkstra.draw_path_global(path, dj_path_image, mission_phys_bounds, CELLS_SIZE)

    return dj_path_image


def main():
    # since the seed is 0, the env will always be the same, helps when debugging
    random.seed(0)

    # read in and process image
    processed_img, raw_reward_image, raw_risk_image = get_env(None)
    # if you want to use your own image, CAUB (comment above, uncomment below), and change the filename parameter
    # processed_img, raw_reward_image, raw_risk_image = get_env('../../../maps/002.bmp')

    # create our axiom reward graphs
    processed_img_cells, reward_graphs, (mission_phys_start, mission_phys_finish) = create_reward_graphs(processed_img, raw_reward_image, raw_risk_image)

    # get the task details using LTL
    ltl_state_diag, aps, start_ltl_state, final_ltl_state = parse_ltl_hoa_file()

    # create our basic LTL heuristic model
    ltl_heuristic = dijkstra.dj_algo_ltl_heuristic(ltl_state_diag, final_ltl_state)

    # create our assumed risk image
    assumed_risk_image = get_assumed_risk(raw_risk_image)

    # pathfind while updating risk
    path, min_path_len, assumed_risk_image_filled = pathfind(reward_graphs, raw_risk_image, assumed_risk_image, ltl_state_diag, ltl_heuristic, (start_ltl_state, final_ltl_state), (mission_phys_start, mission_phys_finish))

    # pathfind without any risk
    # path, min_path_len, assumed_risk_image_filled = pathfind(reward_graphs, raw_risk_image, raw_risk_image, ltl_state_diag, ltl_heuristic, (start_ltl_state, final_ltl_state), (mission_phys_start, mission_phys_finish))

    # pathfinding on assumed risk without updating
    # path, min_path_len, assumed_risk_image_filled = pathfind(reward_graphs, assumed_risk_image, assumed_risk_image, ltl_state_diag, ltl_heuristic, (start_ltl_state, final_ltl_state), (mission_phys_start, mission_phys_finish))

    # path len is the length of the actual final path taken
    print("path len:: ", len(path))
    # min len is the triangle distance sum[(x2-x1)+(y2-y1)]
    print("min  len:: ", min_path_len)

    # draw the path on img_cell to show the end user
    dj_path_image = create_final_image(processed_img, raw_risk_image, assumed_risk_image_filled, path, (mission_phys_start, mission_phys_finish))
    cv2.imwrite(f"{ output_images_dir }/!picfinal.bmp", cv2.cvtColor(dj_path_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()

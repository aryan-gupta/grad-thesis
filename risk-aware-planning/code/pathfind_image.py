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
UPDATE_WEIGHT = 0 #5

# final image dimensions (must be divisiable by CELL_SIZE)
map_h = 640
map_w = 576



def read_process_image():
    # read image and show it
    img = img_process.read_image('./sample.jpg', show=False)

    # perspective warp the image so its a top down view
    points = [[1025, 132], [855, 2702], [3337, 2722], [2974, 165]]
    wpcc_img = img_process.perspective_warp(img, points, map_w, map_h, show=False)

    # seperate the image into the color segments to create binary images of each color
    (red_channel, green_channel, blue_channel, yellow_channel) = img_process.color_segment_image(wpcc_img, show=False)

    # merge the color channels into one fully saturated and colorified image
    processed_img = img_process.merge_colors(red_channel, green_channel, blue_channel, yellow_channel, show=False)

    # create an image of all the goals and objectives
    raw_reward_image = cv2.add(cv2.add(red_channel, blue_channel), yellow_channel)

    return processed_img, raw_reward_image, green_channel


def create_reward_graphs(processed_img, raw_reward_image, raw_risk_image):
    # create cells based off of map and risk and assign costs to cells
    img_cells, cell_type, cell_cost = cell_process.create_cells(processed_img, raw_risk_image, CELLS_SIZE, show=False)

    # convert cells to seperate objectives and goals (@TODO find a better way to do this)
    cell_type = cell_process.convert_cells(cell_type, objectives=["A", "B"], goals=["S", "F"])

    # get start and finish locations from cell graph
    global_start, global_finish = cell_process.get_start_finish_locations(cell_type)

    # get reward map for each objectives and goals
    reward_graphs = img_process.get_reward_images(cell_type, raw_reward_image, CELLS_SIZE, show=False)

    return img_cells, reward_graphs, (global_start, global_finish)


def parse_ltl_hoa_file():
    # parse through LTL automata
    return ltl_process.parse_ltl_hoa("ltl.hoa.txt")


def get_assumed_risk(raw_risk_image):
    # create blurred risk image
    return img_process.create_risk_img(raw_risk_image, 64, show=False)


def pathfind_no_sensing_rage(reward_graphs, assumed_risk_image, ltl_state_diag, ltl_state_bounds, mission_phys_bounds):
    current_ltl_state = ltl_state_bounds[0]
    start_phys_loc = mission_phys_bounds[0]
    next_phys_loc = mission_phys_bounds[1]

    total_shortest_path = []

    # the loop to traverse the LTL formula
    while current_ltl_state != ltl_state_bounds[1]:
        # get reward map of current LTL state
        current_ltl_state_reward_graph = ltl_process.get_reward_img_state(ltl_state_diag, current_ltl_state, reward_graphs, (map_h, map_w))

        # merge risk, reward into a single image map
        risk_reward_image = cv2.merge([current_ltl_state_reward_graph, assumed_risk_image, current_ltl_state_reward_graph])

        # Convert risk_reward_image into cells
        risk_reward_img_cells, risk_reward_cell_type, risk_reward_cell_cost = cell_process.create_cells(risk_reward_image, assumed_risk_image, CELLS_SIZE, show=False)

        risk_reward_cell_type = cell_process.convert_cells(risk_reward_cell_type, objectives=["A", "B"], goals=["S", "F"])

        # convert cells to state diagram so we can apply dj's algo to it
        state_diagram, state_dict = cell_process.cells_to_state_diagram(risk_reward_cell_type, risk_reward_cell_cost, show=False)

        # get start and finish locations for this ltl node
        _, next_phys_loc = cell_process.get_start_finish_locations(risk_reward_cell_type)

        # apply dj's algo
        shortest_path = dijkstra.dj_algo(risk_reward_img_cells, risk_reward_cell_type, (start_phys_loc, next_phys_loc), state_diagram, CELLS_SIZE)

        # find next state that we should go to
        next_ltl_state = ltl_process.get_next_state(ltl_state_diag, reward_graphs, current_ltl_state, next_phys_loc, CELLS_SIZE)

        # setup next interation
        current_ltl_state = next_ltl_state
        start_phys_loc = next_phys_loc
        total_shortest_path[0:0] = shortest_path


    return total_shortest_path, assumed_risk_image


def pathfind_updateing_risk(reward_graphs, raw_risk_image, assumed_risk_image, ltl_state_diag, ltl_state_bounds, mission_phys_bounds, show=True):
    # get the start conditions
    current_ltl_state = ltl_state_bounds[0]
    current_phys_loc = mission_phys_bounds[0]

    # the shortest path of the entire mission
    total_shortest_path = []

    # our local copy of risk from using our viewing range
    assumed_risk_image_filled = assumed_risk_image.copy()
    
    # any risk of 0 has min risk 5
    # assumed_risk_image_filled = cv2.normalize(assumed_risk_image_filled, None, 254, 10, norm_type = cv2.NORM_MINMAX)

    # the loop to traverse the LTL formula
    img_tmp_idx_ltl = 0
    while current_ltl_state != ltl_state_bounds[1]:
        # get reward map of current LTL state
        current_ltl_state_reward_graph = ltl_process.get_reward_img_state(ltl_state_diag, current_ltl_state, reward_graphs, (map_h, map_w))

        # we wont know the next_phys_loc or the dj's target location until we run our algo
        next_phys_loc = (-1,-1)
    
        # the loop to traverse the phys enviroment
        img_tmp_idx_phys=0
        while current_phys_loc != next_phys_loc:
            # add current node to path
            total_shortest_path.insert(0, current_phys_loc)

            # update risk map everytime we move
            assumed_risk_image_filled = img_process.update_local_risk_image(assumed_risk_image_filled, raw_risk_image, current_phys_loc, CELLS_SIZE, VIEW_CELLS_SIZE, UPDATE_WEIGHT)
            # print("risk"); plt.imshow(assumed_risk_image_filled); plt.show()
            
            # reapply DJ's algo using new risk map
            # create required data structures
            risk_reward_image_local = cv2.merge([current_ltl_state_reward_graph, assumed_risk_image_filled, current_ltl_state_reward_graph])
            risk_reward_img_cells_local, risk_reward_cell_type_local, risk_reward_cell_cost_local = cell_process.create_cells(risk_reward_image_local, assumed_risk_image_filled, CELLS_SIZE, show=False)
            # since this img_cells image was created by using reward images and not the raw_processed_img, we dont need to run cell_process.cells_to_state_diagram
            state_diagram_local, state_dict_local = cell_process.cells_to_state_diagram(risk_reward_cell_type_local, risk_reward_cell_cost_local, show=False)
            
            # get target
            _, next_phys_loc = cell_process.get_start_finish_locations(risk_reward_cell_type_local)

            # apply dj's algo
            shortest_path = dijkstra.dj_algo(risk_reward_img_cells_local, risk_reward_cell_type_local, (current_phys_loc, next_phys_loc), state_diagram_local, CELLS_SIZE)

            if show:
                # draw our current future path on an image
                dj_path_image_local = risk_reward_img_cells_local.copy()
                dijkstra.draw_path_global(shortest_path, dj_path_image_local, (current_phys_loc, next_phys_loc), CELLS_SIZE)

                # draw the agent as a circle
                half_cell = math.ceil((CELLS_SIZE/2))
                center = (current_phys_loc[0]*CELLS_SIZE+half_cell, current_phys_loc[1]*CELLS_SIZE+half_cell)
                dj_path_image_local = cv2.circle(dj_path_image_local, center, 4, (255, 255, 255), 1)

                # save the image
                print(img_tmp_idx_ltl, "-", img_tmp_idx_phys, " :: ", current_phys_loc)
                plt.imshow(dj_path_image_local); plt.savefig(f"/tmp/thesis/pic{ img_tmp_idx_ltl }-{ img_tmp_idx_phys }.png")
                img_tmp_idx_phys += 1

            # get the next location in the shortest path
            current_phys_loc = dijkstra.get_next_cell_shortest_path(shortest_path, current_phys_loc)

        # find next state that we should go to
        next_ltl_state = ltl_process.get_next_state(ltl_state_diag, reward_graphs, current_ltl_state, next_phys_loc, CELLS_SIZE)
        img_tmp_idx_ltl += 1

        # setup next interation
        current_ltl_state = next_ltl_state
        current_phys_loc = next_phys_loc
    
    return total_shortest_path, assumed_risk_image_filled


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
    dj_path_image, _, _ = cell_process.create_cells(dj_path_image, assumed_risk_image_filled, CELLS_SIZE, show=False)

    # draw the path on img_cell
    dijkstra.draw_path_global(path, dj_path_image, mission_phys_bounds, CELLS_SIZE)

    return dj_path_image
    

def main():
    # read in and process image
    processed_img, raw_reward_image, raw_risk_image = read_process_image()

    # create our axiom reward graphs
    processed_img_cells, reward_graphs, (mission_phys_start, mission_phys_finish) = create_reward_graphs(processed_img, raw_reward_image, raw_risk_image)
    
    # get the ltl formula
    ltl_state_diag, aps, start_ltl_state, final_ltl_state = parse_ltl_hoa_file()

    # create our assumed risk image
    assumed_risk_image = get_assumed_risk(raw_risk_image)

    # pathfind the image
    # path, assumed_risk_image_filled = pathfind_no_sensing_rage(reward_graphs, assumed_risk_image, ltl_state_diag, (start_ltl_state, final_ltl_state), (mission_phys_start, mission_phys_finish))
    path, assumed_risk_image_filled = pathfind_updateing_risk(reward_graphs, raw_risk_image, assumed_risk_image, ltl_state_diag, (start_ltl_state, final_ltl_state), (mission_phys_start, mission_phys_finish))

    # draw the path on img_cell to show the end user
    dj_path_image = create_final_image(processed_img, raw_risk_image, assumed_risk_image_filled, path, (mission_phys_start, mission_phys_finish)) 
    print("final"); plt.imshow(dj_path_image); plt.show()


if __name__ == "__main__":
    main()

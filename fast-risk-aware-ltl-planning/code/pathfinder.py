import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import bisect
import PIL as pil
from enum import Enum


import main
import img
import cell
import task
import dijkstra
import env
import random
import optimizer

# This simple enum tells how to output the intermediary images
class OutputType(Enum):
    NONE       = 0 # do not output anything
    DISK       = 1 # output images to the disk
    SHOW       = 2 # show image window to user
    SHOW_FINAL = 3 # only show final image window to user


# This class holds information about one single pathfinding instance
# this class holds the high level algorithm for this paper
class Pathfinder:
    def __init__(self, e, t):
        self.env = e
        self.task = t

        # get the start conditions
        self.current_ltl_state = self.task.task_bounds[0]
        self.current_phys_loc = self.env.mission_phys_bounds[0]

        # the shortest path of the entire mission
        self.total_shortest_path = []

        # our local copy of risk from using our viewing range
        self.assumed_risk_image_filled = self.env.r.assumed_risk_image.copy()

        # any risk of 0 has min risk 5
        # self.assumed_risk_image_filled = cv2.normalize(self.assumed_risk_image_filled, None, 254, 10, norm_type = cv2.NORM_MINMAX)

        # how to show the results
        self.output = OutputType.DISK

        #  the index for ltl steps for the image output
        self.img_tmp_idx_ltl = 0

        # the index for phys steps for the image output
        self.img_tmp_idx_phys = 0


    def pathfind_task(self, start_task_node=None, final_task_node=None):
        # set the start node as the start task node
        if start_task_node is None:
            start_task_node = self.task.task_bounds[0]

        # set the final node as the final task node
        if final_task_node is None:
            final_task_node = self.task.task_bounds[1]

        # reset the counter:
        self.img_tmp_idx_ltl = 0

        # create the cells needed
        img_cells, env_min = self.env.create_cells_internal(self.assumed_risk_image_filled)

        # start the ltl pathfinding loop
        self.current_ltl_state = start_task_node
        while self.current_ltl_state != final_task_node:
            # get reward locations
            # @TODO move self.env.reward_graphs to task class

            # locate all the potential reward locations
            target_locations = self.task.get_reward_locations(self.current_ltl_state, self.env.reward_locations)

            # pick best one based off of task heuristic
            target_phys_loc = self.env.pick_best_target_location(target_locations, self.task, self.current_ltl_state, self.current_phys_loc)

            # self.pathfind_phys(
            #     start_phys_loc=self.current_phys_loc,
            #     final_phys_loc=target_phys_loc
            # )

            print(target_phys_loc)
            self.pathfind_until_final_loc(
                target_phys_loc,
                img_cells,
                env_min
            )

            # find next state that we should go to and setup next interation
            self.current_ltl_state = self.task.get_next_state(self.env.reward_graphs, self.current_ltl_state, self.current_phys_loc)
            self.img_tmp_idx_ltl += 1


    def pathfind_env(self, start_phys_loc=None, final_phys_loc=None):
        if final_phys_loc == None:
            pass

        if start_phys_loc == None:
            start_phys_loc = self.current_phys_loc

        # This is needed so we can do partial replans
        current_planned_path = []

        # reset the counter:
        self.img_tmp_idx_phys = 0

        while self.current_phys_loc != final_phys_loc:
            pass

    # pathfinds from the physical environment
    # starts from the \param self.current_phys_loc to the final_phys_loc
    # @TODO instead of passing in current_ltl_state_reward_graph, pass in a list of the cell locations it can go to
    # @TODO This function, in theory, should move the agent from the current loc to the next loc that would minimize the LTL jumps
    def pathfind_until_final_loc(self, final_phys_loc, risk_reward_img_cells_local, env_min):
        show = False

        # This is needed so we can do partial replans
        current_planned_path = []

        # reset the counter:
        self.img_tmp_idx_phys = 0

        # the loop to traverse the phys enviroment
        while self.current_phys_loc != final_phys_loc:
            # update risk map everytime we move
            self.assumed_risk_image_filled, amount_risk_updated, cells_updated = img.update_local_risk_image(self.assumed_risk_image_filled, self.env.r.raw_risk_image, self.current_phys_loc, main.CELLS_SIZE, main.VIEW_CELLS_SIZE, main.UPDATE_WEIGHT)

            # the temp target for partial astar algorithm
            astar_target = None

            # instead of recreating out required data structures, just update the ones we "saw"
            # these are the same calls as full_replan except update_cells instead of create_cells
            risk_reward_img_cells_local, env_min.cell_type, env_min.cell_cost = self.env.update_cells(cells_updated, self.env.raw_reward_image, self.assumed_risk_image_filled, env_min.cell_type, env_min.cell_cost, risk_reward_img_cells_local, self.current_phys_loc)

            if show: print(amount_risk_updated)

            if amount_risk_updated > 0:
                if show: print("full astar replanning")
                opt = optimizer.Optimizer(env_min, self.task)
                opt.set_task_state(0, self.current_ltl_state)
                current_planned_path = dijkstra.astar_algo(risk_reward_img_cells_local, env_min.cell_type, (self.current_phys_loc, final_phys_loc), env_min.cell_cost, main.CELLS_SIZE)
            elif amount_risk_updated > 0:
                if show: print("part astar replanning")

                # get astar's target cell
                # this target cell will be somewhere on the current_planned_path line
                # idx is the index of the astar_target cell
                astar_target, idx = dijkstra.get_astar_target(self.current_phys_loc, current_planned_path, main.VIEW_CELLS_SIZE * 2)

                # get new path from current loc to astar_target
                shortest_path_astar_target = dijkstra.astar_algo_partial_target(risk_reward_img_cells_local, env_min.cell_type, (self.current_phys_loc, astar_target), final_phys_loc, env_min.cell_cost, main.CELLS_SIZE)

                # splice our two shortest_paths together
                current_planned_path = current_planned_path[0:idx]
                current_planned_path = current_planned_path + shortest_path_astar_target
                if show: print(len(current_planned_path))

            # add current node to path
            self.total_shortest_path.insert(0, self.current_phys_loc)

            # get the next location in the shortest path
            self.current_phys_loc = dijkstra.get_next_cell_shortest_path(current_planned_path, self.current_phys_loc)
            self.output_current_state(risk_reward_img_cells_local, current_planned_path, final_phys_loc, astar_target)

            # increment our image file counter
            self.img_tmp_idx_phys += 1


    # outputs the current state of the pathfinding class
    # @TODO remove img_cells and put it in here. img_cells is an expensive thing to calculate so I do not want it in the main algo part of the code
    def output_current_state(self, risk_reward_img_cells_local, current_planned_path, final_phys_loc, astar_target):
        if self.output is OutputType.NONE:
            return

        # draw our taken path and future path on an image
        dj_path_image_local = risk_reward_img_cells_local.copy()
        dijkstra.draw_path_global(self.total_shortest_path, dj_path_image_local, (self.total_shortest_path[-1], self.total_shortest_path[0]), main.CELLS_SIZE)
        dijkstra.draw_path_global(current_planned_path, dj_path_image_local, (self.current_phys_loc, final_phys_loc), main.CELLS_SIZE)

        # draw the agent as a circle
        half_cell = math.ceil((main.CELLS_SIZE/2))
        center = (self.current_phys_loc[0]*main.CELLS_SIZE+half_cell, self.current_phys_loc[1]*main.CELLS_SIZE+half_cell)
        dj_path_image_local = cv2.circle(dj_path_image_local, center, 4, (255, 255, 255), 1)

        # draw the partial target is there is one
        if astar_target != None:
            center = (astar_target[0]*main.CELLS_SIZE+half_cell, astar_target[1]*main.CELLS_SIZE+half_cell)
            dj_path_image_local = cv2.circle(dj_path_image_local, center, 4, (255, 255, 255), 1)

            # reset out astar_target so if we fully plan next loop, the circle doesnt get added
            astar_target = None

        # save the image
        # print(img_tmp_idx_ltl, "-", img_tmp_idx_phys, " :: ", self.current_phys_loc, "(", amount_risk_updated, ")")
        img_tmp_idx_ltl_str  = str(self.img_tmp_idx_ltl).zfill(2)
        img_tmp_idx_phys_str = str(self.img_tmp_idx_phys).zfill(3)



        if self.output is OutputType.DISK:
            cv2.imwrite(f"{ main.output_images_dir }/pic{ img_tmp_idx_ltl_str }-{ img_tmp_idx_phys_str }.png", cv2.cvtColor(dj_path_image_local, cv2.COLOR_RGB2BGR) )


    # returns the shortest path the algo has determined
    def get_total_shortest_path(self):
        return self.total_shortest_path


    # returns the filled in risk map after the agent has traveled around the environment
    def get_filled_assumed_risk(self):
        return self.assumed_risk_image_filled

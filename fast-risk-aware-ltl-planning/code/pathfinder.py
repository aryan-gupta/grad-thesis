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


class OutputType(Enum):
    NONE       = 0 # do not output anything
    DISK       = 1 # output images to the disk
    SHOW       = 2 # show image window to user
    SHOW_FINAL = 3 # only show final image window to user



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

    # pathfinds until the set task node is reached
    def pathfind_until_task(self, task_node=None):
        # set the final node as the final task node
        if task_node is None:
            task_node = self.task.task_bounds[1]

        # reset the counter:
        self.img_tmp_idx_ltl = 0

        # the loop to traverse the LTL formula
        while self.current_ltl_state != task_node:
            # get reward map of current LTL state
            current_ltl_state_reward_graph = self.task.get_reward_img_state(self.current_ltl_state, self.env.reward_graphs)

            self.pathfind_until_final_loc(current_ltl_state_reward_graph)

            # find next state that we should go to and setup next interation
            self.current_ltl_state = self.task.get_next_state(self.env.reward_graphs, self.current_ltl_state, self.current_phys_loc)
            self.img_tmp_idx_ltl += 1


    def pathfind_until_final_loc(self, current_ltl_state_reward_graph):
        show = False
        # we wont know the final_phys_loc or the dj's target location until we run our algo
        # store a dummy result in the meantime
        final_phys_loc = (-1,-1)

        # This is needed so we can do partial replans
        current_planned_path = []
        risk_reward_image_local = None
        risk_reward_img_cells_local = None
        risk_reward_cell_type_local = None
        risk_reward_cell_cost_local = None


        # reset the counter:
        self.img_tmp_idx_phys = 0

        # the loop to traverse the phys enviroment
        while self.current_phys_loc != final_phys_loc:
            # add current node to path
            self.total_shortest_path.insert(0, self.current_phys_loc)

            # update risk map everytime we move
            self.assumed_risk_image_filled, amount_risk_updated, cells_updated = img.update_local_risk_image(self.assumed_risk_image_filled, self.env.r.raw_risk_image, self.current_phys_loc, main.CELLS_SIZE, main.VIEW_CELLS_SIZE, main.UPDATE_WEIGHT)

            # the temp target for partial astar algorithm
            astar_target = None

            # only do a full replan if its our first run
            # if we havent, then update the local data structures and do a partial replan if the risk values have changed
            if risk_reward_image_local is None:
                if show: print("full cells replanning")
                # create required data structures
                risk_reward_image_local = cv2.merge([current_ltl_state_reward_graph, self.assumed_risk_image_filled, np.zeros((main.map_h, main.map_w), np.uint8)])
                risk_reward_img_cells_local, risk_reward_cell_type_local, risk_reward_cell_cost_local = cell.create_cells(risk_reward_image_local, self.assumed_risk_image_filled, main.CELLS_SIZE, show=False)

                # cv2.imwrite(f"{ main.output_images_dir }/paper-tmp.png", cv2.cvtColor(risk_reward_img_cells_local, cv2.COLOR_RGB2BGR) ); exit()

                # get next phys loc based off the LTL state diag
                final_phys_loc = self.task.get_finish_location(risk_reward_cell_type_local, self.env.reward_graphs, self.current_ltl_state)
                # print(final_phys_loc)

                # apply dj's algo
                opt = optimizer.Optimizer(env.Enviroment.get_minimal_env(risk_reward_cell_type_local, risk_reward_cell_cost_local), self.task)
                opt.set_task_state(0, self.current_ltl_state)
                current_planned_path = dijkstra.astar_opt(risk_reward_img_cells_local, risk_reward_cell_type_local, (self.current_phys_loc, final_phys_loc), risk_reward_cell_cost_local, main.CELLS_SIZE, opt)
                if show: print(len(current_planned_path))
            else:
                # instead of recreating out required data structures, just update the ones we "saw"
                # these are the same calls as full_replan except update_cells instead of create_cells
                risk_reward_image_local = cv2.merge([current_ltl_state_reward_graph, self.assumed_risk_image_filled, np.zeros((main.map_h, main.map_w), np.uint8)])
                risk_reward_img_cells_local, risk_reward_cell_type_local, risk_reward_cell_cost_local = cell.update_cells(cells_updated, risk_reward_image_local, risk_reward_cell_type_local, risk_reward_cell_cost_local, risk_reward_img_cells_local, self.current_phys_loc, self.assumed_risk_image_filled, main.CELLS_SIZE, main.VIEW_CELLS_SIZE)

                if show: print(amount_risk_updated)

                if amount_risk_updated > 0:
                    if show: print("full astar replanning")
                    opt = optimizer.Optimizer(env.Enviroment.get_minimal_env(risk_reward_cell_type_local, risk_reward_cell_cost_local), self.task)
                    opt.set_task_state(0, self.current_ltl_state)
                    current_planned_path = dijkstra.astar_algo(risk_reward_img_cells_local, risk_reward_cell_type_local, (self.current_phys_loc, final_phys_loc), risk_reward_cell_cost_local, main.CELLS_SIZE)
                elif amount_risk_updated > 0:
                    if show: print("part astar replanning")

                    # get astar's target cell
                    # this target cell will be somewhere on the current_planned_path line
                    # idx is the index of the astar_target cell
                    astar_target, idx = dijkstra.get_astar_target(self.current_phys_loc, current_planned_path, main.VIEW_CELLS_SIZE * 2)

                    # get new path from current loc to astar_target
                    shortest_path_astar_target = dijkstra.astar_algo_partial_target(risk_reward_img_cells_local, risk_reward_cell_type_local, (self.current_phys_loc, astar_target), final_phys_loc, risk_reward_cell_cost_local, main.CELLS_SIZE)

                    # splice our two shortest_paths together
                    current_planned_path = current_planned_path[0:idx]
                    current_planned_path = current_planned_path + shortest_path_astar_target
                    if show: print(len(current_planned_path))

            # get the next location in the shortest path
            self.current_phys_loc = dijkstra.get_next_cell_shortest_path(current_planned_path, self.current_phys_loc)
            self.output_current_state(risk_reward_img_cells_local, current_planned_path, final_phys_loc, astar_target)

            # increment our image file counter
            self.img_tmp_idx_phys += 1


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


    def get_total_shortest_path(self):
        return self.total_shortest_path


    def get_filled_assumed_risk(self):
        return self.assumed_risk_image_filled

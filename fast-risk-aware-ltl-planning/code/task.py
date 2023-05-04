import main
import numpy as np
import cv2
import math
import sys
import matplotlib.pyplot as plt

import main

# stores an LTL task
class Task:
    def __init__(self, filename=None, ltl_task=None):
        # use spot to convert ltl task string into a graph
        if ltl_task is not None:
            pass # @TODO
        # use an HOA file. HOA files can be created here: https://spot.lre.epita.fr/app/
        elif filename is not None:
            self.parse_ltl_hoa(filename)
        else:
            raise Exception("Task not specified. Please use HOA file or pass an LTL task string into Task class")


    # creates the task heuristics
    def create_task_heuristic(self, env=None):
        # create our basic LTL heuristic model
        self.create_node_distance_heuristic()
        self.create_euclidean_heuristic(env)


    # runs a search algorithm on the task graph
    # labels each node with the shortest path to the accepting state
    # this allows the algorithm to know which next nodes leads to the
    # accepting state the quickest
    def create_node_distance_heuristic(self):
        # ltw[start][finish] = weight
        # ltl_transiton_weights = {{}}

        # lnw[node] = weight
        final_node = self.task_bounds[1]
        self.ltl_heuristic = { final_node : 0}
        queue = [final_node]
        distance = 1

        while len(queue) != 0:
            # get first element
            current_node = queue[0]
            queue = queue[1:]

            for node in self.ltl_state_diag.keys():
                for next_node in self.ltl_state_diag[node].keys():
                    trans = self.ltl_state_diag[node][next_node]
                    axioms = trans.split('&')

                    cond = 0
                    for axiom in axioms:
                        if axiom[0] != '!':
                            cond += 1

                    if next_node == current_node and cond == 1:
                        if node not in self.ltl_heuristic.keys():
                            queue.append(node)
                            self.ltl_heuristic[node] = distance
                        elif self.ltl_heuristic[node] > distance:
                            self.ltl_heuristic[node] = distance
            # print(queue)
            distance += 1

        # print(self.ltl_heuristic)
        # ltl_heuristic_aps = {}
        # for key in self.ltl_heuristic.keys():
        #     ltl_heuristic_aps[aps[key]] = self.ltl_heuristic[key]

        # print(ltl_heuristic_aps)


    # runs a search algorithm on the task graph
    # extracts all paths from the start node to the accepting state
    # node. Then uses those paths to determine the total euclidean
    # distance of that path. See presentation in git repo for more info
    def create_euclidean_heuristic(self, env=None):
        self.paths = []
        self.__create_euclidean_heuristic_recurse([], self.task_bounds[0], main.START_CELL_CHAR.lower(), 0)
        if env != None: self.update_euclidean_heuristic_w_env(env)


    # a recursive helper function for create_euclidean_heuristic function
    def __create_euclidean_heuristic_recurse(self, path, node, target, depth):
        # append the current node into the current path
        path.append((node, target))

        # base condition
        # if we are 5 layers deep, exit it would take too long to calculate
        # also if we are at the accepting state then add the current path
        # into the paths. This path will be used later to calculate the
        # euclidean distances for each possible path
        if depth > 5 or node == self.task_bounds[1]:
            self.paths.append(path)
            return


        # rerun this recursive algo for each path in the task graph
        # skipping self loops, !targets, and multiple ap targets
        # @TODO figure out how to handle not loops
        next_nodes = self.ltl_state_diag[node]
        for n in next_nodes:
            if n == node: continue # skip self loops

            ap = next_nodes[n] # get ap to satisfy
            ap = ap.split('&') # split into indivisual targets
            ap = [ x for x in ap if '!' not in x] # remove not targets
            if len(ap) > 1: continue # skip targets with multiple aps to satisfy

            next_target = ap[0]
            self.__create_euclidean_heuristic_recurse(path.copy(), n, next_target, depth + 1)


    # calculates the euclidean distance of each path in the task automata
    def update_euclidean_heuristic_w_env(self, env):
        self.euclidean_heuristic = []


        def find(cell_type, char):
            for ycell in range(len(cell_type)):
                for xcell in range(len(cell_type[ycell])):
                        if cell_type[ycell][xcell] == char:
                            return (ycell, xcell)
            return (None, None)

                # figure out the euclidean distances of each path

        for path in self.paths:
            euclidean_distance = 0
            for idx in range(len(path) - 1):
                node, target = path[idx]
                nnode, ntarget = path[idx + 1]

                y, x = find(env.cell_type, target.upper())
                ny, nx = find(env.cell_type, ntarget.upper())

                dx = x - nx
                dy = y - ny
                distance = math.sqrt((dx**2) + (dy**2))

                euclidean_distance += distance
            self.euclidean_heuristic.append((euclidean_distance, path))

        self.euclidean_heuristic.sort(key=lambda a: a[0])

        # print(self.euclidean_heuristic)
        # exit()


    # parse an ltl HOA formatted file
    def parse_ltl_hoa(self, filename, show=False):
        # The ltl graph is a dict{ current_state: dict{ next_state : str(AP) } }
        self.ltl_state_diag = {}
        self.aps = []
        state = -1
        final_state = -1
        start_state = -1
        next_state_dict = None
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()

                if line.startswith("Start:"):
                    start_state = int(line.split(" ")[1])

                if line.startswith("AP:"):
                    self.aps = line.replace("\"", "").split(" ")[2:]

                if line.startswith("State:"):
                    # we are finished parsing the previous state, add it to the master ltl dict
                    if next_state_dict is not None and state != -1:
                        self.ltl_state_diag[state] = next_state_dict
                        next_state_dict = {}
                    state = int(line.split(" ")[1])
                    next_state_dict = {}
                    if final_state == -1 and len(line.split(" ")) >= 3 and line.split(" ")[2] == "{0}":
                        final_state = state

                if line.startswith("["):
                    splits = line.split(" ", maxsplit=1)
                    next_state = int(splits[1])
                    ap_temp = splits[0].replace("[", "").replace("]", "")
                    for ap_num in range(len(self.aps)):
                        ap_temp = ap_temp.replace(str(ap_num), self.aps[ap_num])
                    next_state_dict[next_state] = ap_temp

        if next_state_dict is not None and state != -1:
            self.ltl_state_diag[state] = next_state_dict
            next_state_dict = {}

        if show:
            print(self.ltl_state_diag)
            print(start_state)
            print(final_state)

        self.task_bounds = (start_state, final_state)


    # @TODO
    # checks if the axiom allows any jump to the next LTL state from the
    # \p current_ltl_state
    def check_valid_jump(self, current_ltl_state, axiom):
        return True


    # returns the next target locations
    def get_reward_locations(self, current_state, reward_locations):
        potential_reward_location = set([])

        for next_state in self.ltl_state_diag[current_state].keys():
            this_state_reward_graph = None
            axon = self.ltl_state_diag[current_state][next_state].upper()
            nomials = axon.split('&')

            for nomial in nomials:
                if nomial[0] != '!':
                    if this_state_reward_graph is None:
                        this_state_reward_graph = set(reward_locations[nomial[0]])
                    else:
                        this_state_reward_graph = this_state_reward_graph.intersection(reward_locations[nomial[0]])
                else:
                    if this_state_reward_graph is None:
                        # @TODO place all cells in potential reward locations and then take union or intersection
                        pass
                    else:
                        this_state_reward_graph = this_state_reward_graph.difference(reward_locations[nomial[1]])


            if this_state_reward_graph:
                potential_reward_location = potential_reward_location.union(this_state_reward_graph)

        return potential_reward_location


    # get the reward image based off the possible transitions from the current state
    def get_reward_img_state(self, current_state, reward_graphs):
        # get the image for each transition from the current state
        map_h, map_w = (main.map_h, main.map_w)
        ltl_reward_graph = np.zeros((map_h, map_w, 1), dtype = "uint8")
        for next_state in self.ltl_state_diag[current_state].keys():
            this_state_reward_graph = np.full((map_h, map_w, 1), 255, dtype = "uint8")
            axon = self.ltl_state_diag[current_state][next_state].upper()
            nomials = axon.split('&')
            valid = False
            for nomial in nomials:
                if nomial[0] != '!':
                    this_state_reward_graph = cv2.bitwise_and(this_state_reward_graph, reward_graphs[nomial[0]])
                    valid = True
            # plt.imshow(this_state_reward_graph); plt.show()
            if valid:
                ltl_reward_graph = cv2.bitwise_or(ltl_reward_graph, this_state_reward_graph)

        # plt.imshow(ltl_reward_graph); plt.show()
        return ltl_reward_graph


    # get the next state of the ltl buchii automata
    def get_next_state(self, reward_graphs, current_ltl_state, current_phys_loc):
        next_state = None
        for next_state in self.ltl_state_diag[current_ltl_state]:
            current_cell_type = get_current_phys_state_type(reward_graphs, current_phys_loc)
            axioms = self.ltl_state_diag[current_ltl_state][next_state].upper()
            axioms = axioms.split('&')

            valid = True
            for axiom in axioms:
                if axiom[0] == '!':
                    if axiom[1] in current_cell_type:
                        valid = False
                else:
                    if axiom[0] not in current_cell_type:
                        valid = False
            # plt.imshow(this_state_reward_graph); plt.show()
            if valid:
                break

        return next_state


    def get_optimization_state(self, cell_type, current_ltl_state, current_phys_loc):
        next_state = None
        for next_state in self.ltl_state_diag[current_ltl_state]:
            current_cell_type = cell_type[current_phys_loc[1]][current_phys_loc[0]]
            axioms = self.ltl_state_diag[current_ltl_state][next_state].upper()
            axioms = axioms.split('&')

            valid = True
            for axiom in axioms:
                if axiom[0] == '!':
                    if axiom[1] in current_cell_type:
                        valid = False
                else:
                    if axiom[0] not in current_cell_type:
                        valid = False
            # plt.imshow(this_state_reward_graph); plt.show()
            if valid:
                break

        return next_state

    # gets the locations of the next physical locations that would cause
    # a LTL task jump
    def get_finish_location(self, cell_type, reward_graphs, current_ltl_state):
        jump_cost = sys.maxsize
        next_state = None
        for next_possible_state in self.ltl_state_diag[current_ltl_state].keys():
            trans = self.ltl_state_diag[current_ltl_state][next_possible_state]
            axioms = trans.split('&')

            cond = 0
            for axiom in axioms:
                if axiom[0] != '!':
                    cond += 1

            # if cond == 1 and self.ltl_heuristic[next_possible_state] <= jump_cost:
            if cond == 1 and self.ltl_heuristic[next_possible_state] < jump_cost:
                next_state = next_possible_state
                jump_cost = self.ltl_heuristic[next_possible_state]

        print(next_state)

        # get maximization of what axiom leads to next_state
        axioms = self.ltl_state_diag[current_ltl_state][next_state]
        axiom_list = axioms.split('&')
        axiom = None
        for a in axiom_list:
            if a[0] != '!':
                axiom = a.upper()

        print(axiom)

        # return the location of that axiom using T and reward graphs
        for row in range(len(cell_type)):
            for col in range(len(cell_type[row])):
                y = row * main.CELLS_SIZE + (main.CELLS_SIZE//2)
                x = col * main.CELLS_SIZE + (main.CELLS_SIZE//2)

                pixel_valid = reward_graphs[axiom][y][x] != 0
                if cell_type[row][col] == main.LTL_TARGET_CELL_CHAR and pixel_valid:
                    return (col, row)


# get the axiom the current physical state is activating
def get_current_phys_state_type(reward_graphs, current_phys_loc):
    for axiom in reward_graphs.keys():
        y = current_phys_loc[1] * main.CELLS_SIZE
        x = current_phys_loc[0] * main.CELLS_SIZE

        for u in range(y, y + main.CELLS_SIZE, 1):
            for v in range(x, x + main.CELLS_SIZE, 1):
                if reward_graphs[axiom][u,v]:
                    return axiom

    print("Illegal, didnt want to throw")
    return main.LTL_TARGET_CELL_CHAR

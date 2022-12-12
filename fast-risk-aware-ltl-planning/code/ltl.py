import numpy as np
import cv2
import math
import sys
# import matplotlib.pyplot as plt


class Task:
    def __init__(self, filename):
        self.parse_ltl_hoa(filename)


    # @todo use markov desicison process table instead of dictionary
    def create_task_heuristic(self):
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
                    if len(line.split(" ")) >= 3 and line.split(" ")[2] == "{0}":
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

    @staticmethod
    def check_jump(target, trans):
        axioms = trans.split('&')
        target = target.lower()

        valid = 0
        for axiom in axioms:
            # if we jump needs a not but our input has it, jump is not valid
            if   axiom[0] == '!' and target == axiom[1]:
                return -1
            elif axiom[0] == '!' and target != axiom[1]:
                valid += 1

            # if the jump doesnt have the right input but needs it for the transition, its not valid
            elif target != axiom[0]:
                return -1
            elif target == axiom[0]:
                valid += 1

        return valid


    def check_valid_jump(self, current_ltl_state, axiom):
        # print(current_ltl_state, axiom)

        valid = False
        for target in self.ltl_state_diag[current_ltl_state].keys():
            num = Task.check_jump(axiom, self.ltl_state_diag[current_ltl_state][target])
            # print(self.ltl_state_diag[current_ltl_state][target]," :: " , num)
            valid |= (num >= 0)

        # print(valid)

        return valid

# get the reward image based off the possible transitions from the current state
def get_reward_img_state(ltl_state_diag, current_state, reward_graphs, size):
    # get the image for each transition from the current state
    map_h, map_w = size
    ltl_reward_graph = np.zeros((map_h, map_w, 1), dtype = "uint8")
    for next_state in ltl_state_diag[current_state].keys():
        this_state_reward_graph = np.full((map_h, map_w, 1), 255, dtype = "uint8")
        axon = ltl_state_diag[current_state][next_state].upper()
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


# get the axiom the current physical state is activating
def get_current_phys_state_type(reward_graphs, current_phys_loc, CELLS_SIZE):
    for axiom in reward_graphs.keys():
        y = current_phys_loc[1] * CELLS_SIZE
        x = current_phys_loc[0] * CELLS_SIZE

        for u in range(y, y + CELLS_SIZE, 1):
            for v in range(x, x + CELLS_SIZE, 1):
                if reward_graphs[axiom][u,v]:
                    return axiom

    print("Illegal, didnt want to throw")
    return 'T'


# get the next state of the ltl buchii automata
def get_next_state(ltl_state_diag, reward_graphs, current_ltl_state, current_phys_loc, CELLS_SIZE):
    next_state = None
    for next_state in ltl_state_diag[current_ltl_state]:
        current_cell_type = get_current_phys_state_type(reward_graphs, current_phys_loc, CELLS_SIZE)
        axioms = ltl_state_diag[current_ltl_state][next_state].upper()
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


def get_finish_location(cell_type, ltl_state_diag, ltl_heuristic, reward_graphs, current_ltl_state, CELLS_SIZE):
    jump_cost = sys.maxsize
    next_state = None
    for next_possible_state in ltl_state_diag[current_ltl_state].keys():
        trans = ltl_state_diag[current_ltl_state][next_possible_state]
        axioms = trans.split('&')

        cond = 0
        for axiom in axioms:
            if axiom[0] != '!':
                cond += 1

        # if cond == 1 and ltl_heuristic[next_possible_state] <= jump_cost:
        if cond == 1 and ltl_heuristic[next_possible_state] < jump_cost:
            next_state = next_possible_state
            jump_cost = ltl_heuristic[next_possible_state]

    # print(next_state)

    # get maximization of what axiom leads to next_state
    axioms = ltl_state_diag[current_ltl_state][next_state]
    axiom_list = axioms.split('&')
    axiom = None
    for a in axiom_list:
        if a[0] != '!':
            axiom = a.upper()

    # print(axiom)

    # return the location of that axiom using T and reward graphs
    for row in range(len(cell_type)):
        for col in range(len(cell_type[row])):
            y = row * CELLS_SIZE + (CELLS_SIZE//2)
            x = col * CELLS_SIZE + (CELLS_SIZE//2)

            pixel_valid = reward_graphs[axiom][y][x] != 0
            if cell_type[row][col] == 'T' and pixel_valid:
                return (col, row)

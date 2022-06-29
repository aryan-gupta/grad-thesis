import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def parse_ltl_hoa(filename):
    # The ltl graph is a dict{ current_state: dict{ next_state : str(AP) } }
    ltl_state_diag = {}
    aps = []
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
                aps = line.replace("\"", "").split(" ")[2:]

            if line.startswith("State:"):
                # we are finished parsing the previous state, add it to the master ltl dict
                if next_state_dict is not None and state != -1:
                    ltl_state_diag[state] = next_state_dict
                    next_state_dict = {}
                state = int(line.split(" ")[1])
                next_state_dict = {}
                if len(line.split(" ")) >= 3 and line.split(" ")[2] == "{0}":
                    final_state = state
            
            if line.startswith("["):
                splits = line.split(" ", maxsplit=1)
                next_state = int(splits[1])
                ap_temp = splits[0].replace("[", "").replace("]", "")
                for ap_num in range(len(aps)):
                    ap_temp = ap_temp.replace(str(ap_num), aps[ap_num])
                next_state_dict[next_state] = ap_temp

    if next_state_dict is not None and state != -1:
        ltl_state_diag[state] = next_state_dict
        next_state_dict = {}

    # print(ltl_state_diag)
    # print(start_state)
    # print(final_state)

    return ltl_state_diag, start_state, final_state


def get_reward_img_state(ltl_state_diag, current_state, reward_graphs, size):
    # get the image for each transition from the current state
    map_h, map_w = size
    ltl_reward_graph = np.zeros((map_h, map_w, 1), dtype = "uint8")
    for next_state in ltl_state_diag[current_state].keys():
        this_state_reward_graph = np.full((map_h, map_w, 1), 255, dtype = "uint8")
        axon = ltl_state_diag[current_state][next_state].upper()
        nomials = axon.split('&')
        print(nomials)
        valid = False
        for nomial in nomials:
            if nomial[0] != '!':
                this_state_reward_graph = cv2.bitwise_and(this_state_reward_graph, reward_graphs[nomial[0]])
                valid = True
        # plt.imshow(this_state_reward_graph); plt.show()
        if valid:
            ltl_reward_graph = cv2.bitwise_or(ltl_reward_graph, this_state_reward_graph)

    plt.imshow(ltl_reward_graph); plt.show()
    return ltl_reward_graph
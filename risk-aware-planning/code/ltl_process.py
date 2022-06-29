

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
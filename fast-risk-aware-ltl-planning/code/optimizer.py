

# This class checks weather a environment move by the agent
# is valid against the LTL task. Prunes the search tree so
# the astar/djk algo completes quicker
class Optimizer:
    def __init__(self, e=None, t=None):
        self.tasks = []
        self.task_state = []
        self.env = e

        self.tasks.append(t)
        self.task_state.append(t.task_bounds[0])


    # adds a task into the list of tasks the agent must follow
    def add_task(self, t):
        self.tasks.append(t)
        self.task_state.append(t.task_bounds[0])


    # adds the environment the agent is going through
    def add_env(self, e):
        self.env = e

    # sets the task state for the specified task
    # @TODO refactor this function out
    def set_task_state(self, task_num, state):
        self.task_state[task_num] = state

    # checks if the direction the agent wants to go is valid
    # if its not valid it will foll of the edge of the map and
    # this function will return false
    def is_valid_direction(self, current_phys_loc, direction):
        x, y = current_phys_loc

        if direction == 0:
            if y > 0:
                return True

        if direction == 1:
            if x > 0:
                return True

        if direction == 2:
            if x < (len(self.env.cell_cost[0]) - 1):
                return True

        if direction == 3:
            if y < (len(self.env.cell_cost) - 1):
                return True

        return False


    # checks if the direction the agent wants to go is forbidden by any task
    # if it is forbidden, the function will return false
    def is_direction_forbidden_by_task(self, current_phys_loc, direction):
        x, y = current_phys_loc

        if direction == 0: y -= 1
        if direction == 1: x -= 1
        if direction == 2: x += 1
        if direction == 3: y += 1

        global_valid = True
        for task_num in range(len(self.tasks)):
            # get what value the direction optimizes
            axiom = self.env.cell_type[y][x]
            # check is that value is allowed by ltl
            current_ltl_state = self.task_state[task_num]
            valid = self.tasks[task_num].check_valid_jump(current_ltl_state, axiom)

            global_valid &= valid

        return not global_valid

import math
import cv2
import random
import matplotlib.pyplot as plt
import numpy as np

import main
import risk
import img
import cell
import env

L = math.pi # length of the box
N = 30 # number of grid cells along each axis
Ns = 3 # number of samples
uConst = 0.3 # mean wind in x direction (u-velocity)
vConst = -0.4 # mean wind in y direction (v-velocity)

class WindEnvironmentCreator:
    def __init__(self, targets=2):
        self.create_random(targets)
        self.create_risk()
        self.reward_graphs, self.reward_locations = img.get_reward_images(self.cell_type, self.raw_reward_image, main.CELLS_SIZE, show=False)

        empty_channel = np.zeros(self.r.assumed_risk_image.shape, np.uint8)
        image = cv2.merge([self.raw_reward_image, self.r.assumed_risk_image, empty_channel])
        self.ar_img_cells, _, _ = cell.create_cells(self.raw_reward_image, self.r.assumed_risk_image, image, show=False)
        # plt.imshow(self.raw_reward_image)
        # plt.show()

        # self.__show_env_cell_wind_conv()
        # exit()

    def create_risk(self):
        npa = np.array(self.cell_cost)
        # plt.imshow(npa, origin='lower')
        # plt.show()
        raw_risk_image = cv2.resize(npa, dsize=(0, 0), fx=main.CELLS_SIZE, fy=main.CELLS_SIZE, interpolation=cv2.INTER_AREA)
        # plt.imshow(raw_risk_image, origin='lower')
        # plt.show()
        # exit()
        self.r = risk.Risk(raw_risk_image)
        self.r.create_empty_assumed_risk()


    # updates a list of cells
    def update_cells(self, cells_updated, ltl_reward_map, assumed_risk_image, cell_type, cell_cost, img_cells, phys_loc):
        img_cells, cell_type, cell_cost = cell.update_cells(cells_updated, ltl_reward_map, cell_type, cell_cost, img_cells, phys_loc, assumed_risk_image)
        # npa = np.array(img_cells)
        # plt.imshow(npa, origin='lower')
        # plt.show()
        return img_cells, cell_type, cell_cost


    # def create_reward(self):
    #     npa = np.array(self.cell_type)
    #     rewards = set(main.CHAR_COLOR_MAP.values())
    #     for y in range(len(npa)):
    #         for x in range(len(npa[y])):
    #             if npa[y][x] in rewards:
    #                 raw_reward_image_pre = main.CHAR_COLOR_MAP[npa[y][x]]
    #             else:

    # returns a minimal environment with the assumed risk.
    # this is a replacement for the static get_minimal_env method
    def get_ar_minimal_env(self):
        env_min = env.EnviromentMinimal()
        env_min.cell_type = self.ar_cell_type
        env_min.cell_cost = self.ar_cell_cost

        return env_min

    def create_random(self, targets):
        self.num_targets = targets
        # interval = 2*L/(N-1)
        # @NOTE in matlab the meshgrid function params are start:interval:stop
        # however in python the meshgrid function params are start, stop, num_ticks
        # so in python calculating the interval is unnecessary
        # see https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        # see https://www.mathworks.com/help/matlab/ref/meshgrid.html
        # this below link isnt necessary, but leaving it in here
        # see https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        self.X, self.Y = np.meshgrid(np.linspace(-L, L, N), np.linspace(-L, L, N))
        self.U = np.sin(self.Y / 2.0) + uConst
        self.V = np.cos(self.X) + vConst

        # My code needs a 2D array to work on
        # this converts the 4 array X, Y, U, V into 2D array of vectors
        # not 100% sure how meshgrid works but this works for some reason so :shrug:
        # @TODO learn what meshgrid is, looks important to learn
        self.convert_cell_wind()
        self.create_cell_type()
        self.__add_ltl_targets()
        # print(self.cell_type)
        # print(self.cell_cost)


    def create_cell_type(self):
        self.cell_type = []

        for ycell in range(len(self.U)):
            self.cell_type.append([])
            for xcell in range(len(self.U[ycell])):

                border = False
                border = border or (ycell == 0)
                border = border or (ycell == len(self.U) - 1)
                border = border or (xcell == 0)
                border = border or (xcell == len(self.U[ycell]) - 1)

                if border:
                    self.cell_type[ycell].append(main.HAZARD_CELL_CHAR)
                else:
                    self.cell_type[ycell].append(main.EMPTY_CELL_CHAR)


    def __try_draw_cell(self, points, color, size):
        x, y = points

        if y >= self.cell_height: return False
        if x >= self.cell_width: return False
        if self.cell_type[y][x] == main.HAZARD_CELL_CHAR: return False

        self.cell_type[y][x] = main.CHAR_COLOR_MAP[color]
        # self.processed_img = cv2.rectangle(self.processed_img, (x,y), (x + size,y + size), (color,0,0), -1)

        return True


    def __add_ltl_targets(self):
        # ltl_target_colors_red = {
        #     'target' : 250,
        #     'start'  : 225, <-- start here
        #     'finish' : 200,
        # }

        color = 225
        start = None
        finish = None
        raw_reward_image_pre = np.zeros(np.array(self.cell_type).shape, np.uint8)

        for i in range(self.num_targets):
            cell_drawn = False

            while not cell_drawn:
                x = random.randrange(0, self.cell_width)
                y = random.randrange(0, self.cell_height)

                cell_drawn = self.__try_draw_cell((x, y), color, 16)

            if main.CHAR_COLOR_MAP[color] == main.START_CELL_CHAR:
                start = (x, y)

            if main.CHAR_COLOR_MAP[color] == main.END_CELL_CHAR:
                finish = (x, y)

            raw_reward_image_pre[y][x] = color

            color -= 25

        self.raw_reward_image = cv2.resize(raw_reward_image_pre, dsize=(0, 0), fx=main.CELLS_SIZE, fy=main.CELLS_SIZE, interpolation=cv2.INTER_AREA)
        self.mission_phys_bounds = (start, finish)


    def convert_cell_wind(self):
        self.cell_wind = []

        for ycell in range(len(self.U)):
            self.cell_wind.append([])
            for xcell in range(len(self.U[ycell])):
                self.cell_wind[ycell].append((self.U[ycell][xcell], self.V[ycell][xcell]))

        self.cell_height = len(self.cell_wind)
        self.cell_width  = len(self.cell_wind[0])

        self.convert_cell_cost()


    def convert_cell_cost(self):
        self.cell_cost = []

        for ycell in range(len(self.U)):
            self.cell_cost.append([])
            for xcell in range(len(self.U[ycell])):
                x, y = self.cell_wind[ycell][xcell]
                cost = math.sqrt( x**2 + y**2 ) * 127
                self.cell_cost[ycell].append(cost)


    def show_env(self):
        plt.quiver(self.X, self.Y, self.U, self.V)
        plt.show()
        # self.__show_env_cell_wind_conv()


    def __show_env_cell_wind_conv(self):
        nparray = np.array(self.cell_cost)
        # https://stackoverflow.com/questions/42776583
        plt.imshow(nparray, origin='lower')
        # plt.show()

        # @TODO draw rectangles for the hazard locations and targets

        U = []
        V = []
        for ycell in range(len(self.cell_wind)):
            U.append([])
            V.append([])

            for xcell in range(len(self.cell_wind[ycell])):

                y_val, x_val = self.cell_wind[ycell][xcell]

                U[ycell].append(y_val)
                V[ycell].append(x_val)

        plt.quiver(U, V)
        plt.show()


    def create_cells_ar(self):
        # @TODO do we need this to be a copy?
        self.ar_cell_type = self.cell_type

        self.ar_cell_cost = []
        for y in range(len(self.cell_cost)):
            self.ar_cell_cost.append(self.cell_cost[y].copy())
            for x in range(len(self.ar_cell_cost[y])):
                self.ar_cell_cost[y][x] = 0.0

    # choses the next best location based off of the environment and task heuristic
    # @TODO move this to the pathfinder class
    def pick_best_target_location(self, reward_locations, task, ltl_state, phys_loc):
        # figure out which one is the smallest
        # @TODO
        min_idx = 0

        # figure out where in the path we are and get the next target
        target = None
        for idx in range(len(task.euclidean_heuristic[min_idx][1])):
            s, _ = task.euclidean_heuristic[min_idx][1][idx]
            if s == ltl_state:
                _, target = task.euclidean_heuristic[min_idx][1][idx + 1]

        # figure out which location to go to next
        # checks if
        # - location is the target we found out in previous step
        # - location is the closest cell to us euclidean dist wise
        target_loc = (0, 0)
        min_dist = float("inf")
        for xcell, ycell in reward_locations:
            if self.cell_type[ycell][xcell] == target.upper():

                dx = xcell - phys_loc[0]
                dy = ycell - phys_loc[1]
                distance = math.sqrt((dx**2) + (dy**2))

                if distance < min_dist:
                    target_loc = (xcell, ycell)
                    min_dist = distance

        return target_loc

# creates and saves a random environment
def create_save_random_environment(seed):
    random.seed(seed)
    e = WindEnvironmentCreator(targets=4)
    e.show_env()
    # e.save_env(f"../../../maps/000.bmp")


if __name__ == "__main__":
    create_save_random_environment(0)

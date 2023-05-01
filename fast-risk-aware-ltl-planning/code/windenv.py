import math
import random
import matplotlib.pyplot as plt
import numpy as np

import main

L = math.pi # length of the box
N = 30 # number of grid cells along each axis
Ns = 3 # number of samples
uConst = 0.3 # mean wind in x direction (u-velocity)
vConst = -0.4 # mean wind in y direction (v-velocity)

class WindEnvironmentCreator:
    def __init__(self, targets=2):
        self.create_random(targets)

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
        for i in range(self.num_targets):
            cell_drawn = False

            while not cell_drawn:
                x = random.randrange(0, self.cell_width)
                y = random.randrange(0, self.cell_height)

                cell_drawn = self.__try_draw_cell((x, y), color, 16)

            color -= 25

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
                cost = math.sqrt( x**2 + y**2 )
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


# creates and saves a random environment
def create_save_random_environment(seed):
    random.seed(seed)
    e = WindEnvironmentCreator(targets=4)
    e.show_env()
    # e.save_env(f"../../../maps/000.bmp")


if __name__ == "__main__":
    create_save_random_environment(0)

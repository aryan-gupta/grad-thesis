import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

import main
import risk
import img
import cell
import dijkstra
from vector2 import vector2

CELLS_SIZE = 8 # 32 pixels

TEMP_XTRA_TARGETS = [ vector2(5, 8), vector2(15, 3) ]

# this class creates a blank environment from scratch
class EnviromentCreator:
    def __init__(self, targets=2, size=(800, 800), validate=False):
        while self.create_random(targets, size) and validate:
            pass
        self.xtra_targets = TEMP_XTRA_TARGETS


    def create_random(self, targets, size):
        # self.width, self.height = size
        self.height, self.width = size
        self.num_targets = targets
        self.processed_img = None

        self.processed_img = np.zeros((self.height, self.width, 3), np.uint8)

        self.__add_circular_risk()
        self.__add_ltl_targets()
        self.__add_small_circular_risk()

        return self.verify_valid_env()

    # https://stackoverflow.com/questions/4047935/
    def preprocess(self):
        self.__class__ = Enviroment
        self.__init__(filename=None)
        return self


    def __add_circular_risk(self):
        # large circles
        for i in range(int(random.randrange(0, 10))):
            # determine size of circle
            circle_size = int(random.gauss(64, 16))
            # determine location
            x = random.randrange(0, self.width)
            y = random.randrange(0, self.height)
            # add risk
            self.processed_img = cv2.circle(self.processed_img, (x, y), circle_size, (0, 255, 0), -1)


        # medium circles
        for i in range(int(random.randrange(30, 40))):
            # determine size of circle
            circle_size = int(random.gauss(18, 5))
            # determine location
            x = random.randrange(0, self.width)
            y = random.randrange(0, self.height)
            # add risk
            self.processed_img = cv2.circle(self.processed_img, (x, y), circle_size, (0, 255, 0), -1)


    def __try_draw_cell(self, points, color, size):
        x, y = points

        for yp in range(y, y+16):
            if yp >= self.height: return False
            for xp in range(x, x+16):

                if xp >= self.width: return False

                if self.processed_img[yp, xp][1] == 255:
                    return False

        self.processed_img = cv2.rectangle(self.processed_img, (x,y), (x + size,y + size), (color,0,0), -1)

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
                x = random.randrange(0, self.width)
                y = random.randrange(0, self.height)

                cell_drawn = self.__try_draw_cell((x, y), color, 16)

            color -= 25


    def __try_draw_circle(self):
        # determine size of circle
        circle_size = 4
        # determine location
        x = random.randrange(0, self.width)
        y = random.randrange(0, self.height)

        for yp in range(y-circle_size, y+circle_size):
            if yp >= self.height: return False

            for xp in range(x-circle_size, x+circle_size):
                if xp >= self.width: return False

                if self.processed_img[yp, xp][0] != 0:
                    return False

        # add risk
        self.processed_img = cv2.circle(self.processed_img, (x, y), circle_size, (0, 255, 0), -1)

        return True


    def __add_small_circular_risk(self):
        # small circles
        for i in range(int(random.randrange(200, 300))):
            circle_drawn = False
            while not circle_drawn:
                circle_drawn = self.__try_draw_circle()


    def verify_valid_env(self):
        return False


    def show_env(self):
        plt.imshow(self.processed_img)
        plt.show()


    def save_env(self, filename):
        env_bgr = cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, env_bgr)


# this class is a place holder to store an minimal environment
# this will be deprecated in the future, hopefully
class EnviromentMinimal:
    def __init__(self):
        self.cell_cost = None
        self.cell_type = None

        self.cell_img = None


# this class stores the environment
class Enviroment(EnviromentCreator):
    def __init__(self, filename=None):
        self.processed_img = None
        self.height = None
        self.width = None
        self.num_targets = None
        self.raw_reward_image = None
        self.r = None


        # load enviroment file
        if filename is not None:
            self.load_env(filename)

        # split the env into R (targets), G (Risk), B (Unused)
        self.channel_split()

        # create our axiom reward graphs
        # convert this to get_reward_locations (store the reward locations as coorinates, not as image graphs)
        self.create_reward_graphs()

        # create our assumed risk image
        self.create_assumed_risk()


    # loads in an image by filename, and checks the size of the
    # image to be the same as the global variables
    def load_env(self, filename):
        self.processed_img = cv2.imread(filename)
        self.processed_img = cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR)

        self.height, self.width, _ = self.processed_img.shape

        assert self.height == main.map_h, f"Image height not same as main.map_h"
        assert self.width == main.map_w, f"Image height not same as main.map_h"


    # splits the images into the RGB channels
    def channel_split(self):
        self.raw_reward_image, raw_risk_image, _ = cv2.split(self.processed_img)
        self.r = risk.Risk(raw_risk_image)


    # creates all the reward graphs for each axiom
    def create_reward_graphs(self):
        # create cells based off of map and risk and assign costs to cells
        self.img_cells, self.cell_type, self.cell_cost = cell.create_cells(self.raw_reward_image, self.r.raw_risk_image, self.processed_img, show=False)

        # get start and finish locations from cell graph
        self.mission_phys_bounds = cell.get_start_finish_locations(self.cell_type)

        # get reward map for each objectives and goals
        self.reward_graphs, self.reward_locations = img.get_reward_images(self.cell_type, self.raw_reward_image, CELLS_SIZE, show=False)


    # creates the assumed risk by applying a blurring to it
    def create_assumed_risk(self):
        self.r.create_assumed_risk()


    # creates the final image to output
    def create_final_image(self, filename, assumed_risk_image_filled, path):
        # seperate the image into RGB channels
        _, _, blue_channel = cv2.split(self.processed_img)

        # add our filled out assumed risk
        combined_risk_image = cv2.add(self.r.raw_risk_image, assumed_risk_image_filled)

        # make the actual walls white so its easy to tell apart from the green assumed risk surroundings
        # red_channel = cv2.add(red_channel, self.r.raw_risk_image)
        # green_channel = cv2.add(green_channel, self.r.raw_risk_image)
        # blue_channel = cv2.add(blue_channel, self.r.raw_risk_image)

        image = cv2.merge([self.raw_reward_image,combined_risk_image,blue_channel])

        # create our img_cell
        dj_path_image, _, _ = cell.create_cells(self.raw_reward_image, assumed_risk_image_filled, image, show=False)

        # draw the path on img_cell
        dijkstra.draw_path_global(path, dj_path_image, self.mission_phys_bounds, CELLS_SIZE)

        cv2.imwrite(filename, cv2.cvtColor(dj_path_image, cv2.COLOR_RGB2BGR))


    # returns a minimal environment. Used to pass it to the optimizer
    # @TODO deprecate this function by passing the entire environment class to the optimizer
    @staticmethod
    def get_minimal_env(cell_type, cell_cost):
        env_min = EnviromentMinimal()
        env_min.cell_type = cell_type
        env_min.cell_cost = cell_cost

        return env_min

    def get_ar_minimal_env(self):
        env_min = EnviromentMinimal()
        env_min.cell_type = self.ar_cell_type
        env_min.cell_cost = self.ar_cell_cost

        return env_min

    # create the cells on the environment
    # @TODO modify this function so its only run once in pathfinder, rather than everytime a LTL path jump takes place
    # @TODO deprecated
    def dep_create_cells(self, ltl_reward_map, assumed_risk_image):
        empty_channel = np.zeros((main.map_h, main.map_w), np.uint8)
        # create required data structures
        image = cv2.merge([ltl_reward_map, assumed_risk_image, empty_channel])
        img_cells, cell_type, cell_cost = cell.create_cells(ltl_reward_map, assumed_risk_image, image, show=False)

        return img_cells, self.get_minimal_env(cell_type, cell_cost)


    def create_cells_internal(self, assumed_risk_image):
        empty_channel = np.zeros((main.map_h, main.map_w), np.uint8)
        image = cv2.merge([self.raw_reward_image, assumed_risk_image, empty_channel])

        self.ar_img_cells, self.ar_cell_type, self.ar_cell_cost = cell.create_cells(self.raw_reward_image, assumed_risk_image, image, show=False)

    # updates a list of cells
    def update_cells(self, cells_updated, ltl_reward_map, assumed_risk_image, cell_type, cell_cost, img_cells, phys_loc):
        img_cells, cell_type, cell_cost = cell.update_cells(cells_updated, ltl_reward_map, cell_type, cell_cost, img_cells, phys_loc, assumed_risk_image)

        return img_cells, cell_type, cell_cost


    # @TODO move img_cells to this function so that the profiler doesnt have to run it during the timing
    def create_cell_image(r, g, b):
        pass


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
    e = EnviromentCreator(targets=4, size=(800,800), validate=False)
    e.show_env()
    # e.save_env(f"../../../maps/000.bmp")


if __name__ == "__main__":
    create_save_random_environment(0)

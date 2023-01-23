import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

import risk
import img
import cell
import dijkstra
from vector2 import vector2

CELLS_SIZE = 8 # 32 pixels

TEMP_XTRA_TARGETS = [ vector2(5, 8), vector2(15, 3) ]

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
        self.__init__(self)
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



class Enviroment(EnviromentCreator):
    def __init__(self, ec, filename=None):
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


    def load_env(self, filename):
        pass


    def channel_split(self):
        self.raw_reward_image, raw_risk_image, _ = cv2.split(self.processed_img)
        self.r = risk.Risk(raw_risk_image)


    # creates all the reward graphs for each axiom
    def create_reward_graphs(self):
        # create cells based off of map and risk and assign costs to cells
        self.img_cells, self.cell_type, self.cell_cost = cell.create_cells(self.processed_img, self.r.raw_risk_image, CELLS_SIZE, show=False)

        # get start and finish locations from cell graph
        self.mission_phys_bounds = cell.get_start_finish_locations(self.cell_type)

        # get reward map for each objectives and goals
        self.reward_graphs = img.get_reward_images(self.cell_type, self.raw_reward_image, CELLS_SIZE, show=False)


    def create_assumed_risk(self):
        self.r.create_assumed_risk()


    # creates the final image to output
    def create_final_image(self, filename, assumed_risk_image_filled, path):
        # seperate the image into RGB channels
        red_channel, green_channel, blue_channel = cv2.split(self.processed_img)

        # add our filled out assumed risk
        green_channel = cv2.add(green_channel, assumed_risk_image_filled)

        # make the actual walls white so its easy to tell apart from the green assumed risk surroundings
        red_channel = cv2.add(red_channel, self.r.raw_risk_image)
        green_channel = cv2.add(green_channel, self.r.raw_risk_image)
        blue_channel = cv2.add(blue_channel, self.r.raw_risk_image)

        # merge back our image into a single image
        dj_path_image = cv2.merge([red_channel, green_channel, blue_channel])

        # create our img_cell
        dj_path_image, _, _ = cell.create_cells(dj_path_image, assumed_risk_image_filled, CELLS_SIZE, show=False)

        # draw the path on img_cell
        dijkstra.draw_path_global(path, dj_path_image, self.mission_phys_bounds, CELLS_SIZE)

        cv2.imwrite(filename, cv2.cvtColor(dj_path_image, cv2.COLOR_RGB2BGR))


def main():
    random.seed(0)
    e = Enviroment(targets=4, size=(800,800), validate=False)
    e.show_env()
    e.save_env(f"../../../maps/000.bmp")


if __name__ == "__main__":
    main()

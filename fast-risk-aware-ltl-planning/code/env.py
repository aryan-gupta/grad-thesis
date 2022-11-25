import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Enviroment:
    def __init__(self, targets=2, size=(800, 800), validate=False, filename=None):
        if filename is None:
            while self.create_random(targets, size) and validate:
                pass
        else:
            self.load_env(filename)


    def load_env(self, filename):
        pass


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


def main():
    random.seed(0)
    e = Enviroment(targets=4, size=(800,800), validate=False)
    e.show_env()
    e.save_env(f"../../../maps/000.bmp")


if __name__ == "__main__":
    main()

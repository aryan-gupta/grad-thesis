import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

map_h = 640
map_w = 576

def add_circular_risk(img):
    # large circles
    for i in range(3):
        # determine size of circle
        circle_size = int(random.gauss(64, 16))
        # determine location
        x = random.randrange(0, map_w)
        y = random.randrange(0, map_h)
        # add risk
        img = cv2.circle(img, (x, y), circle_size, (0, 255, 0), -1)


    # medium circles
    for i in range(25):
        # determine size of circle
        circle_size = int(random.gauss(18, 5))
        # determine location
        x = random.randrange(0, map_w)
        y = random.randrange(0, map_h)
        # add risk
        img = cv2.circle(img, (x, y), circle_size, (0, 255, 0), -1)

    # # small circles
    for i in range(200):
        # determine size of circle
        circle_size = 4
        # determine location
        x = random.randrange(0, map_w)
        y = random.randrange(0, map_h)
        # add risk
        img = cv2.circle(img, (x, y), circle_size, (0, 255, 0), -1)


    return img

def try_draw_cell(img, points, color, size):
    x, y = points
    for yp in range(y, y+16):
        if yp >= map_h: return False
        for xp in range(x, x+16):

            if xp >= map_w: return False

            if img[yp, xp][1] == 255:
                return False

    img = cv2.rectangle(img, (x,y), (x + size,y + size), (color,0,0), -1)
    return True


def add_ltl_targets(num_ltl_target, img):
    # ltl_target_colors_red = {
    #     'target' : 250,
    #     'start'  : 225, <-- start here
    #     'finish' : 200,
    # }

    color = 225
    for i in range(num_ltl_target):
        cell_drawn = False

        while not cell_drawn:
            x = random.randrange(0, map_w)
            y = random.randrange(0, map_h)

            cell_drawn = try_draw_cell(img, (x, y), color, 16)

        color -=25

    return img


def verify_valid_env(img):
    return True


def create_env(num_ltl_target, size):
    valid = False
    map_w, map_h = size
    img = None

    while not valid:
        img = np.zeros((map_h,map_w,3), np.uint8)
        img = add_circular_risk(img)
        img = add_ltl_targets(num_ltl_target + 2, img)
        valid = verify_valid_env(img)

    return img


def main():
    random.seed()
    num_ltl_target = 2
    env = create_env(num_ltl_target, (map_w, map_h))
    plt.imshow(env); plt.show()
    env = cv2.cvtColor(env, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"../../../maps/001.bmp", env)


if __name__ == "__main__":
    main()

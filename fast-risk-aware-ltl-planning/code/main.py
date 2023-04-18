import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import bisect
import PIL as pil


import img
import cell
import task
import dijkstra
import env
import random
import optimizer
import pathfinder

# GLOBAL VARS
# stores the size of each cell or square in the environment
CELLS_SIZE = 8 # 32 pixels

# stores the agent's viewing distance. As the agent moves around,
# it can see \p VIEW_CELLS_SIZE distance away in each direction.
# the viewing circle diameter is `2*VIEW_CELLS_SIZE`
VIEW_CELLS_SIZE = 8

# [deprecated], will be removed in a future commit
UPDATE_WEIGHT = 0 #5

# final image dimensions (must be divisiable by CELLS_SIZE)
map_h = 640
map_w = 576

# directory the progress images, images can then be combined with
# `ffmpeg -framerate 5 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mkv`
output_images_dir = '../../../tmp'
tmp_raw_env_save_file = f"{output_images_dir}/raw_env.png"

# location of where the final image should go. This image has all
# the cells, uncovered risk, assumed risk (if any), path that the
# agent traveled, and LTL targets
final_image = f"{ output_images_dir }/!picfinal.png"

# input for the LTL hoa file, @TODO will become a array to support
# multiple HOA files
ltl_hoa_file = '../tasks/basic-ab.hoa.txt'

# the environment file the agent is in. This file must be a
# - RGB (R - LTL Targets, G - Environment Risk, B - Unused)
# - PNG (to prevent JPEG aliasing and artifacts)
enviroment_file = '../maps/002.png'

CREATE_NEW_ENVIRONMENT = False
PATHFIND_NO_ASSUMED_RISK = False
PATHFIND_IGNORE_RISK_UPDATES = False

# CHAR REPRESENTATIONS
# char representation of a hazard cell or wall cell
HAZARD_CELL_CHAR = 'X'

# char representation of a empty traversable cell
EMPTY_CELL_CHAR = '#'

# chars representing the different targets
START_CELL_CHAR = 'A'
LTL_TARGET_CELL_CHAR = 'Y'
END_CELL_CHAR = 'Z'
CHAR_COLOR_MAP = {
    250 : LTL_TARGET_CELL_CHAR,
    225 : START_CELL_CHAR,
    200 : END_CELL_CHAR,
    175 : 'B',
    150 : 'C',
    125 : 'D',
    100 : 'E',
     75 : 'F',
     50 : 'G',
     25 : 'H'
}

def main():
    # since the seed is 0, the env will always be the same, helps when debugging
    random.seed(1)

    e = None
    if CREATE_NEW_ENVIRONMENT:
        e = env.EnviromentCreator(targets=6, size=(map_h,map_w), validate=False)
        e.save_env(tmp_raw_env_save_file)
        e = e.preprocess()
    else:
        e = env.Enviroment(filename=enviroment_file)

    # img.save_channel_image("../maps/assumed_risk.png", g=e.r.assumed_risk_image)

    # pathfind without any risk
    if PATHFIND_NO_ASSUMED_RISK: e.r.assumed_risk_image = e.r.raw_risk_image

    # pathfinding on assumed risk without updating
    if PATHFIND_IGNORE_RISK_UPDATES: e.r.raw_risk_image = e.r.assumed_risk_image

    # create the cells needed
    e.create_cells_internal(e.r.assumed_risk_image)

    # get the task details using LTL
    t = task.Task(ltl_hoa_file)

    p = pathfinder.Pathfinder(e, t)
    p.pathfind_task()

    # draw the path on img_cell to show the end user
    e.create_final_image(final_image, p.get_filled_assumed_risk(), p.get_total_shortest_path())

if __name__ == "__main__":
    main()

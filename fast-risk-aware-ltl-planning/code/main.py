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
CELLS_SIZE = 8 # 32 pixels
VIEW_CELLS_SIZE = 8
UPDATE_WEIGHT = 0 #5

# final image dimensions (must be divisiable by CELLS_SIZE)
map_h = 485
map_w = 485

output_images_dir = '../../../tmp'
ltl_hoa_file = 'ltl.hoa.txt'


HAZARD_CELL_CHAR = 'X'
EMPTY_CELL_CHAR = ' '

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
    random.seed(0)

    # read in and process image
    # e = env.EnviromentCreator(targets=4, size=(800,800), validate=False)
    # e.save_env(f"./map.bmp")

    # e = e.preprocess()
    # if you want to use your own image, CAUB (comment above, uncomment below), and change the filename parameter
    e = env.Enviroment(filename='../maps/hospital.png')

    # get the task details using LTL
    t = task.Task(ltl_hoa_file)

    # create our basic LTL heuristic model
    t.create_task_heuristic()

    # pathfind without any risk
    # e.r.assumed_risk_image = e.r.raw_risk_image

    # pathfinding on assumed risk without updating
    # e.r.raw_risk_image = e.r.assumed_risk_image

    p = pathfinder.Pathfinder(e, t)
    p.pathfind_until_task()

    # draw the path on img_cell to show the end user
    e.create_final_image(f"{ output_images_dir }/!picfinal.bmp", p.get_filled_assumed_risk(), p.get_total_shortest_path())


if __name__ == "__main__":
    main()

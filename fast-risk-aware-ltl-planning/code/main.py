import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import bisect
import PIL as pil

from timeit import default_timer as timer

import global_vars as gv
import img
import cell
import task
import dijkstra
import env
import random
import optimizer
import pathfinder
import windenv

def preprocessor():
    e = None
    if gv.CREATE_NEW_ENVIRONMENT:
        e = env.EnviromentCreator(targets=6, size=(gv.map_h,gv.map_w), validate=False)
        e.save_env(gv.tmp_raw_env_save_file)
        e = e.preprocess()
    else:
        e = env.Enviroment(filename=gv.enviroment_file)

    # img.save_channel_image("../maps/assumed_risk.png", g=e.r.assumed_risk_image)

    # pathfind without any risk
    if gv.PATHFIND_NO_ASSUMED_RISK: e.r.assumed_risk_image = e.r.raw_risk_image

    # pathfinding on assumed risk without updating
    if gv.PATHFIND_IGNORE_RISK_UPDATES: e.r.raw_risk_image = e.r.assumed_risk_image

    # create the cells needed
    e.create_cells_ar(e.r.assumed_risk_image)

    # get the task details using LTL
    t = task.Task(gv.ltl_hoa_file)

    t.create_task_heuristic(e)

    p = pathfinder.Pathfinder(e, t)

    return e, t, p


def main():
    start_housekeeping = timer()

    # since the seed is 0, the env will always be the same, helps when debugging
    random.seed(gv.SEED)

    end_housekeeping = timer()

    start_preprocessing = timer()
    # preprocess all necessary data
    e, t, p = preprocessor()
    end_preprocessing = timer()

    start_processing = timer()
    # run the pathfinding algorithm
    p.pathfind_task()
    end_processing = timer()

    start_postprocessing = timer()
    # draw the path on img_cell to show the end user
    e.create_final_image(gv.final_image_fspath, p.get_filled_assumed_risk(), p.get_total_shortest_path())
    end_postprocessing = timer()

    print(p.get_total_shortest_path())
    print(len(p.get_total_shortest_path()))

    elapsed_housekeeping = end_housekeeping - start_housekeeping
    print(f"Housekeeping    took { elapsed_housekeeping } seconds")

    elapsed_preprocessing = end_preprocessing - start_preprocessing
    print(f"Pre -Processing took { elapsed_preprocessing } seconds")

    elapsed_processing = end_processing - start_processing
    print(f"    -Processing took { elapsed_processing } seconds")

    elapsed_postprocessing = end_postprocessing - start_postprocessing
    print(f"Post-Processing took { elapsed_postprocessing } seconds")

    elapsed = end_postprocessing - start_preprocessing
    print(f"All -Processing took { elapsed } seconds")

    return elapsed_processing


def wind_main():
    # since the seed is 0, the env will always be the same, helps when debugging
    random.seed(1)
    e = windenv.WindEnvironmentCreator(targets=4)
    e.create_cells_ar()
    t = task.Task(ltl_hoa_file)
    t.create_task_heuristic(e)
    p = pathfinder.Pathfinder(e, t)

    p.pathfind_task()


if __name__ == "__main__":
    # wind_main()
    # exit()

    sum = 0
    num = 1 # 10

    for i in range(num):
        sum = sum + main()

    avg = sum / num
    print(f"Avg -Processing took { avg } seconds")

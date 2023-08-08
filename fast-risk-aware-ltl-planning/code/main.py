import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import bisect
import PIL as pil
import argparse
import collections.abc

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


def parse_args():
    # https://stackoverflow.com/questions/20063/
    parser = argparse.ArgumentParser(description='Fask Risk Aware LTL Planning')

    # parser.add_argument('--help', action='help', help='Create a new environment')
    parser.add_argument('--seed', help='Seed for random library')
    parser.add_argument('--cell-size', help='Cell size in pixels for the discretization of the environment')
    parser.add_argument('--view-cell-size', help='View cell size in cells for risk updates')
    parser.add_argument('--height', help='Height of the environment')
    parser.add_argument('--width', help='Width of the environment')
    parser.add_argument('--new-env', action='store_true', help='Create a new environment')
    parser.add_argument('--env', help='Input a file to use as the environment')
    parser.add_argument('--no-risk', action='store_true', help='Pathfind without any risk')
    parser.add_argument('--no-assumed-risk', action='store_true', help='Pathfind without any assumed risk. Use the real risk values')
    parser.add_argument('--no-risk-updates', action='store_true', help='Use the assumed risk values, but do not live update with real risk values')
    parser.add_argument('--assumed-risk-live', action='store_true', help='Use the assumed risk values, and live update with real risk values. Default')
    parser.add_argument('--task', action='append', help='Input a file to use as the environment')
    parser.add_argument('--switch-task-idx', action='append', help='Index of step at which the task should be switched')
    parser.add_argument('--switch-task-node', action='append', help='Node number of when the task should switch. In format LTL_Node,PHYS_Node')
    parser.add_argument('--output', help='Output location for intermediary items')
    # parser.add_argument('--', help='Output location for intermediary items')

    return parser.parse_args()

def apply_args(args):
    # a simple function that returns the default if the test value is None
    # some arguments will not be present in the command line arguments and in that case the default must be used
    # Im pretty sure argparse has a feature that does this automatically but I already wrote this code so a future
    # update will fix it @TODO
    def default_if_none(test, default):
        return default if test is None else test

    # set SEED to argument passed if available, keep it the same if user didnt pass one in
    gv.SEED = int(default_if_none(args.seed, gv.SEED))
    #SEED = SEED if args.seed is None else args.seed

    gv.CELLS_SIZE = int(default_if_none(args.cell_size, gv.CELLS_SIZE))

    gv.VIEW_CELLS_SIZE = int(default_if_none(args.view_cell_size, gv.VIEW_CELLS_SIZE) )

    # global map_h,map_w
    gv.map_h = int(default_if_none(args.height, gv.map_h))
    gv.map_w = int(default_if_none(args.width, gv.map_w))

    gv.CREATE_NEW_ENVIRONMENT = default_if_none(args.new_env, gv.CREATE_NEW_ENVIRONMENT)

    gv.enviroment_file = default_if_none(args.env, gv.enviroment_file)

    # @TODO swap the before if and after else statement, read this carefully for logic issue
    # _                               = True if args.no_risk         is None else False
    gv.PATHFIND_NO_ASSUMED_RISK     = default_if_none(args.no_assumed_risk, gv.PATHFIND_NO_ASSUMED_RISK)
    gv.PATHFIND_IGNORE_RISK_UPDATES = default_if_none(args.no_risk_updates, gv.PATHFIND_IGNORE_RISK_UPDATES)
    # _                               = True if args.assumed_risk_live is None else False

    # @TODO convert this variable to an array and fix tasks to it can input multiple tasks
    gv.ltl_hoa_files = default_if_none(args.task, gv.ltl_hoa_files)
    # gv.ltl_hoa_file  = default_if_none(args.task[0], gv.ltl_hoa_file)
    gv.ltl_hoa_file  = gv.ltl_hoa_file if args.task is None else args.task[0]

    # @TODO FIX THIS
    # if output is true then
    # gv.output_type = default_if_none(args.output, None)


def main():
    start_housekeeping = timer()
    args = parse_args()
    apply_args(args)

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

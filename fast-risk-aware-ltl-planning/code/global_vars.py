
# debug var
# 0 - nodebug
# 1 - basic output
# 2 -
DEBUG = 0

# GLOBAL VARS
SEED = 1

# stores the size of each cell or square in the environment
CELLS_SIZE = None # 32 pixels

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
final_image_fspath = f"{ output_images_dir }/!picfinal.png"

# input for the LTL hoa file
ltl_hoa_file =   '../tasks/basic-ab.hoa.txt' # @TODO deprecate this
ltl_hoa_files = ['../tasks/basic-ab.hoa.txt']

# the environment file the agent is in. This file must be a
# - RGB (R - LTL Targets, G - Environment Risk, B - Unused)
# - PNG (to prevent JPEG aliasing and artifacts)
enviroment_file = '../maps/002.png'

output_type = None

CREATE_NEW_ENVIRONMENT = False
PATHFIND_NO_ASSUMED_RISK = True
PATHFIND_IGNORE_RISK_UPDATES = False

PATHFIND_ALGO_PRODUCT_AUTOMATA = False
PATHFIND_ALGO_FRALTLP = True

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

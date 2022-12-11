#!/bin/bash

PYTHON='python3'
EXE='main.py'
ARGS='
--seed 1
--cell-size 8
--view-cell-size 8
--env "../maps/002.png"
--assumed-risk-live
--task "../tasks/basic-ab.hoa.txt"
--output "../../../tmp"
'

cd 'code'
exec $PYTHON $EXE $ARGS
cd '..'

# (46, 10)
# (24, 69)
# (41, 63)
# [(41, 62), (40, 62), (39, 62), (38, 62), (37, 62), (36, 62), (36, 63), (36, 64), (35, 64), (34, 64), (33, 64), (33, 65), (32, 65), (32, 66), (32, 67), (32, 68), (32, 69), (31, 69), (30, 69), (29, 69), (28, 69), (27, 69), (26, 69), (25, 69), (24, 69), (24, 68), (25, 68), (25, 67), (26, 67), (26, 66), (26, 65), (26, 64), (27, 64), (28, 64), (29, 64), (29, 63), (30, 63), (30, 62), (31, 62), (32, 62), (33, 62), (34, 62), (35, 62), (36, 62), (37, 62), (38, 62), (39, 62), (39, 63), (39, 64), (39, 63), (40, 63), (40, 62), (40, 61), (40, 60), (40, 59), (41, 59), (42, 59), (42, 58), (42, 57), (42, 56), (42, 55), (42, 54), (42, 53), (42, 52), (42, 51), (43, 51), (44, 51), (44, 50), (44, 49), (44, 48), (44, 47), (44, 46), (44, 45), (44, 44), (44, 43), (44, 42), (44, 41), (44, 40), (44, 39), (44, 38), (44, 37), (44, 36), (44, 35), (44, 34), (44, 33), (44, 32), (44, 31), (44, 30), (44, 29), (45, 29), (46, 29), (46, 28), (46, 27), (46, 26), (46, 25), (46, 24), (46, 23), (46, 22), (46, 21), (46, 20), (46, 19), (46, 18), (46, 17), (46, 16), (46, 15), (46, 14), (46, 13), (46, 12), (46, 11), (46, 10), (45, 10), (45, 9), (44, 9), (44, 8), (44, 7), (45, 7), (46, 7), (47, 7), (47, 6), (48, 6), (48, 5), (49, 5), (50, 5), (51, 5), (51, 4), (51, 3), (51, 2)]
# 127
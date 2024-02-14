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

exec $PYTHON $EXE $ARGS

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

CELLS_SIZE = 31 # 32 pixels
MAX_WEIGHT = 999999999

img = cv2.imread('./sample.png', cv2.IMREAD_COLOR) 

dim = img.shape
print(dim)

cells_height = math.floor(dim[0] / CELLS_SIZE)
cells_width  = math.floor(dim[1] / CELLS_SIZE)

print((cells_height, cells_width))

colors = []

cell_type = []
cell_num_height = 0
cell_num_width = 0
for y in range(0, dim[0], CELLS_SIZE):
    cell_type.append([])
    cell_num_height = 0

    for x in range(0, dim[1], CELLS_SIZE):
        cell_type[cell_num_width].append('C')

        # draw rectangles
        img2show = cv2.rectangle(img, (x+1,y+1), (x + CELLS_SIZE,y + CELLS_SIZE), (0,0,0), 1)
        
        # determine what the cell is
        cell_known = False
        for u in range(y, y + CELLS_SIZE, 1):
            if u >= dim[0]:
                break
            
            for v in range(x, x + CELLS_SIZE, 1):
                if v >= dim[1]:
                    break

                # keep a record of all the different colors
                if tuple(img[u,v]) not in colors:
                    colors.append(tuple(img[u,v]))
                
                # mark the cells if its corosponding color exists in the cell
                if tuple(img[u,v]) == (128,0,255):
                    cell_known = True
                    img2show = cv2.putText(img2show, 'H',(x+3, y+CELLS_SIZE-3), 2, 1, (0,0,0),1)
                    cell_type[cell_num_width][cell_num_height] = 'H'
                if tuple(img[u,v]) == (255, 128, 0):
                    cell_known = True
                    img2show = cv2.putText(img2show, 'F',(x+3, y+CELLS_SIZE-3), 2, 1, (0,0,0),1)
                    cell_type[cell_num_width][cell_num_height] = 'F'
                if tuple(img[u,v]) == (0, 255, 0):
                    cell_known = True
                    img2show = cv2.putText(img2show, 'S',(x+3, y+CELLS_SIZE-3), 2, 1, (0,0,0),1)
                    cell_type[cell_num_width][cell_num_height] = 'S'
                
            # Exit loop if we know the cell type
            if cell_known:
                break

        # if we dont know the cell type (its all white), mark it as a clean cell
        if not cell_known:    
            img2show = cv2.putText(img2show, 'C',(x+3, y+CELLS_SIZE-3), 2, 1, (0,0,0),1)
        cell_num_height += 1
    cell_num_width += 1

print(colors)

for y in cell_type:
    print(y)

def is_valid_travel_cell(c):
    if c == 'C' or c == 'F' or c == 'S':
        return True
    return False

state_diagram = []
for y in range(len(cell_type)):
    state_diagram.append([])
    for x in range(len(cell_type[0])):
        state_diagram[y].append([MAX_WEIGHT, MAX_WEIGHT, MAX_WEIGHT, MAX_WEIGHT, cell_type[y][x]])
        if cell_type[y][x] == 'H':
            continue
        # check up left
        # NOT IMPL
        
        # check up
        if y > 0 and is_valid_travel_cell(cell_type[y - 1][x]):
            state_diagram[y][x][0] = 1
        # check up right
        # NOT IMPL
        
        # check left
        if x > 0 and is_valid_travel_cell(cell_type[y][x - 1]):
            state_diagram[y][x][1] = 1
        # check right
        if x < (len(cell_type[0]) - 1) and is_valid_travel_cell(cell_type[y][x + 1]):
            state_diagram[y][x][2] = 1
        # check down left
        # NOT IMPL

        # check down
        if y < (len(cell_type) - 1) and is_valid_travel_cell(cell_type[y + 1][x]):
            state_diagram[y][x][3] = 1
        # check down right
        # NOT IMPL

# pretty print state diagram
for row in state_diagram:
    # Up arrows
    for col in row:
        print(" ", end="") # space for the left arrow
        if col[0] != MAX_WEIGHT:
            print("↑", end="")
        else:
            print(" ", end="")
        print(" ", end="") # space for the right arrow
    print()
    # left/right and center char
    for col in row:
        if col[1] != MAX_WEIGHT:
            print("←", end="")
        else:
            print(" ", end="")
        print(col[4], end="")
        if col[2] != MAX_WEIGHT:
            print("→", end="")
        else:
            print(" ", end="")
    print()
    # Down arrows
    for col in row:
        print(" ", end="") # space for the left arrow
        if col[3] != MAX_WEIGHT:
            print("↓", end="")
        else:
            print(" ", end="")
        print(" ", end="") # space for the right arrow
    print()
    

plt.imshow(img2show)
plt.show()
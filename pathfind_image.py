import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

CELLS_SIZE = 31 # 32 pixels

img = cv2.imread('./sample.png', cv2.IMREAD_COLOR) 

dim = img.shape
print(dim)

cells_height = math.floor(dim[0] / CELLS_SIZE)
cells_width  = math.floor(dim[1] / CELLS_SIZE)

print((cells_height, cells_width))

for y in range(0, dim[0], CELLS_SIZE):
    for x in range(0, dim[1], CELLS_SIZE):
        img2show = cv2.rectangle(img, (x+1,y+1), (x + CELLS_SIZE,y + CELLS_SIZE), (0,0,0), 1)

colors = []

for y in range(0, dim[0], CELLS_SIZE):
    for x in range(0, dim[1], CELLS_SIZE):
        cell_known = False
        for u in range(y, y + CELLS_SIZE, 1):
            if u >= dim[0]:
                break
            
            for v in range(x, x + CELLS_SIZE, 1):
                if v >= dim[1]:
                    break
                
                if tuple(img[u,v]) not in colors:
                    colors.append(tuple(img[u,v]))
                    
                if tuple(img[u,v]) == (128,0,255):
                    cell_known = True
                    img2show = cv2.putText(img2show, 'H',(x+3, y+CELLS_SIZE-3), 2, 1, (0,0,0),1)
                    break
                if tuple(img[u,v]) == (255, 128, 0):
                    cell_known = True
                    img2show = cv2.putText(img2show, 'F',(x+3, y+CELLS_SIZE-3), 2, 1, (0,0,0),1)
                if tuple(img[u,v]) == (0, 255, 0):
                    cell_known = True
                    img2show = cv2.putText(img2show, 'S',(x+3, y+CELLS_SIZE-3), 2, 1, (0,0,0),1)

            if cell_known:
                break
        if not cell_known:    
            img2show = cv2.putText(img2show, 'C',(x+3, y+CELLS_SIZE-3), 2, 1, (0,0,0),1)



print(colors)
plt.imshow(img2show)
plt.show()
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./sample.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

src = np.float32([[1025, 132], [855, 2702], [3337, 2722], [2974, 165]])
dest = np.float32([[0, 0], [0, 2000], [2000, 2000], [2000, 0]])

transmtx = cv2.getPerspectiveTransform(src, dest)
out = cv2.warpPerspective(img, transmtx, (2000, 2000))

plt.imshow(out)
plt.show()

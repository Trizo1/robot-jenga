import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("many_jenga.jpeg")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

_, binary = cv2.threshold(gray, 180, 225, cv2.THRESH_BINARY)
plt.imshow(binary, cmap="gray")
plt.show()

coutours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(image, coutours, -1, (0,255,0), 2)
plt.imshow(image)
plt.show()


import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)


while True:
    success, image = cap.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    cv2.imshow('rez', image)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
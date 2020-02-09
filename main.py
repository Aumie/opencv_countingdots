import cv2

img = cv2.imread('./img/dot.jpg')
dimensions = img.shape
h, w, _ = dimensions
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(imgGray, 240, 255, cv2.CHAIN_APPROX_NONE)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

count = 0
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
    if len(approx) > 10:
        count += 1

cv2.putText(img, "counted dot: {}".format(count), (0, h-25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

cv2.imshow("Counted_dot", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

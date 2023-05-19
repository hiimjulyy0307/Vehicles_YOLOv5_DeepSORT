import cv2
text1 = 'car'
text2 = 'motorcycle'
w1,h1 = cv2.getTextSize(text1,0,1,1)[1]
w2,h2 = cv2.getTextSize(text2,0,1,1)
print(w1,h1)
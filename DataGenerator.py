import cv2

cap = cv2.VideoCapture(0)
i = 0

while cap.isOpened():
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_name = "generated\\face1\\{0}.jpg".format(str(i))
    i += 1
    frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_AREA)
    # gray_mid = gray[:, gray.shape[1]//8:-gray.shape[1]//8]
    cv2.imwrite(img_name, frame)
    cv2.imshow('frame', frame)
    if not i % 128:
        print(i)
    if cv2.waitKey(1) & 0xFF == ord('q') or i == 1024:
        break

cap.release()
cv2.destroyAllWindows()

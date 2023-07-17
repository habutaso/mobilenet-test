import cv2

HAAR_FILE = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(HAAR_FILE)

cap = cv2.VideoCapture(1)
cap.set

while True:
    ret, frame = cap.read()

    face = cascade.detectMultiScale(frame)

    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

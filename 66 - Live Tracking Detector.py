import mediapipe as mp
import cv2
import time
from time import strftime

holisticModel = mp.solutions.holistic
modelParam = holisticModel.Holistic(
    static_image_mode = False,
    model_complexity = 1,
    smooth_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

modelDrawing = mp.solutions.drawing_utils

videoCapture = cv2.VideoCapture(0)

previousTime = 0
currentTime = 0

while videoCapture.isOpened():
    ret, frame = videoCapture.read()

    frame = cv2.resize(frame, (800, 600))

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = modelParam.process(image)
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    modelDrawing.draw_landmarks(image, results.face_landmarks, holisticModel.FACEMESH_CONTOURS,
                                modelDrawing.DrawingSpec(color = (255, 0, 0), thickness = 1, circle_radius = 1),
                                modelDrawing.DrawingSpec(color = (0, 255, ), thickness = 1, circle_radius = 1))

    modelDrawing.draw_landmarks(image, results.right_hand_landmarks, holisticModel.HAND_CONNECTIONS)
    modelDrawing.draw_landmarks(image, results.left_hand_landmarks, holisticModel.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(image, "Live Tracking Detector", (590, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(image, strftime("%x: %X"), (590, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(image, "Framerate: " + str(int(fps)) + " FPS", (590, 90), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)


    cv2.imshow("Live Tracking Detector", image)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

videoCapture.release()
cv2.destroyAllWindows()
                                
                                 
    

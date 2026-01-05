import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model
POSE_CONNECTIONS = [
    # FACE
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (3, 7),
    (6, 8),
    (9, 10),
    # TORSO
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    # LEFT ARM
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    # RIGHT ARM
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    # LEFT LEG
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    # RIGHT LEG
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
]

BBOX_FONT_SIZE = 1
BBOX_MARGIN_X = 10
BBOX_MARGIN_Y = 20
BBOX_THICKNESS = 1
BBOX_COLOR = (255, 0, 0)

LANDMARK_RADIUS = 5
LANDMARK_THICKNESS = 1
LANDMARK_COLOR = (255, 0, 0)


def create_object_detector():
    # https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/python#create_the_task
    base_options = python.BaseOptions(
        # DOWNLOAD MODELS: https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector#models
        model_asset_path="models/efficientdet_lite0.tflite"
    )
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.5,
        category_allowlist=["person"],
    )
    return vision.ObjectDetector.create_from_options(options)


def create_pose_landmarker():
    # https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#create_the_task
    base_options = python.BaseOptions(
        # DOWNLOAD MODELS: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
        model_asset_path="models/pose_landmarker_lite.task"
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main():
    object_detector = create_object_detector()
    pose_landmarker = create_pose_landmarker()

    # https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # DETECT OBJECT
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        object_detection_result = object_detector.detect(mp_image)

        for object_detection in object_detection_result.detections:
            # https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/python#handle_and_display_results
            bbox = object_detection.bounding_box
            x1 = bbox.origin_x
            y1 = bbox.origin_y
            x2 = bbox.origin_x + bbox.width
            y2 = bbox.origin_y + bbox.height

            category = object_detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + " (" + str(probability) + ")"
            text_location = (BBOX_MARGIN_X + x1, BBOX_MARGIN_Y + y1)

            # DRAW BBOX
            cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, 3)
            cv2.putText(
                frame,
                result_text,
                text_location,
                cv2.FONT_HERSHEY_PLAIN,
                BBOX_FONT_SIZE,
                BBOX_COLOR,
                BBOX_THICKNESS,
            )

            # DETECT POSE
            frame_copy = np.copy(frame)
            cropped_frame = frame_copy[y1:y2, x1:x2]
            cropped_frame = cv2.resize(cropped_frame, (0, 0), fx=1, fy=1)
            cropped_mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=cropped_frame
            )

            pose_detection_result = pose_landmarker.detect(cropped_mp_image)
            if len(pose_detection_result.pose_landmarks) <= 0:
                continue

            pose_landmarks = pose_detection_result.pose_landmarks[0]

            # DRAW POSE
            # https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#handle_and_display_results
            points = []
            for landmark in pose_landmarks:

                # OPTIONAL LANDMARK CONFIDENCE FILTER
                # if landmark.visibility < 0.5:
                #     continue

                x = int(landmark.x * bbox.width) + x1
                y = int(landmark.y * bbox.height) + y1
                point = (x, y)
                points.append(point)

                cv2.circle(
                    frame,
                    point,
                    LANDMARK_RADIUS,
                    LANDMARK_COLOR,
                    LANDMARK_THICKNESS,
                )

            for p1, p2 in POSE_CONNECTIONS:
                if (
                    p1 < len(points)
                    and p2 < len(points)
                    and points[p1] is not None
                    and points[p2] is not None
                ):
                    cv2.line(
                        frame,
                        points[p1],
                        points[p2],
                        LANDMARK_COLOR,
                        LANDMARK_THICKNESS,
                    )

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Mediapipe", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

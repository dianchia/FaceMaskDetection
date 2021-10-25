import logging
import time

import cv2
from rich.logging import RichHandler

from detector import (
    MaskDetector,
    FaceDetectorHaarcascade,
    MaskDetectorONNX,
    FaceDetectorMediapipe,
)
from reader import VideoReader


def main():
    LABELS = ["No Mask Detected", "Mask Detected", "Incorrect Mask Detected"]
    COLORS = [(0, 0, 255), (0, 255, 0), (0, 128, 255)]

    handler = RichHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Setting up camera...")
    cap = VideoReader("data/videos/test_video.avi")

    logger.info("Loading face detector and mask detector models")
    # face_detector = FaceDetectorMediapipe(h_margin=20, w_margin=10)
    face_detector = FaceDetectorHaarcascade()
    # Default to MobileNetV2. Change model_path to load another model.
    # mask_detector = MaskDetector(
    #     model_path="data/models/MobileNetV2_MaskDetectorFull.h5"
    # )
    mask_detector = MaskDetectorONNX(
        model_path="data/models/MobileNetV2_MaskDetectorFull.onnx"
    )

    last_check = time.perf_counter()
    last_fps = time.perf_counter()
    faces = None
    label = ""
    color = COLORS[0]
    fps = 0
    confidence = 0

    cap.start()
    logger.info("Starting detection...")

    while not cap.stopped:
        frame = cap.frame

        if frame is None:
            continue

        if time.perf_counter() - last_check > 0.05:
            faces = face_detector.detect_faces(frame)
            if faces is not None:
                for (x, y, x2, y2) in faces:
                    face_frame = frame[y:y2, x:x2]
                    prediction, confidence = mask_detector.detect(face_frame)
                    label = LABELS[prediction]
                    color = COLORS[prediction]
            last_check = time.perf_counter()

        if faces is not None:
            for x, y, x2, y2 in faces:
                cv2.rectangle(frame, (x, y), (x2, y2), (255, 127, 0), 2)
                cv2.putText(
                    frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
                cv2.putText(
                    frame,
                    f"{confidence:.2f}",
                    (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        fps = 1 / (time.perf_counter() - last_fps)
        fps = int(fps * 100) / 100
        last_fps = time.perf_counter()

        cv2.putText(
            frame, f"fps: {fps}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    logger.info("Stopping detection...")
    cap.stop()
    cv2.destroyAllWindows()
    logger.info("Quitting now! Bye!")


if __name__ == "__main__":
    main()

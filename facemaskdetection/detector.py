import logging
import os
import time
from typing import Union, List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime
import onnxruntime as rt
from rich.logging import RichHandler

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import mobilenet_v2, resnet50, inception_v3


handler = RichHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


"""
This module contains the different classes of detector, including two face detector and one face mask detector.
"""


class FaceDetectorHaarcascade:
    """
    Face Detector class using Haarcascade from OpenCV.
    """

    def __init__(self):
        self.cascade = cv2.CascadeClassifier()
        start = time.perf_counter()
        logger.info("Loading haarcascade classifier...")
        self.cascade.load("data/models/haarcascade_frontalface_default.xml")
        elapsed = time.perf_counter() - start
        logger.info(f"Finish loading haarcascde classifier in {elapsed: .2f} s")

    def detect_faces(self, img: np.ndarray) -> Optional[List[List[int]]]:
        """
        Detect faces from the given image.

        Parameters
        ----------
        img: :obj:`numpy` :obj: `ndarray`
            Input image to detect faces from.

        Returns
        -------
        :obj:`list` of :obj:`list` of :obj:`int`, optional
            A nested lists of points in the format of ``min_x, min_y, max_x, max_y``.
            If no faces are detected, return None.
        """
        output = []
        img.flags.writeable = False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray)
        for x, y, w, h in faces:
            output.append([x, y, x + w, y + h])
        return output


class FaceDetectorMediapipe:
    """
    Face Detector from `MediaPipe <https://google.github.io/mediapipe/>`_

    Attributes
    ----------
    h_margin: int
        Margin around the face at the vertical axis.
    w_margin: int
        Margin around the face at the horizontal axis.
    """

    def __init__(self, h_margin: int = 20, w_margin: int = 20):
        self.detector = mp.solutions.face_detection
        self.drawer = mp.solutions.drawing_utils
        self.h_margin = h_margin
        self.w_margin = w_margin

    def detect_faces(self, img: np.ndarray) -> Optional[List[List[int]]]:
        """
        Detect faces from the given image.

        Parameters
        ----------
        img: numpy.ndarray

        Returns
        -------
        :obj:`list` of :obj:`list` of :obj:`int`, optional
            A nested lists of points in the format of ``min_x, min_y, max_x, max_y``.
            If no faces are detected, return None.
        """
        img.flags.writeable = False
        faces = []

        with self.detector.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as face_detection:
            results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.detections:
                return None
            for detection in results.detections:
                boundingBox = detection.location_data.relative_bounding_box
                x = int(boundingBox.xmin * img.shape[1]) - self.w_margin
                y = int(boundingBox.ymin * img.shape[0]) - self.h_margin
                x2 = x + int(boundingBox.width * img.shape[1]) + self.w_margin
                y2 = y + int(boundingBox.height * img.shape[0]) + self.h_margin
                faces.append([x, y, x2, y2])

        return faces


class MaskDetector:
    """
    Mask Detector class.

    Attributes
    ----------
    threshold: float
        Threshold for the confidence level that a person is wearing the mask correctly.
        If confidence level is lower than threshold, it will be treated as wearing mask incorrectly.
    """

    def __init__(
        self,
        model_path: Union[
            str, bytes, os.PathLike
        ] = "data/models/MobileNetV2_MaskDetectorFull.h5",
        threshold: float = 0.8,
        model_type: str = "MobileNetV2",
    ):
        self.mask_detector = load_model(model_path)
        self.threshold = threshold
        if model_type.lower() == "mobilenet_v2":
            self.preprocessor = mobilenet_v2.preprocess_input
        elif model_type.lower() == "resnet50":
            self.preprocessor = resnet50.preprocess_input
        else:
            self.preprocessor = inception_v3.preprocess_input

    def detect(self, img: np.ndarray) -> Tuple[int, float]:
        """
        Detect the state of mask in the given image.

        Parameters
        ----------
        img: numpy ndarray
            Input image to detect mask from.

        Returns
        -------
        :obj:Tuples of :obj: int and :obj: float
            The prediction and confidence level.
            Prediction is 0, 1, or 2.
            0 for not wearing a mask, 1 for wearing a mask and 2 for wearing a mask but incorrectly.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = self.preprocessor(img)
        confidences = self.mask_detector.predict([img])[0]

        prediction = np.argmax(confidences)
        if prediction == 1 and confidences[1] < self.threshold:
            prediction = 2

        return prediction, confidences[prediction]


class MaskDetectorONNX:
    def __init__(self, model_path: Union[str, bytes, os.PathLike]):
        self.path = model_path
        self.session = self._init_session()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.preprocessor = mobilenet_v2.preprocess_input
        self.threshold = 0.8

    def _init_session(self) -> rt.InferenceSession:
        return rt.InferenceSession(self.path)

    def detect(self, img: np.ndarray) -> Tuple[int, float]:
        """
        Detect the state of mask in the given image.

        Parameters
        ----------
        img: numpy ndarray
            Input image to detect mask from.

        Returns
        -------
        :obj:Tuples of :obj: int and :obj: float
            The prediction and confidence level.
            Prediction is 0, 1, or 2.
            0 for not wearing a mask, 1 for wearing a mask and 2 for wearing a mask but incorrectly.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = self.preprocessor(img)
        confidences = self.session.run([self.output_name], {self.input_name: img})
        confidences = np.squeeze(confidences)

        prediction = np.argmax(confidences)
        if prediction == 1 and confidences[1] < self.threshold:
            prediction = 2
        return prediction, confidences[prediction]

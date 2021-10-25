import os
import time
from threading import Thread, Lock
from typing import Union

import cv2


"""
Video Reader Class
Threaded implementation of the OpenCV VideoCapture Class.
"""


class VideoReader:
    """
    Video Reader Class
    """

    def __init__(self, source: Union[str, int, os.PathLike], fps: int = 60):
        self.lock = Lock()
        self.cap = cv2.VideoCapture(source)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.fps = fps
        self.stopped = False
        self.frame = None

    def __del__(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

    def start(self):
        """
        Starts the capturing process.
        """
        self.stopped = False
        t = Thread(target=self.run)
        t.start()

    def stop(self):
        """
        Stops the capturing process.
        """
        self.stopped = True

    def release(self):
        self.cap.release()

    def run(self):
        while not self.stopped:
            try:
                ret, frame = self.cap.read()
                frame = cv2.flip(frame, 1)
                if not ret:
                    self.stop()
                    break

                with self.lock:
                    self.frame = frame

                time.sleep(1 / self.fps)
            except cv2.error:
                pass

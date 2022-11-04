import numpy as np
import cv2


class Dataset:
    def __init__(self, imageCount: int = 1000, loadOnInit: bool = False):
        self.ImageCount = imageCount
        self.TrainingSet = None

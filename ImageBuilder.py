import os.path
import random

import cv2
import numpy as np
from Network import NormalGeneratorNetwork
from Dataset import Dataset


class ImageBuilder:
    def __init__(self, workDir: str, network: NormalGeneratorNetwork, imageSize: int):
        self.Network = network
        self.ImageSize = imageSize
        self.RGBPath = "rgb"
        self.NormalPath = "normal"
        self.WorkDir = workDir
        self.InputImage = None
        self.ExpectedImage = None
        if self.WorkDir[0] == '/':
            self.WorkDir = self.WorkDir[1:]

    def GenerateImage(self):
        self.SelectRandomImage()
        parts = self.SplitImage()
        predicted = self.PredictParts(parts)

    def SelectRandomImage(self):
        rgb_files = [f for f in os.listdir(self.RGBPath) if os.path.isfile(os.path.join(self.RGBPath, f))]

        selectedImage = random.choice(rgb_files)
        self.InputImage = cv2.imread(os.path.join(self.RGBPath, selectedImage))
        self.ExpectedImage = cv2.imread(os.path.join(self.NormalPath, selectedImage))

        self.InputImage = cv2.cvtColor(self.InputImage, cv2.COLOR_BGR2RGB)

    def SplitImage(self):
        parts = []
        x = 0
        while x < self.InputImage.shape[0] - self.ImageSize:
            y = 0
            while y < self.InputImage.shape[1] - self.ImageSize:
                part = np.asarray(self.InputImage[x:x + self.ImageSize, y:y + self.ImageSize, :]).astype('float32') / 255
                parts.append(part)

                y += self.ImageSize
            x += self.ImageSize

        return np.asarray(parts)

    def RebuildImage(self, parts):
        pass

    def SaveImage(self, image: np.ndarray, name: str):
        cv2.imwrite(os.path.join(self.WorkDir, name), image)

    def PredictParts(self, parts: np.ndarray):
        return self.Network.Predict(parts)

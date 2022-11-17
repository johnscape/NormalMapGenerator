import logging
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
        logging.info("Generating full normal map!")
        self.SelectRandomImage()
        parts = self.SplitImage()
        predicted = self.PredictParts(parts)
        rebuilt = self.RebuildImage(predicted)
        self.SaveImage(rebuilt, "built_" + str(self.ImageSize) + ".png")
        logging.info("Normal map generation is finished!")

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

    def RebuildImage(self, parts: np.ndarray, windowed: bool = False):
        newImage = np.zeros((self.InputImage.shape[0], self.InputImage.shape[1], 3))
        shift = self.ImageSize
        if windowed:
            shift = 2
        steps = int(self.InputImage.shape[0] / shift)

        for x in range(steps):
            for y in range(steps):
                xStart = x * shift
                xEnd = (x + 1) * shift
                yStart = y * shift
                yEnd = (y + 1) * shift
                selectedPart = y * steps + x
                selectedPart = parts[selectedPart, :, :, :].reshape((self.ImageSize, self.ImageSize, 3))
                if (x == 0 and y == 0) or not windowed:
                    newImage[xStart:xEnd, yStart:yEnd, :] = selectedPart
                elif y == 0:
                    newImage[xEnd - shift: xEnd, yStart:yEnd, :] = selectedPart[selectedPart.shape[0] - shift:, :]
                else:
                    newImage[xEnd - shift: xEnd, yEnd - shift:yEnd, :] = selectedPart[
                                                                         selectedPart.shape[0] - shift:,
                                                                         selectedPart.shape[1] - shift:]

        newImage = np.round(newImage * 255, decimals=0).astype('uint8')
        newImage = cv2.cvtColor(newImage, cv2.COLOR_RGB2BGR)
        return newImage

    def SaveImage(self, image: np.ndarray, name: str):
        cv2.imwrite(os.path.join(self.WorkDir, name), image)
        cv2.imwrite(os.path.join(self.WorkDir, "expected_full_" + str(self.ImageSize) + ".png"), self.ExpectedImage)
        cv2.imwrite(os.path.join(self.WorkDir, "input_rgb_" + str(self.ImageSize) + ".png"), self.InputImage)

    def PredictParts(self, parts: np.ndarray):
        return self.Network.Predict(parts)

import logging
import os.path
import random

import cv2
import numpy as np
from Network import NormalGeneratorNetwork
from ImageProcessor import SplitImage
from tqdm import tqdm

def PartCount(imageSize: int, windowSize: int, shiftSize: int) -> int:
    count = 0
    xPos = 0
    while xPos + windowSize <= imageSize:
        yPos = 0
        while yPos + windowSize <= imageSize:
            count += 1
            yPos += shiftSize
        xPos += shiftSize

    return count

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
        self.WindowShiftSize = 4

    def GenerateImage(self):
        logging.info("Generating full normal map!")
        self.SelectRandomImage()
        # full tiling
        parts = SplitImage(self.InputImage, self.ImageSize, self.ImageSize).astype('float64')
        parts /= 255
        predicted = self.PredictParts(parts)
        rebuilt = self.BuildTiledImage(predicted)
        self.SaveImage(rebuilt, "built_tiled_" + str(self.ImageSize) + ".png")
        # windowed tiling
        shifts = [8, 16, 32, 64]
        for s in shifts:
            if s >= self.ImageSize:
                continue
            logging.debug("Building shifted image with shift {0}".format(s))
            shifted = self.BuildShiftedImage(s)
            self.SaveImage(shifted, "built_shifted_" + str(self.ImageSize) + "_" + str(s) + ".png")

        logging.info("Normal map generation is finished!")

    def SelectRandomImage(self):
        rgb_files = [f for f in os.listdir(self.RGBPath) if os.path.isfile(os.path.join(self.RGBPath, f))]

        selectedImage = random.choice(rgb_files)
        self.InputImage = cv2.imread(os.path.join(self.RGBPath, selectedImage))
        self.ExpectedImage = cv2.imread(os.path.join(self.NormalPath, selectedImage))

        while self.InputImage.shape[0] > 1024:
            dim = (int(self.InputImage.shape[0] / 2), int(self.InputImage.shape[1] / 2))
            self.InputImage = cv2.resize(self.InputImage, dim, interpolation=cv2.INTER_AREA)
            self.ExpectedImage = cv2.resize(self.ExpectedImage, dim, interpolation=cv2.INTER_AREA)

    def BuildTiledImage(self, parts: np.ndarray):
        newImage = np.zeros((self.InputImage.shape[0], self.InputImage.shape[1], 3))
        shift = self.ImageSize
        steps = int(((self.InputImage.shape[0] - self.ImageSize) / shift) + 1)
        count = 0

        for x in range(steps):
            for y in range(steps):
                xStart = x * shift
                xEnd = (x + 1) * shift
                yStart = y * shift
                yEnd = (y + 1) * shift
                selectedPart = y * steps + x
                selectedPart = parts[count, :, :, :].reshape((self.ImageSize, self.ImageSize, 3))
                count += 1
                newImage[xStart:xEnd, yStart:yEnd, :] = selectedPart

        newImage = np.round(newImage * 255, decimals=0).astype('uint8')
        return newImage

    def BuildShiftedImage(self, shift: int) -> np.ndarray:
        newImage = np.zeros((self.InputImage.shape[0], self.InputImage.shape[1], 3))
        step_count = int((self.InputImage.shape[0] - self.ImageSize) / shift) + 1
        for cols in tqdm(range(step_count)):
            parts = []
            for rows in range(step_count):
                part = self.InputImage[
                       cols * shift: cols * shift + self.ImageSize,
                       rows * shift : rows * shift + self.ImageSize,
                       :]
                parts.append(part)
            parts = np.asarray(parts).astype('float64')
            parts /= 255
            predicted = self.PredictParts(parts)
            parts = 0
            for rows in range(step_count):
                current_part = predicted[parts, :, :, :].reshape((self.ImageSize, self.ImageSize, 3))
                newImage[
                    cols * shift : cols * shift + self.ImageSize,
                    rows * shift : rows * shift + self.ImageSize,
                    :
                ] = current_part
                parts += 1
        newImage = np.round(newImage * 255, decimals=0).astype('uint8')
        return newImage

    def SaveImage(self, image: np.ndarray, name: str):
        path = os.path.join(self.WorkDir, "result_" + str(self.ImageSize))
        cv2.imwrite(os.path.join(path, name), image)
        cv2.imwrite(os.path.join(path, "expected_full_" + str(self.ImageSize) + ".png"), self.ExpectedImage)
        cv2.imwrite(os.path.join(path, "input_rgb_" + str(self.ImageSize) + ".png"), self.InputImage)

    def BuildSinglePicture(self, filePath: str):
        self.InputImage = cv2.imread(filePath)
        while self.InputImage.shape[0] > 1024:
            dim = (int(self.InputImage.shape[0] / 2), int(self.InputImage.shape[1] / 2))
            self.InputImage = cv2.resize(self.InputImage, dim, interpolation=cv2.INTER_AREA)

        parts = SplitImage(self.InputImage, self.ImageSize, self.ImageSize).astype('float64')
        parts /= 255

        predicted = self.PredictParts(parts)
        rebuilt = self.BuildTiledImage(predicted)

        cv2.imwrite("generated_normal.png", rebuilt)

    def PredictParts(self, parts: np.ndarray):
        return self.Network.Predict(parts, False)

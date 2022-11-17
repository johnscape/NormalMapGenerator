import random

import numpy as np
import cv2
from ImageProcessor import ImageProcessor
from Dataset import Dataset
from Network import NormalGeneratorNetwork
from ImageBuilder import ImageBuilder
import logging

logging.basicConfig(filename='log.txt', level=logging.DEBUG)
imageSizes = [16, 32, 64, 128, 256, 512]


for imageSize in imageSizes:
    logging.info("Starting new round with image size " + str(imageSize))
    processor = ImageProcessor()
    processor.ProcessImages(imageSize, imageSize)
    processor.CreateTestingSet()

    trainingDataset = Dataset("/work", 1000, True, True)
    testingDataset = Dataset("/work", 100, True, False)

    network = NormalGeneratorNetwork("work", imageSize, trainingDataset, testingDataset)
    network.CreateModel()
    network.Train()

    imageNumber = random.randint(0, len(testingDataset.Dataset[0]))
    inputImage = np.asarray(testingDataset.Dataset[0][imageNumber])
    expectedImage = np.asarray(testingDataset.Dataset[1][imageNumber])
    generatedImage = network.Predict(inputImage.reshape((1, imageSize, imageSize, 3))).reshape(
        (imageSize, imageSize, 3))

    inputImage = np.round(inputImage * 255, decimals=0).astype('uint8')
    expectedImage = np.round(expectedImage * 255, decimals=0).astype('uint8')
    generatedImage = np.round(generatedImage * 255, decimals=0).astype('uint8')

    cv2.imwrite("work/rgb_" + str(imageSize) + ".png", cv2.cvtColor(inputImage, cv2.COLOR_RGB2BGR))
    cv2.imwrite("work/normal_" + str(imageSize) + ".png", cv2.cvtColor(expectedImage, cv2.COLOR_RGB2BGR))
    cv2.imwrite("work/generated_" + str(imageSize) + ".png", cv2.cvtColor(generatedImage, cv2.COLOR_RGB2BGR))

    logging.info("Test images saved!")

    builder = ImageBuilder("work", network, imageSize)
    builder.GenerateImage()
    logging.info("Round finished!")

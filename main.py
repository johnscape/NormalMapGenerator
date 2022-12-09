import os.path
import random

import numpy as np
import cv2
from ImageProcessor import ImageProcessor
from Dataset import Dataset
from Network import NormalGeneratorNetwork
from ImageBuilder import ImageBuilder
import logging

logging.basicConfig(
    filename='log.txt',
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)

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
    #network.LoadModel()


    processor = None
    trainingDataset = None
    testingDataset = None

    builder = ImageBuilder("work", network, imageSize)
    builder.GenerateImage()
    builder = None
    logging.info("Round finished!")

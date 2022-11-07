import os.path
import random
import cv2
import numpy as np


class Dataset:
    def __init__(self, datasetDirectory: str, imageCount: int = 1000, loadOnInit: bool = False, training: bool = True):
        self.ImageCount = imageCount
        self.Dataset = None
        self.DatasetDir = datasetDirectory
        self.IsLoaded = False

        if self.DatasetDir[0] == '/':
            self.DatasetDir = self.DatasetDir[1:]

        self.RGBPath = None
        self.NormalPath = None

        if training:
            self.RGBPath = os.path.join(self.DatasetDir, "training/rgb")
            self.NormalPath = os.path.join(self.DatasetDir, "training/normal")
        else:
            self.RGBPath = os.path.join(self.DatasetDir, "testing/rgb")
            self.NormalPath = os.path.join(self.DatasetDir, "testing/normal")

        if loadOnInit:
            self.Load()

    def Load(self):
        files = [f for f in os.listdir(self.RGBPath) if os.path.isfile(os.path.join(self.RGBPath, f))]
        selectedFiles = []

        while len(selectedFiles) < self.ImageCount:
            selection = random.choice(files)
            if selection not in selectedFiles:
                selectedFiles.append(selection)

        self.Dataset = [[], []]
        for f in selectedFiles:
            # load rgb first
            path = os.path.join(self.RGBPath, f)
            rgbImg = cv2.imread(path)
            rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB)
            # load normal next
            path = os.path.join(self.NormalPath, f)
            normalImg = cv2.imread(path)
            normalImg = cv2.cvtColor(normalImg, cv2.COLOR_BGR2RGB)
            # normalise
            rgbImg = np.asarray(rgbImg, dtype=np.float32)
            normalImg = np.asarray(normalImg, dtype=np.float32)

            rgbImg /= 255
            normalImg /= 255
            # add images to array
            self.Dataset[0].append(rgbImg)
            self.Dataset[1].append(normalImg)

        self.IsLoaded = True

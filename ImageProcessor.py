import math
import os.path
import cv2


class ImageProcessor:
    def __init__(self, rgbBath: str = "/rgb", normalPath: str = "/normal", datasetDir: str = "/work", verbose: int = 0):
        self.RGBPath = rgbBath
        self.NormalPath = normalPath
        self.DatasetDirectory = datasetDir
        self.Verbose = verbose  # 0 - none, 1 - low, 2 - high

        if self.RGBPath[0] == '/':
            self.RGBPath = self.RGBPath[1:]
        if self.NormalPath[0] == '/':
            self.NormalPath = self.NormalPath[1:]
        if self.DatasetDirectory[0] == '/':
            self.DatasetDirectory = self.DatasetDirectory[1:]

        self.TrainingPath = os.path.join(self.DatasetDirectory, "training")
        self.TestingPath = os.path.join(self.DatasetDirectory, "testing")

        if not self.CreateMissingFolder(self.RGBPath):
            return
        if not self.CreateMissingFolder(self.NormalPath):
            return
        if not self.CreateMissingFolder(self.DatasetDirectory):
            return
        if not self.CreateMissingFolder(self.TrainingPath):
            return
        if not self.CreateMissingFolder(self.TestingPath):
            return

    @staticmethod
    def IsDirectoryValid(path: str) -> bool:
        return os.path.exists(os.path.join(os.getcwd(), path))

    def CreateMissingFolder(self, path: str) -> bool:
        if not self.IsDirectoryValid(path):
            try:
                if self.Verbose == 2:
                    print("Creating folder at " + path)
                realPath = os.path.join(os.getcwd(), path)
                os.mkdir(realPath)
                return True
            except OSError as e:
                print(e)
                return False
        return True

    def ProcessImages(self, windowSize: int = 32, windowStep: int = 2):
        if not self.CreateMissingFolder(os.path.join(self.TrainingPath, "rgb")): return
        if not self.CreateMissingFolder(os.path.join(self.TrainingPath, "normal")): return

        if not self.CreateMissingFolder(os.path.join(self.TestingPath, "rgb")): return
        if not self.CreateMissingFolder(os.path.join(self.TestingPath, "normal")): return

        rgb_files = [f for f in os.listdir(self.RGBPath) if os.path.isfile(os.path.join(self.RGBPath, f))]
        normal_files = [f for f in os.listdir(self.NormalPath) if os.path.isfile(os.path.join(self.NormalPath, f))]

        if self.Verbose == 2:
            print("Tiling RGB images...")
        count = 1
        for f in rgb_files:
            self.CreateTiledImage(windowSize, windowStep, os.path.join(self.RGBPath, f), False, count)
            count += 1
        if self.Verbose == 2:
            print("Tiling normal images...")
        count = 1
        for f in normal_files:
            self.CreateTiledImage(windowSize, windowStep, os.path.join(self.NormalPath, f), True, count)
            count += 1
        if self.Verbose >= 1:
            print("Image processing done!")

    def CreateTiledImage(self, windowSize: int, windowStep: int, fileName: str, isNormal: bool, imgNumber: int):
        img = cv2.imread(fileName)

        if isNormal:
            savePath = os.path.join(self.TrainingPath, "normal")
        else:
            savePath = os.path.join(self.TrainingPath, "rgb")

        if img is None:
            raise ValueError(fileName + " cannot be opened!")
        if img.shape[0] % windowSize != 0 or img.shape[1] % windowSize != 0:
            raise ValueError(fileName + " cannot be tiled with " + str(windowSize) + "!")

        count = 0
        x = 0
        while x < img.shape[0] - windowSize:
            y = 0
            while y < img.shape[1] - windowSize:
                part = img[x:x + windowSize, y:y + windowSize, :]

                path = str(imgNumber).zfill(2) + "_" + str(count).zfill(4) + ".png"
                path = os.path.join(savePath, path)

                cv2.imwrite(path, part)

                count += 1
                y += windowStep
            x += windowStep

    def CreateTestingSet(self, percent: float = 0.2):
        rgbTrainingPath = os.path.join(self.TrainingPath, "rgb")
        rgbTestingPath = os.path.join(self.TestingPath, "rgb")

        normalTrainingPath = os.path.join(self.TrainingPath, "normal")
        normalTestingPath = os.path.join(self.TestingPath, "normal")

        rgbTiles = [f for f in os.listdir(rgbTrainingPath) if os.path.isfile(os.path.join(rgbTrainingPath, f))]
        normalTiles = [f for f in os.listdir(normalTrainingPath) if os.path.isfile(os.path.join(normalTrainingPath, f))]

        if len(rgbTiles) != len(normalTiles):
            raise ValueError("RGB and normal tile numbers do not match! " +
                             str(len(rgbTiles)) + " vs " + str(len(normalTiles)))
        numberOfCopies = len(rgbTiles) * percent
        numberOfCopies = math.floor(numberOfCopies)

        if numberOfCopies == 0:
            return

        count = 0
        for f in rgbTiles:
            os.replace(os.path.join(rgbTrainingPath, f), os.path.join(rgbTestingPath, f))
            count += 1
            if count >= numberOfCopies:
                break

        count = 0
        for f in normalTiles:
            os.replace(os.path.join(normalTrainingPath, f), os.path.join(normalTestingPath, f))
            count += 1
            if count >= numberOfCopies:
                break

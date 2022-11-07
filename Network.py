import os.path

import keras.models
from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam

from Dataset import Dataset


class NormalGeneratorNetwork:

    def __init__(self, datasetDirectory: str, trainingSet: Dataset = None, testingSet: Dataset = None):
        self.WorkingDirectory = datasetDirectory
        self.ModelPath = os.path.join(datasetDirectory, "model")
        self.ImageSize = 32
        self.TrainingDataset = trainingSet
        self.TestingDataset = testingSet
        self.Model = None # type: keras.Model

        if self.TrainingDataset is None:
            self.TrainingDataset = Dataset(self.WorkingDirectory, 1000, True, True)
        if self.TestingDataset is None:
            self.TestingDataset = Dataset(self.WorkingDirectory, 100, True, False)

    def CreateModel(self):
        inputImg = Input(shape=(self.ImageSize, self.ImageSize, 3))

        conv = Conv2D(15, (3, 3), activation='relu', padding='same', use_bias=False)(inputImg)
        conv = BatchNormalization()(conv)
        conv = AveragePooling2D((4, 4), padding='same')(conv)
        conv = Conv2D(30, (3, 3), activation='relu', padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = UpSampling2D((4, 4))(conv)
        conv = Conv2D(15, (3, 3), activation='relu', padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = Conv2D(3, (3, 3), activation='relu', padding='same')(conv)
        generator = Model(inputImg, conv)
        generator.compile(Adam(amsgrad=True), loss='mse')
        self.Model = generator

    def IsModelExists(self) -> bool:
        return os.path.isfile(self.ModelPath)

    def Train(self):
        if not self.TrainingDataset.IsLoaded:
            self.TrainingDataset.Load()
        if not self.TestingDataset.IsLoaded:
            self.TestingDataset.Load()

        # Training cycle
        self.Model.fit(self.TrainingDataset.Dataset[0], self.TrainingDataset.Dataset[1],
                       validation_data=(self.TestingDataset.Dataset[0], self.TestingDataset.Dataset[1]),
                       batch_size=32,
                       epochs=100)

    def SaveModel(self, overwrite: bool = False):
        if self.IsModelExists():
            if not overwrite:
                return
            else:
                os.remove(self.ModelPath)
        self.Model.save(self.ModelPath)

    def LoadModel(self):
        if self.IsModelExists():
            self.Model = keras.models.load_model(self.ModelPath)
        # TODO: Error when does not exists

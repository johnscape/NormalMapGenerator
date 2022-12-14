import os.path
import matplotlib.pyplot as plt
import keras.models
from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, BatchNormalization, Layer, InputSpec
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
from Dataset import Dataset

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'REFLECT')


class NormalGeneratorNetwork:

    def __init__(self, datasetDirectory: str, imageSize: int, trainingSet: Dataset = None, testingSet: Dataset = None):
        self.WorkingDirectory = datasetDirectory
        self.ModelPath = os.path.join(datasetDirectory, "model_" + str(imageSize))
        self.ImageSize = imageSize
        self.TrainingDataset = trainingSet
        self.TestingDataset = testingSet
        self.Model = None  # type: keras.Model
        self.TrainingEpochs = 100

    def CreateModel(self):
        inputImg = Input(shape=(self.ImageSize, self.ImageSize, 3))

        conv = ReflectionPadding2D()(inputImg)
        conv = Conv2D(15, (3, 3), activation='relu', padding='valid', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = ReflectionPadding2D()(conv)
        conv = AveragePooling2D((4, 4), padding='valid')(conv)
        conv = ReflectionPadding2D()(conv)
        conv = Conv2D(30, (3, 3), activation='relu', padding='valid', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = UpSampling2D((4, 4))(conv)
        conv = ReflectionPadding2D()(conv)
        conv = Conv2D(15, (3, 3), activation='relu', padding='valid', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = ReflectionPadding2D()(conv)
        conv = Conv2D(3, (3, 3), activation='relu', padding='valid')(conv)
        generator = Model(inputImg, conv)
        generator.compile(Adam(amsgrad=True), loss='mse')

        self.Model = generator

    def IsModelExists(self) -> bool:
        return os.path.isdir(self.ModelPath)

    def Train(self):
        if self.TrainingDataset is None:
            self.TrainingDataset = Dataset(self.WorkingDirectory, 1000, True, True)
        if self.TestingDataset is None:
            self.TestingDataset = Dataset(self.WorkingDirectory, 100, True, False)

        if not self.TrainingDataset.IsLoaded:
            self.TrainingDataset.Load()
        if not self.TestingDataset.IsLoaded:
            self.TestingDataset.Load()

        # load model if not exists
        self.PrepareModel()
        # Training cycle
        trainingRGB = np.asarray(self.TrainingDataset.Dataset[0])
        trainingNormal = np.asarray(self.TrainingDataset.Dataset[1])

        testingRGB = np.asarray(self.TestingDataset.Dataset[0])
        testingNormal = np.asarray(self.TestingDataset.Dataset[1])

        checkpoint = ModelCheckpoint(
            filepath=self.ModelPath,
            save_weights_only=False,
            mode='max',
            save_best_only=True
        )

        history = self.Model.fit(trainingRGB, trainingNormal,
                                 validation_data=(testingRGB, testingNormal),
                                 batch_size=32,
                                 epochs=self.TrainingEpochs,
                                 callbacks=[checkpoint])

        # plotting the loss
        losses = history.history["loss"]
        valLosses = history.history["val_loss"]
        epochs = np.arange(0, self.TrainingEpochs)
        fig = plt.figure()
        plt.ion()
        plt.plot(epochs, losses, '-b', label="Epoch loss")
        plt.plot(epochs, valLosses, '-g', label="Epoch evaluation loss")
        plt.title("Loss with image size of " + str(self.ImageSize))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper left")
        plt.ioff()
        path = os.path.join(self.WorkingDirectory, "result_" + str(self.ImageSize))
        path = os.path.join(path, "loss_" + str(self.ImageSize) + ".png")
        plt.savefig(path)

    def Predict(self, image: np.ndarray, verbose: bool = True):
        self.PrepareModel()
        v = 0 if verbose is False else 1
        result = self.Model.predict(image, verbose=v)
        return result

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

    def DeleteModel(self):
        os.unlink(self.ModelPath)

    def PrepareModel(self):
        if self.Model is None:
            if self.IsModelExists():
                self.LoadModel()
            else:
                self.CreateModel()

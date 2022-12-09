from ImageProcessor import ImageProcessor
from Dataset import Dataset
from Network import NormalGeneratorNetwork
from ImageBuilder import ImageBuilder
import logging
import argparse
import os.path

def file_path(string):
    if os.path.isfile(string) or string is None:
        return string
    else:
        raise NotADirectoryError(string)

def dir_path(string):
    if os.path.isdir(string) or string is None:
        return string
    else:
        raise NotADirectoryError(string)

def DemoCycle(workdir: str, rgbDir: str, normalDir: str):
    imageSizes = [16, 32, 64, 128, 256]

    for imageSize in imageSizes:
        if imageSize < 256:
            continue
        logging.info("Starting new round with image size " + str(imageSize))
        processor = ImageProcessor(rgbDir, normalDir, workdir)
        processor.ProcessImages(imageSize, imageSize)
        processor.CreateTestingSet()

        trainingDataset = Dataset(workdir, 1000, True, True)
        testingDataset = Dataset(workdir, 100, True, False)

        network = NormalGeneratorNetwork(workdir, imageSize, trainingDataset, testingDataset)
        network.CreateModel()
        network.Train()
        #network.LoadModel()


        processor = None
        trainingDataset = None
        testingDataset = None

        builder = ImageBuilder(workdir, network, imageSize)
        builder.GenerateImage()
        builder = None
        logging.info("Round finished!")



parser = argparse.ArgumentParser(description="A neural network for normal map generation")
parser.add_argument("-s", "--size", type=int, default=64, help="The window size to use on the network")
parser.add_argument("-f", "--file", type=file_path, default=None, help="The path of the diffuse map to generate normal map from")
parser.add_argument("-d", "--demo", action="store_true", help="Set this to run the demo cycle")
parser.add_argument("-w", "--workdir", type=dir_path, default="work", help="Set the working directory for the project")
parser.add_argument("-r", "--rgb", type=dir_path, default="rgb", help="Set the path to the RGB files")
parser.add_argument("-n", "--normal", type=dir_path, default="normal", help="Set the path to the normal maps")
parser.add_argument("-l", "--log", type=file_path, default="log.txt", help="The path to the logfile")

if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG)

    if args.demo:
        DemoCycle(args.workdir, args.rgb, args.normal)
    elif args.file is not None:
        generatorNetwork = NormalGeneratorNetwork(args.workdir, args.size)
        if not generatorNetwork.IsModelExists():
            logging.error("No model of {0} size exists!".format(args.size))
            exit(1)
        generatorNetwork.LoadModel()
        imgBuilder = ImageBuilder(args.workdir, generatorNetwork, args.size)
        imgBuilder.BuildSinglePicture(args.file)


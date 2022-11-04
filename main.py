from ImageProcessor import ImageProcessor
from Dataset import Dataset

processor = ImageProcessor(verbose=2)
processor.ProcessImages(32, 32)
processor.CreateTestingSet()

dataset = Dataset("/work", 100, False, True)
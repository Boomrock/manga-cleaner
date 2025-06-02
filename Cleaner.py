import PIL
import PIL.Image
from simple_lama_inpainting import SimpleLama
from Cleaner.Preprocessor import Preprocessor
from Cleaner.Postprocessor import Postprocessor
from Cleaner.ModelLoader import ModelLoader
from Cleaner.DetectorType import DetectorType
import torch
import torch.nn as nn
from PIL import Image
from typing import  Tuple, Union
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

class Cleaner:
    _detector: nn.Module
    _cleaner: SimpleLama
    _device: torch.device
    _preprocessor: Preprocessor
    _postprocessor: Postprocessor

    def __init__(self, detector_type: DetectorType = DetectorType.UNet):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._detector, self._preprocessor, self._postprocessor = self.__initDetector(detector_type)
        self._detector = self._detector.to(self._device)

        self._cleaner = SimpleLama(self._device) # type: ignore

    @staticmethod 
    def _get_model(type: DetectorType) -> nn.Module | None:
        if(type == DetectorType.UNet):
            model = smp.Unet(
                encoder_name="vgg13",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation="sigmoid"
            )
        elif(type == DetectorType.UNetPlusPlus):
            model = smp.UnetPlusPlus(
                encoder_name="vgg11",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation="sigmoid")
        return model
    @staticmethod 
    def _get_processor(type: DetectorType) -> Tuple[Preprocessor, Postprocessor]:
        if(type == DetectorType.UNet):
            preprocessor = Preprocessor()
            postprocessor = Postprocessor()
        elif(type == DetectorType.UNetPlusPlus):
            preprocessor = Preprocessor()
            postprocessor = Postprocessor()

        return preprocessor, postprocessor
    
    def __initDetector(self, type: DetectorType) ->  Tuple[nn.Module, Preprocessor, Postprocessor]:
        model = self._get_model(type)
        preprocessor, postprocessor = self._get_processor(type)
        if(model == None or preprocessor == None):
            raise TypeError("Model not emplemented")
        loader = ModelLoader()
        loader.load(model, type, self._device.type)
        return model, preprocessor, postprocessor

    def clean(self, image: Union[str, Image.Image]) -> Image.Image:
        # Если передан путь — открыть изображение
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # 2. Обработка через segment_model (на полном изображении)
        mask = self.detect(image)# [H, W]
        result = self._cleaner(image, mask)

        return result
    
    def detect(self, image: PIL.Image.Image) -> PIL.Image.Image:
        tensor = self._preprocessor(image).to(self._device)
        predicted_mask = self._detector(tensor)
        preprocessed_image = self._postprocessor(predicted_mask, self._device)
        return preprocessed_image

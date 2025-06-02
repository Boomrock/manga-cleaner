import PIL
import PIL.Image
import numpy
import torch
from DetectorType import DetectorType
from torchvision import transforms


class Preprocessor:
    def __call__(self, image: PIL.Image.Image) -> torch.Tensor:
        return self.process(image)
    
    def process(self, image: PIL.Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform(image).unsqueeze(0) # type: ignore [1,3,h,w]
        return image # type: ignore
    


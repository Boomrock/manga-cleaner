import PIL
import PIL.Image
import numpy
import torch
from Cleaner.DetectorType import DetectorType
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
    
class UnetPlusPlusPreprocessor(Preprocessor):
    def process(self, image: PIL.Image.Image) -> torch.Tensor:
        original_size = image.size  # (width, height)

        # Вычисление необходимого паддинга до кратности 32
        def to_divisible(x, divisor=32):
            return ((x + divisor) // divisor) * divisor

        new_width = to_divisible(original_size[0])
        new_height = to_divisible(original_size[1])

        # Применение Resize и других трансформаций
        resized_image = image.resize((new_width, new_height))
        
        return super().process(resized_image)
    

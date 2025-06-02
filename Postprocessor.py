import PIL
import PIL.Image
from PIL import Image
import torch

import torch
import torch.nn.functional as F

from PIL import Image

class Postprocessor:
    def __call__(self, tensor: torch.Tensor, device: torch.device, kernel_size: int = 15) -> PIL.Image.Image:
        return self.process(tensor, device, kernel_size)
    
    def process(self, tensor: torch.Tensor, device: torch.device,  kernel_size: int = 15) -> PIL.Image.Image:
        image = morphological_merge_single_mask(tensor, device, kernel_size).cpu().numpy()
        image = Image.fromarray(image).convert("L")
        return image
    

def morphological_merge_single_mask(mask: torch.Tensor, device: torch.device, kernel_size:int=15):
    """
    Объединяет близкие фрагменты в одной маске с помощью морфологических операций.

    Args:
        mask (Tensor): Бинарная маска формы [H, W]
        kernel_size (int): Размер структурирующего элемента (ядра)

    Returns:
        Tensor: Объединенная маска формы [H, W]
    """
    # 1. Преобразуем маску в бинарный формат (если ещё не бинарная)
    binary_mask = (mask > 0.5).float()

    # 2. Создаем структурирующий элемент (ядро)
    kernel = torch.ones(1, 1, kernel_size, kernel_size).to(device)

    # 3. Дилатация — увеличивает объекты, соединяя близкие фрагменты
    dilated = F.conv2d(binary_mask, kernel, padding=kernel_size // 2)
    dilated_mask = (dilated > 0).float()

    # 4. Эрозия — уменьшает объекты, но сохраняет соединенные области
    eroded = F.conv2d(dilated_mask, kernel, padding=kernel_size // 2)
    merged_mask = (eroded > 0).float()

    # 5. Удаляем лишние измерения
    return merged_mask.squeeze(0).squeeze(0)

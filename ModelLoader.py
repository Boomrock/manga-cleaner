import torch
from torch import nn

from DetectorType import DetectorType


_github = "https://github.com/Boomrock/Cleaner/"

class ModelLoader: 
    repo: str = _github
    default_path: str = "./models"


    def load(self, model: nn.Module, model_type: DetectorType, map_location='cpu'):
        model_name = self._get_path(model_type)


        path = self.repo + model_name
        
        
        self._load_from_github(model, path, map_location)
        

    def _load_from_github(self, model: nn.Module, github_url: str = _github, map_location='cpu'):
        """
        Скачивает веса с GitHub и загружает их в модель.
        
        :param model: Объект модели PyTorch (уже созданной)
        :param github_url: Прямая ссылка на .pth / .pt файл в GitHub
        :param map_location: Устройство для загрузки ('cpu' или 'cuda')
        :return: Обновлённая модель
        """
        state_dict = torch.hub.load_state_dict_from_url(github_url, map_location)
        model.load_state_dict(state_dict)
        
        return model
    

    @staticmethod
    def _get_path(model_path: DetectorType) -> str:
        if DetectorType.UNet:
            return "Unet.pth"
        elif DetectorType.UNetPlusPlus:
            return "UnetPlusPlus.pth"
        raise ValueError("Неизвестная модель")
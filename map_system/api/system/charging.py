import os
import torch

from charging_system.pythons.api.models.prediction_state import Prediction_State_Module
from charging_system.pythons.api.models.prediction_temperature import Prediction_Temperature_Module


class Charging_System:
    def __init__(self, path_ckpt):
        self.model_state, self.model_temperature = self.load_model(path_ckpt)


    def load_model(self, path_ckpt):
        path_model_state = path_ckpt + 'state/version_0/checkpoints/'
        path_model_temperature = path_ckpt + 'temperature/version_0/checkpoints/'
        ckpt_model_state = path_model_state + os.listdir(path_model_state)[0]
        ckpt_model_temperature = path_model_temperature + os.listdir(path_model_temperature)[0]

        # 加载模型
        ckpt = torch.load(ckpt_model_state)
        model_state = Prediction_State_Module(ckpt['hyper_parameters']['paras'])
        for key, value in ckpt['state_dict'].copy().items():
            key_new = key[(len(key.split('.')[0]) + 1):]
            ckpt['state_dict'].update({key_new: ckpt['state_dict'].pop(key)})
        model_state.load_state_dict(state_dict=ckpt['state_dict'])

        ckpt = torch.load(ckpt_model_temperature)
        model_temperature = Prediction_Temperature_Module(ckpt['hyper_parameters']['paras'])
        for key, value in ckpt['state_dict'].copy().items():
            key_new = key[(len(key.split('.')[0]) + 1):]
            ckpt['state_dict'].update({key_new: ckpt['state_dict'].pop(key)})
        model_temperature.load_state_dict(state_dict=ckpt['state_dict'])
        return model_state, model_temperature



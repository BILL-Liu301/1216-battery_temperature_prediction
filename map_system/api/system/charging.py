import os
import torch

from charging_system.pythons.api.models.prediction_state import Prediction_State_Module
from charging_system.pythons.api.models.prediction_temperature import Prediction_Temperature_Module


class Charging_System:
    def __init__(self, path_ckpt):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        return model_state.to(self.device).eval(), model_temperature.to(self.device).eval()

    def charging(self, charging_data):
        # charging_data: [25, charging_stamp, [stamp, location, current, soc, condition_temperature, voltage, ntc_max, ntc_min, temperature_max]]
        # 将数据移至device
        charging_data = torch.from_numpy(charging_data).to(torch.float32).to(self.device)

        with torch.no_grad():
            # 使用模型预测state
            oup_m_state, oup_var_state, _ = self.model_state(charging_data[:, 0:1, 2:5], charging_data[:, 0:1, 5:8], charging_data[:, 1:, 2:5])
            charging_data[:, 1:, 5:8] = oup_m_state

            # 使用模型预测temperature
            oup_m_temperature, oup_var_temperature, _ = self.model_temperature(torch.cat([charging_data[:, 0:1, 1:4], charging_data[:, 0:1, 5:8]], dim=2),
                                                                               charging_data[:, 0:1, -1:],
                                                                               torch.cat([charging_data[:, 1:, 1:4], charging_data[:, 1:, 5:8]], dim=2))
            charging_data[:, 1:, -1:] = oup_m_temperature

        return charging_data.cpu().numpy()


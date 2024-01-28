import torch
import torch.nn as nn
import pytorch_lightning as pl


class Prediction_All_Module(nn.Module):
    def __init__(self, model_state, model_temperature):
        super(Prediction_All_Module, self).__init__()
        self.model_state = model_state
        self.model_temperature = model_temperature

    def forward(self, inp_his, inp_info):
        # inp_his: [B, 1, [location, Current, SOC, Voltage, NTC_max, NTC_min, Temperature_max]]
        # inp_info: [B, seq_predict, [location, Current, SOC]]
        oup_m_state, oup_var_state, _ = self.model_state(inp_his[:, :, 1:3], inp_his[:, :, 3:6], inp_info[:, :, 1:], h_his=None, c_his=None)
        inp_info = torch.cat([inp_info, oup_m_state], dim=2)
        oup_m_temperature, oup_var_temperature, _ = self.model_temperature(inp_his[:, :, 0:6], inp_his[:, :, 6:7], inp_info, h_his=None, c_his=None)
        return torch.cat([oup_m_state, oup_m_temperature], dim=2), torch.cat([oup_var_state, oup_var_temperature], dim=2)

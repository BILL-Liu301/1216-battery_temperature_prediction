import torch
import torch.nn as nn
import pytorch_lightning as pl


class Prediction_All_Module(nn.Module):
    def __init__(self, model_state, model_temperature):
        super(Prediction_All_Module, self).__init__()
        self.model_state = model_state
        self.model_temperature = model_temperature

    def forward(self, inp_his, inp_info):
        # inp_his: [25, 1, [location, current, soc, condition, voltage, ntc_max, ntc_min, temperature_max]]
        # inp_info: [25, seq_predict, [location, current, soc, condition]]

        oup_m_state, oup_var_state, _ = self.model_state(inp_his[:, :, 1:4], inp_his[:, :, 4:7], inp_info[:, :, 1:4], h_his=None, c_his=None)
        inp_info = torch.cat([inp_info[:, :, :-1], oup_m_state], dim=2)  # [25, seq_predict, [location, current, soc, voltage, ntc_max, ntc_min]]
        oup_m_temperature, oup_var_temperature, _ = self.model_temperature(inp_info[:, 0:1], inp_his[:, :, -1:], inp_info, h_his=None, c_his=None)
        return torch.cat([oup_m_state, oup_m_temperature], dim=2), torch.cat([oup_var_state, oup_var_temperature], dim=2)

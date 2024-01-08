import numpy as np


def integration_predicted_temperature(temperature_prediction, temperature_origin, mode):
    # mode：
    #   0：直接拼接
    #   1：逐渐求平均
    temperature_prediction_integration = dict()
    for cell_name, temperature_groups in temperature_prediction.items():
        temperature_reference = temperature_origin[cell_name]
        temperature_prediction_integration_np = np.zeros(temperature_reference.shape)
        temperature_prediction_integration_np[0:3] = temperature_reference[0:3]

        seq_predict = temperature_groups[0].cpu().numpy().shape[1]
        temperature_prediction_integration_np[:, 0:seq_predict] = temperature_groups[0].cpu().numpy()
        for group_id, group in enumerate(temperature_groups):
            group_np = group.cpu().numpy()
            seq_predict = group_np.shape[1]
            if mode == 0:
                pass
            elif mode == 1:
                temperature_prediction_integration_np[3:, group_id:(group_id + seq_predict-1)] = (
                        (temperature_prediction_integration_np[3:, group_id:(group_id + seq_predict-1)] + group_np[3:, 0:seq_predict-1]) / 2)
                temperature_prediction_integration_np[3:, group_id + seq_predict - 1] = group_np[3:, seq_predict-1]

        temperature_prediction_integration.update({cell_name: temperature_prediction_integration_np})
    return temperature_prediction_integration

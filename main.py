import torch
from fastapi import FastAPI
from models.lstm import LSTM
from models.models import Data

model_config = {
    "num_classes": 5,
    "input_size": 5,
    "hidden_size": 64,
    "num_layers": 2,
    "bi": False
}
data_max = torch.tensor([258.41, 259.9, 252.55, 258.44, 11909662.889], dtype=torch.float32)
data_min = torch.tensor([1.1987, 1.2891, 1.0301, 1.198, 1393.58], dtype=torch.float32)
model = LSTM(**model_config)
checkpoint = torch.load("best_3.pt", map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
app = FastAPI()


@app.post("/api/v1/predict")
async def get_predict(data: Data):
    with torch.no_grad():
        input_data = data.data
        num_predict = data.num_predict
        input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        window_size = input_data.shape[1]
        # print(window_size)

        input_data = (input_data - data_min) / (data_max - data_min)
        result = []
        count = 0
        file = open("input.txt", 'w')
        file_out = open("output.txt", 'w')
        while count < num_predict:
            file.write(str(input_data[:, -window_size:, :]))
            file.write("\n")
            predict = model(input_data[:, -window_size:, :])

            next = predict.clone().unsqueeze(0)
            next[:, :, 0] = input_data[:, -1:, :][:, :, 3].clone()
            count += 1
            # print(next.shape)
            input_data = torch.cat([input_data.clone(), next], dim=1)
            # print(input_data.shape)
            # file.write(str(input_data))
            # file.write("\n")
            predict[:, 0:1] = next[:, :, 0]
            predict = predict * (data_max - data_min) + data_min
            file_out.write(str(next))
            file_out.write("\n")
            result.append(predict.squeeze(0).numpy().tolist())
        return {"prediction": result}

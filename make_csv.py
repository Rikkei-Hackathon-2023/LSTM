import json
import csv
import os

from tqdm import tqdm

field_name = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
root_folder = "/home/hadoop/Downloads/hackathon/drive-download-20230728T082733Z-001"
for js in os.listdir(root_folder):
    json_file = os.path.join(root_folder, js)
    name = js.split("_")[0]

    with open(json_file, 'r') as f:
        data = json.load(f)

    save_csv = []
    for d in tqdm(data, total=len(data)):
        if len(d) == 0:
            continue
        try:
            record = d[0]
            save_csv.append({
                "Timestamp": record[0],
                "Open": record[1],
                "High": record[2],
                "Low": record[3],
                "Close": record[4],
                "Volume": record[5]
            })
        except Exception:
            print(d)
            exit()

    save_path = f"data_{name}.csv"
    with open(save_path, 'w') as sf:
        writer = csv.DictWriter(sf, fieldnames=field_name)
        writer.writeheader()
        writer.writerows(save_csv)

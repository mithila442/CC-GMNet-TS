# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from sklearn.preprocessing import StandardScaler

# class SmartFallDataset(Dataset):
#     def __init__(self, data_directory, sequence_length=128):
#         self.data_directory = data_directory
#         self.sequence_length = sequence_length
#         self.sequences = []
#         self.labels = []

#         step_size = sequence_length  # no overlap

#         all_data = []

#         for filename in os.listdir(data_directory):
#             if filename.endswith(".csv"):
#                 path = os.path.join(data_directory, filename)

#                 try:
#                     # Read only the 2nd, 3rd, and 4th columns (x, y, z)
#                     acc_data = np.loadtxt(path, delimiter=',', usecols=[1, 2, 3])
#                 except Exception as e:
#                     print(f"[Warning] Skipping file {filename} due to read error: {e}")
#                     continue

#                 try:
#                     activity_code = int(filename.split('A')[1].split('T')[0])
#                 except Exception as e:
#                     print(f"[Warning] Skipping file {filename} due to filename parsing error: {e}")
#                     continue

#                 all_data.append((acc_data, activity_code))

#         if not all_data:
#             raise ValueError("No valid CSV files found in the directory.")

#         # Normalize globally
#         all_features = np.vstack([d for d, _ in all_data])
#         scaler = StandardScaler().fit(all_features)

#         for acc_data, activity_code in all_data:
#             acc_data = scaler.transform(acc_data)
#             binary_label = 1 if activity_code >= 10 else 0  # fall vs. non-fall

#             for start in range(0, len(acc_data) - sequence_length + 1, step_size):
#                 end = start + sequence_length
#                 segment = acc_data[start:end]
#                 self.sequences.append(segment)
#                 self.labels.append(binary_label)

#         self.labels = np.array(self.labels)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return (
#             torch.tensor(self.sequences[idx], dtype=torch.float32),
#             torch.tensor(self.labels[idx], dtype=torch.long)
#         )

import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class SmartFallDataset(Dataset):
    def __init__(self, data_directory, sequence_length=128):
        self.data_directory = data_directory
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []

        step_size = sequence_length
        all_data = []

        for filename in os.listdir(data_directory):
            if not filename.endswith(".csv"):
                continue

            path = os.path.join(data_directory, filename)

            try:
                df = pd.read_csv(path, sep=None, engine="python", header=None)
                if df.shape[1] < 4:
                    print(f"[Warning] Skipping file {filename}: fewer than 4 columns.")
                    continue

                acc_data = df.iloc[:, [1, 2, 3]].apply(pd.to_numeric, errors="coerce").dropna().values

                if len(acc_data) < sequence_length:
                    print(f"[Warning] Skipping file {filename}: too short after cleaning.")
                    continue

            except Exception as e:
                print(f"[Warning] Skipping file {filename} due to read error: {e}")
                continue

            try:
                activity_code = int(filename.split('A')[1].split('T')[0])
            except Exception as e:
                print(f"[Warning] Skipping file {filename} due to filename parsing error: {e}")
                continue

            all_data.append((acc_data, activity_code))

        if not all_data:
            raise ValueError("No valid CSV files found in the directory.")

        all_features = np.vstack([d for d, _ in all_data])
        scaler = StandardScaler().fit(all_features)

        for acc_data, activity_code in all_data:
            acc_data = scaler.transform(acc_data)
            binary_label = 1 if activity_code >= 10 else 0

            for start in range(0, len(acc_data) - sequence_length + 1, step_size):
                end = start + sequence_length
                self.sequences.append(acc_data[start:end])
                self.labels.append(binary_label)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

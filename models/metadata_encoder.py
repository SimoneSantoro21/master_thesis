import torch
import torch.nn as nn
import pandas as pd
import ast
import numpy as np

class metadata_encoder(nn.Module):
    def __init__(self, csv_path, output_shape=(1, 128, 128)):
        super().__init__()
        self.output_shape = output_shape
        self.df = pd.read_csv(csv_path)

        input_dim = 13  # 3 (center) + 9 (3 neighbors) + 1 (bval)
        output_size = output_shape[0] * output_shape[1] * output_shape[2]

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_size),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def extract_input_vector(self, center_index: int) -> torch.Tensor:
        row = self.df[self.df["Center Index"] == center_index]
        if row.empty:
            raise ValueError(f"Center Index {center_index} not found.")
        row = row.iloc[0]

        center_dir = ast.literal_eval(row["Center Direction"])
        neighbors_dirs = ast.literal_eval(row["Neighbors directions"])
        first_three = neighbors_dirs[:3]
        while len(first_three) < 3:
            first_three.append((0.0, 0.0, 0.0))  # pad

        flat_neighbors = [x for vec in first_three for x in vec]
        bval = row["Multi_Bval"]

        vector = torch.tensor(list(center_dir) + flat_neighbors + [bval], dtype=torch.float32)
        return vector

    def forward(self, center_index: int) -> torch.Tensor:
        z = self.extract_input_vector(center_index)
        x = self.decoder(z)
        return x.view(*self.output_shape)
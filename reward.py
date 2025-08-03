import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset
import ast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime
from torchmetrics import Accuracy

df = pd.read_csv("reward_data.csv")

df["state"] = df["state"].apply(ast.literal_eval)
df["action"] = df["action"].apply(ast.literal_eval)
df["move"] = df.apply(lambda row: row["state"] + row["action"], axis=1)

class RewardDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx] - 1, dtype=torch.long)
    
data = RewardDataset(
    X=df["move"].tolist(),
    y=df["reward"].tolist()
)

dataloader = DataLoader(
    dataset=data,
    batch_size=8,
    shuffle=True
)

class RewardModel(nn.Module):
    def __init__(self, in_features, hidden_units, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=out_features)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
if __name__ == "__main__":

    model = RewardModel(
    in_features=44,
    hidden_units=32,
    out_features=10
    )

    loss_fn = nn.CrossEntropyLoss()
    acc_fn = Accuracy(task="multiclass", num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 10000
    for epoch in tqdm(range(epochs)):
        for X, y in dataloader:

            y_logits = model(X)
            y_pred_probs = torch.softmax(y_logits, dim=1)
            y_pred = torch.argmax(y_logits, dim=1)

            loss = loss_fn(y_logits, y)
            acc = acc_fn(y_pred, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch: {epoch + 1} | Loss: {loss:.5f} | Acc: {acc:.3f}") 

    torch.save(model.state_dict(), f"reward_models/model-{datetime.now().timestamp()}")
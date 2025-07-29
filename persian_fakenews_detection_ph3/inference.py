import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


random_seed = 101
set_seed(random_seed)
graph_features = ['S2', 'S3', 'S4', 'S6', 'S7', 'S11', 'S12', 'S13', 'S14', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7',
                  'T8', 'T9', 'T10', 'T11', 'T12']  # remove S1, S5, S10


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(len(graph_features), 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(32, 32)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(32, 16)
        self.dropout5 = nn.Dropout(0.5)
        self.fc6 = nn.Linear(16, 16)
        self.dropout6 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(16, 8)
        self.dropout7 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        x = torch.relu(self.fc5(x))
        x = self.dropout5(x)
        x = torch.relu(self.fc6(x))
        x = self.dropout6(x)
        x = torch.relu(self.fc7(x))
        x = self.dropout7(x)
        x = torch.softmax(self.fc8(x), dim=1)
        return x


def predict_samples(model, samples, is_likelihood=True):
    temp_tensor = samples
    if type(samples) == np.ndarray:
        temp_tensor = torch.tensor(samples, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(temp_tensor)
    if is_likelihood:
        return outputs
    else:
        return 1 - torch.max(outputs, 1)[-1]


graph_model = NeuralNetwork()
graph_model.load_state_dict(torch.load("../codes/extra/graph2025_5_12.pth", weights_only=True, map_location=torch.device('cuda')))
scaler = joblib.load('../codes/extra/graph_scaler(1).pkl')
df = pd.read_csv('../data/features.csv')
# data = pd.read_csv('extra/sample_graph_features.csv')
data = df[graph_features]
labels = df['is_real'].tolist()
# labels = [1 - l for l in labels]
train, test = train_test_split(data, test_size=0.2, random_state=42)
# scaler = StandardScaler().fit(data)
test = data
# test = scaler.transform(data)

cr = classification_report(
    # test['is_real'].values,
    labels,
    predict_samples(graph_model, scaler.transform(test[graph_features]), is_likelihood=False),
    target_names=["fake", "real"]
)
b = predict_samples(graph_model, scaler.transform(test[graph_features]), is_likelihood=False)
print(cr)

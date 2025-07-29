import ast

import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import RandomOverSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def set_random_seed(random_seed: int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(random_seed)


random_seed = 101
set_random_seed(random_seed)


def remove_nodes_with_type(G, remove_type):
    nodes_to_remove = [node for node, attr in G.nodes(data=True) if 'type' in attr and attr['type'] == remove_type]
    new_graph = G.copy()
    for node in nodes_to_remove:
        parent = list(new_graph.predecessors(node))[0]
        children = list(new_graph.successors(node))
        for child in children:
            new_graph.add_edge(parent, child)

        new_graph.remove_node(node)
    return new_graph


def oversample_is_real(df, x):
    X = df.drop(columns=['is_real'])
    y = df['is_real']
    ros = RandomOverSampler(sampling_strategy={0: len(y[y == 1]), 1: len(y[y == 1]) * x}, random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    df_oversampled = pd.concat([X_res, y_res], axis=1)
    return df_oversampled


def oversample_is_fake(df, x):
    X = df.drop(columns=['is_real'])
    y = df['is_real']

    count_0 = len(y[y == 0])  # number of fake samples
    target_0 = int(count_0 * x)
    count_1 = len(y[y == 1])  # number of real samples

    ros = RandomOverSampler(sampling_strategy={0: target_0, 1: count_1}, random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    df_oversampled = pd.concat([X_res, y_res.rename('is_real')], axis=1)
    return df_oversampled


def print_metrics(accuracy, precision, recall, f1_score_val):
    print("Accuracy : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("F1 : {}".format(f1_score_val))


def get_metrics(target, logits, one_hot_rep=True):
    """
    Two numpy one hot arrays
    :param target:
    :param logits:
    :return:
    """

    if one_hot_rep:
        label = np.argmax(target, axis=1)
        predict = np.argmax(logits, axis=1)
    else:
        label = target
        predict = logits

    accuracy = accuracy_score(label, predict)
    precision = precision_score(label, predict)
    recall = recall_score(label, predict)
    f1_score_val = f1_score(label, predict)

    return accuracy, precision, recall, f1_score_val


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(len(features), 64)
        self.dropout1 = nn.Dropout(0.01)
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.01)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.01)
        self.fc4 = nn.Linear(32, 32)
        self.dropout4 = nn.Dropout(0.01)
        self.fc5 = nn.Linear(32, 16)
        self.dropout5 = nn.Dropout(0.01)
        self.fc6 = nn.Linear(16, 16)
        self.dropout6 = nn.Dropout(0.01)
        self.fc7 = nn.Linear(16, 8)
        self.dropout7 = nn.Dropout(0.01)
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
        # x = torch.softmax(self.fc8(x), dim=1)
        return x


def calculate_accuracy(labels, outputs):
    temp_labels = labels
    if type(labels) == pd.Series:
        temp_labels = torch.tensor(labels.to_numpy(), dtype=torch.long)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == temp_labels).sum().item()
    total = temp_labels.size(0)
    accuracy = correct / total
    return accuracy


def predict_samples(model, samples, is_likelihood=True):
    temp_tensor = samples
    if type(samples) == np.ndarray:
        temp_tensor = torch.tensor(samples, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(temp_tensor)
    if is_likelihood:
        return torch.max(outputs, 1)[-1], outputs
    else:
        return torch.max(outputs, 1)[-1]


def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience=10, num_epochs=10):
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accuracy = calculate_accuracy(labels, outputs)
            running_accuracy += accuracy

        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                accuracy = calculate_accuracy(labels, outputs)
                val_accuracy += accuracy

        running_loss /= len(train_loader)
        running_accuracy /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}, Accuracy: {running_accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
            print(f'Early stopping after {epoch + 1} epochs.')
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}, Accuracy: {running_accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}, Accuracy: {running_accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
            break

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}, Accuracy: {running_accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

        print("****" * 50)


df = pd.read_csv('../data/features.csv')
y = df['is_real'].tolist()
df['retem'] = df['retem'].apply(ast.literal_eval)
df['repem'] = df['repem'].apply(ast.literal_eval)
retem_df = pd.DataFrame(df['retem'].tolist(), index=df.index)
repem_df = pd.DataFrame(df['repem'].tolist(), index=df.index)
retem_df.columns = [f'retem_{i}' for i in retem_df.columns]
repem_df.columns = [f'repem_{i}' for i in repem_df.columns]
df = df.drop(columns=['retem', 'repem', 'is_real'])
df = pd.concat([df, retem_df, repem_df], axis=1)
df['is_real'] = y
fs1 = list(df.columns[:25])
fs2 = list(df.columns[25:49])
fs3 = list(df.columns[49:-1])
features = fs1

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df['is_real'], test_size=0.15, random_state=random_seed, stratify=df['is_real']
)

X_train_df = pd.DataFrame(X_train, columns=features)
y_train_df = pd.Series(y_train, name='is_real')
df_train = pd.concat([X_train_df, y_train_df], axis=1)

df_train_oversampled = oversample_is_fake(df_train, 3)
X_train = df_train_oversampled[features]
y_train = df_train_oversampled['is_real']

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scaler = StandardScaler().fit(X_train)
joblib.dump(scaler, 'graph_scaler.pkl')

X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train, test_size=0.015, random_state=random_seed,
                                                        stratify=y_train)

X_train_tensor = torch.tensor(X_train_nn, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_nn.to_numpy(), dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience=7, num_epochs=100)
predicts = predict_samples(model, X_test, is_likelihood=False)
metrics = get_metrics(y_test, predicts, one_hot_rep=False)
print_metrics(metrics[0], metrics[1], metrics[2], metrics[3])
print(classification_report(y_test, predicts, target_names=['fake', 'real']))
# draw_confusion_matrix(y_test, predicts, "feed forward nn")

torch.save(model.state_dict(), "graph.pth")

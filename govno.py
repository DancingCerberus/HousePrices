import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn import preprocessing
import numpy as np


def data_preparation(train_data_path: str, test_data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_train_data_df = pd.read_csv(train_data_path, index_col=0)
    train_SellPrice = raw_train_data_df.pop("SalePrice")
    numeric_features = raw_train_data_df.select_dtypes(include="number").columns.tolist()
    train_data_df = pd.get_dummies(raw_train_data_df, dtype="float32")
    train_data_df["SalePrice"] = train_SellPrice.apply(np.log)

    raw_test_data_df = pd.read_csv(test_data_path, index_col=0)
    test_data_df = pd.get_dummies(raw_test_data_df, dtype="float32")
    test_data_df = test_data_df.reindex(columns=train_data_df.columns, fill_value=0)

    standard_scaler = preprocessing.StandardScaler()
    train_data_df[numeric_features] = standard_scaler.fit_transform(train_data_df[numeric_features])
    test_data_df[numeric_features] = standard_scaler.transform(test_data_df[numeric_features])

    # Проверка данных на наличие NaN и бесконечных значений
    assert not train_data_df.isnull().values.any(), "Train data contains NaN values"
    assert not test_data_df.isnull().values.any(), "Test data contains NaN values"
    assert np.isfinite(train_data_df.values).all(), "Train data contains infinite values"
    assert np.isfinite(test_data_df.values).all(), "Test data contains infinite values"

    return train_data_df, test_data_df


class HouseDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        prices = df.pop("SalePrice").values
        features = df.values

        self.features = torch.tensor(features, dtype=torch.float32)
        self.prices = torch.tensor(prices, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.prices[idx]


# Пути к данным
train_path = "./data/train.csv"
test_path = "./data/test.csv"

# Подготовка данных
full_train_dataset_df, test_dataset_df = data_preparation(train_path, test_path)
full_train_dataset = HouseDataset(full_train_dataset_df)

# Разделение данных на обучающую и валидационную выборки
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [int(0.95 * len(full_train_dataset)),
                                                                                len(full_train_dataset) - int(
                                                                                    0.95 * len(full_train_dataset))])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# Подготовка тестового набора данных
test_dataset = HouseDataset(test_dataset_df)
test_dataloader = DataLoader(test_dataset, batch_size=16)

print(len(full_train_dataset))
print(len(train_dataset))
print(len(val_dataset))
a, b = full_train_dataset[0]
print(a)
print(b)


class HouseNetwork(nn.Module):
    def __init__(self):
        super(HouseNetwork, self).__init__()
        self.fc1 = nn.Linear(287, 100)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 100)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.activation(self.fc1(x)))
        x = self.dropout2(self.activation(self.fc2(x)))
        x = self.fc3(x)
        return x


# Инициализация модели, функции потерь и оптимизатора
model = HouseNetwork()
loss_fun = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Увеличьте learning rate

# Параметры обучения
epochs = 5

# Цикл обучения
for epoch in range(epochs):
    epoch_loss = 0.0
    for house, price in train_dataloader:
        optimizer.zero_grad()
        prediction = model(house)
        prediction = prediction.squeeze()
        loss = loss_fun(prediction, price)
        epoch_loss += loss.item()
        loss.backward()

        # Проверка градиентов
        for param in model.parameters():
            if torch.isnan(param.grad).sum() > 0:
                print("NaN detected in gradients")
                break

        optimizer.step()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
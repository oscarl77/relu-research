import os.path

import joblib
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_openml

def get_dataloaders():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def load_titanic_dataset(sample_idx=None):
    X, y = fetch_openml('titanic', version=1, as_frame=True, return_X_y=True)
    y = y.astype(int)
    continuous_features = ['pclass', 'age', 'sibsp', 'parch', 'fare']
    categorical_features = ['sex', 'embarked']

    continuous_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('cts', continuous_transformer, continuous_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    processor_filename = "../preprocessor/titanic_preprocessor.joblib"
    if not os.path.exists(processor_filename):
        joblib.dump(preprocessor, processor_filename)

    if sample_idx is not None:
        return X_test.iloc[[sample_idx]], X_test_processed[sample_idx], y_test.values[sample_idx]

    return X_train_processed, X_test_processed, y_train, y_test

def convert_dataset_for_pytorch(X_train, X_test, y_train, y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
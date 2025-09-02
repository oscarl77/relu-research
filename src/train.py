import joblib
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.model import TitanicClassifierNN
from src.util.preprocess import load_titanic_dataset, convert_dataset_for_pytorch


def train_models():
    torch.manual_seed(42)
    X_train, X_test, y_train, y_test = load_titanic_dataset()

    _train_neural_network(X_train, X_test, y_train, y_test)
    _train_logistic_regression(X_train, y_train)
    _train_decision_tree(X_train, y_train)


def _train_neural_network(X_train, X_test, y_train, y_test):
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = convert_dataset_for_pytorch(X_train, X_test, y_train, y_test)

    input_features = X_train_tensor.shape[1]
    nn_model = TitanicClassifierNN(input_features)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

    num_epochs = 150

    for epoch in range(num_epochs):
        nn_model.train()
        optimizer.zero_grad()
        y_pred_logits = nn_model(X_train_tensor)
        loss = loss_fn(y_pred_logits, y_train_tensor)
        loss.backward()
        optimizer.step()

    torch.save(nn_model.state_dict(), '../models/nn_model.pth')
    print("Neural network trained and saved to: models/nn_model.pth")

def _train_logistic_regression(X_train, y_train):
    log_reg = LogisticRegression(random_state=42, C=0.5)
    log_reg.fit(X_train, y_train)
    joblib.dump(log_reg, '../models/logreg_model.joblib')
    print("Logistic regression trained and saved to: models/logreg_model.joblib")

def _train_decision_tree(X_train, y_train):
    decision_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    decision_tree.fit(X_train, y_train)
    joblib.dump(decision_tree, '../models/decision_tree.joblib')
    print("Decision tree trained and saved to: models/decision_tree.joblib")

if __name__ == '__main__':
    train_models()
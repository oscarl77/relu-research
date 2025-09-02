import joblib
import torch
from sklearn.metrics import classification_report

from src.model import TitanicClassifierNN
from src.util.preprocess import load_titanic_dataset, convert_dataset_for_pytorch
from src.util.visualisation import plot_confusion_matrices


def evaluate_models():
    X_train, X_test, y_train, y_test = load_titanic_dataset()
    target_names = ['Died (0)', 'Survived (1)']
    nn_preds = _evaluate_nn(X_train, X_test, y_train, y_test, target_names)
    lr_preds = _evaluate_logistic_regression(X_test, y_test, target_names)
    dt_preds = _evaluate_decision_tree(X_test, y_test, target_names)
    plot_confusion_matrices(target_names, nn_preds, lr_preds, dt_preds, y_test)

def _evaluate_nn(X_train, X_test, y_train, y_test, target_names):
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = convert_dataset_for_pytorch(X_train, X_test, y_train, y_test)

    input_features = X_test_tensor.shape[1]
    nn_model = TitanicClassifierNN(input_features)
    nn_model.load_state_dict(torch.load('../models/nn_model.pth'))
    nn_model.eval()

    with torch.no_grad():
        test_logits = nn_model(X_test_tensor)
        test_probabilities = torch.sigmoid(test_logits)
        predictions_tensor = (test_probabilities > 0.5).int()
        predictions = predictions_tensor.numpy().flatten()

    print('--- Neural Network ---')
    print(classification_report(y_test_tensor.numpy(), predictions, target_names=target_names))
    return predictions

def _evaluate_logistic_regression(X_test, y_test, target_names):
    log_reg_model = joblib.load('../models/logreg_model.joblib')
    predictions = log_reg_model.predict(X_test)
    print('--- Logistic Regression ---')
    print(classification_report(y_test, predictions, target_names=target_names))
    return predictions

def _evaluate_decision_tree(X_test, y_test, target_names):
    decision_tree_model = joblib.load('../models/decision_tree.joblib')
    predictions = decision_tree_model.predict(X_test)
    print('--- Decision Tree ---')
    print(classification_report(y_test, predictions, target_names=target_names))
    return predictions

if __name__ == '__main__':
    evaluate_models()
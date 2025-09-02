import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.tree import export_text

from src.model import TitanicClassifierNN
from src.util.preprocess import load_titanic_dataset
from src.util.util_methods import create_diagonal_matrix, compute_parameter_equivalents, create_feature_names_list
from src.util.visualisation import plot_local_explanation, plot_global_explanation, plot_passenger_data, \
    plot_explanation_as_waterfall, combine_comparison_plots


def compare_local_explanation_to_shap(sample_idx):
    torch.manual_seed(42)
    np.random.seed(42)

    features_to_drop = ['name', 'cabin', 'boat', 'ticket', 'body', 'home.dest']
    feature_names = create_feature_names_list('../preprocessor/titanic_preprocessor.joblib')
    cleaned_feature_names = [name.replace('cts__', '').replace('cat__', '') for name in feature_names]

    x_original, x_sample, y_sample = load_titanic_dataset(sample_idx)
    x_original.drop(features_to_drop, axis=1, inplace=True)

    #plot_passenger_data(x_original, sample_idx)

    W_eq, b_eq = _get_nn_feature_importance(x_sample, feature_names, shap=True)

    X_train, X_test, y_train, y_test = load_titanic_dataset(sample_idx=None)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    explainer = create_shap_explainer()

    shap_values = explainer.shap_values(x_test_tensor).squeeze()
    base_value = explainer.expected_value

    active_features_mask = x_test_tensor[sample_idx] != 0
    active_shap_values = shap_values[sample_idx][active_features_mask]

    active_features_mask_np = active_features_mask.numpy()
    cleaned_feature_names_np = np.array(cleaned_feature_names)
    active_feature_names = list(cleaned_feature_names_np[active_features_mask_np])

    explanation = shap.Explanation(
        values=active_shap_values,
        base_values=base_value,
        data=None,
        feature_names=active_feature_names,
    )
    #shap_explanation_2d = explanation[:, :, 0]
    #shap.plots.beeswarm(shap_explanation_2d, show=False)
    plt.figure()
    plot_explanation_as_waterfall(x_sample, W_eq, b_eq.detach().numpy(), active_feature_names)
    plt.title(f"Local Linear Explanation for Passenger {sample_idx}")
    plt.savefig(f"../figures/shap/local_explanations/local_plot_{sample_idx}.png", bbox_inches='tight', transparent=True, dpi=300)
    plt.close()

    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"SHAP Explanation for Passenger {sample_idx}")
    plt.savefig(f"../figures/shap/shap_explanations/shap_plot_{sample_idx}.png", bbox_inches='tight', transparent=True, dpi=300)
    plt.close()

    combine_comparison_plots(sample_idx)

def compare_predictions(sample_idx):
    torch.manual_seed(42)
    np.random.seed(42)

    features_to_drop = ['name', 'cabin', 'boat', 'ticket', 'body', 'home.dest']

    x_original, x_sample, y_sample = load_titanic_dataset(sample_idx)
    x_original.drop(features_to_drop, axis=1, inplace=True)
    true_label = "Survived" if y_sample == 1 else "Died"
    feature_names = create_feature_names_list('../preprocessor/titanic_preprocessor.joblib')

    local_importance_list, nn_predicted_label = _get_nn_feature_importance(x_sample, feature_names)
    global_importance_list, logreg_predicted_label = _get_log_reg_feature_importance(x_sample, feature_names)

    plot_passenger_data(x_original, sample_idx)
    plot_local_explanation(local_importance_list, sample_idx, nn_predicted_label, true_label)
    plot_global_explanation(global_importance_list, local_importance_list, sample_idx, logreg_predicted_label, true_label)

def create_shap_explainer():
    X_train_processed, X_test_processed, y_train, y_test = load_titanic_dataset(sample_idx=None)
    X_background = torch.tensor(X_test_processed, dtype=torch.float32)
    input_features = X_background.shape[1]

    nn_model = TitanicClassifierNN(input_features)
    nn_model.load_state_dict(torch.load('../models/nn_model.pth'))

    explainer = shap.DeepExplainer(nn_model, X_background)
    return explainer

def _get_nn_feature_importance(x_sample, feature_names=None, shap=None):
    input_tensor = torch.tensor(x_sample, dtype=torch.float32).reshape(1, -1)
    input_features = input_tensor.shape[1]
    nn_model = TitanicClassifierNN(input_features)
    nn_model.load_state_dict(torch.load('../models/nn_model.pth'))
    nn_model.eval()

    with torch.no_grad():
        prediction_probability = torch.sigmoid(nn_model(input_tensor)).item()
        predicted_label = "Survived" if prediction_probability > 0.5 else "Died"

    W_eq, b_eq = _extract_local_linear_function_from_nn(nn_model, input_tensor)
    if shap:
        return W_eq, b_eq

    feature_importance_list = sorted(zip(feature_names, W_eq.flatten().tolist()), key=lambda x: abs(x[1]), reverse=True)
    all_contributions = dict(feature_importance_list)
    # Create a dictionary of the passenger's actual feature values
    passenger_values = dict(zip(feature_names, x_sample))

    numerical_contributions = {}
    categorical_contributions = {}

    for feature, weight in all_contributions.items():
        # Check if the feature is one-hot encoded (e.g., contains '_')
        if 'cat' in feature:
            # If it's categorical, only keep it if it's active (value is 1)
            if passenger_values.get(feature, 0) == 1:
                categorical_contributions[feature] = weight
        else:
            # If it's numerical, always keep it
            numerical_contributions[feature] = weight

    final_contributions = {**numerical_contributions, **categorical_contributions}
    processed_importance_list = sorted(final_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    return processed_importance_list, predicted_label

def _get_log_reg_feature_importance(x_sample, feature_names):
    log_reg_model = joblib.load('../models/logreg_model.joblib')

    prediction_probability = log_reg_model.predict_proba(x_sample.reshape(1, -1))[0, 1]
    predicted_label = "Survived" if prediction_probability > 0.5 else "Died"

    coefficients = log_reg_model.coef_[0]
    feature_importance = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
    return feature_importance, predicted_label

def _get_decision_path_rules(x_sample, y_sample, feature_names):
    true_label = "Survived" if y_sample == 1 else "Died"
    decision_tree_model = joblib.load('../models/decision_tree.joblib')

    prediction = decision_tree_model.predict(x_sample.reshape(1, -1))[0]
    if prediction == 1:
        prediction_label = "Survived"
    else:
        prediction_label = "Died"
    decision_path_rules = export_text(decision_tree_model, feature_names=list(feature_names))

    print("\n--- Decision Tree Analysis (Rule-Based) ---")
    print(f"True: {true_label} | Prediction: {prediction_label}")
    print("Decision tree rules:")
    print(decision_path_rules)

def _get_feature_names_list(preprocessor_path):
    preprocessor = joblib.load(preprocessor_path)
    numerical_features = ['age', 'fare', 'pclass', 'sibsp', 'parch']
    categorical_features = ['sex', 'embarked']
    encoded_categorical_names = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
    feature_names = numerical_features + encoded_categorical_names
    return feature_names

def _extract_local_linear_function_from_nn(model, input_tensor):
    pre_activations = []
    def hook(module, input, output):
        pre_activations.append(input[0])

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.register_forward_hook(hook)

    output = model(input_tensor)

    weights, biases = [], []
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.data)
            biases.append(layer.bias)

    diagonal_matrices = []
    for Z in pre_activations:
        diagonal_matrix = create_diagonal_matrix(Z)
        diagonal_matrices.append(diagonal_matrix)

    W_eq, b_eq = compute_parameter_equivalents(diagonal_matrices, weights, biases)
    return W_eq, b_eq

if __name__ == '__main__':
    compare_local_explanation_to_shap(14)
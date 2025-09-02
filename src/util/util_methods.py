import joblib
import torch

def create_diagonal_matrix(pre_activation):
    on_off_state = (pre_activation > 0)
    diagonal_matrix = torch.diag(on_off_state.float().squeeze())
    return diagonal_matrix

def compute_parameter_equivalents(diagonals, weights, biases):
    # First layer
    W_eq = diagonals[0] @ weights[0]
    b_eq = diagonals[0] @ biases[0]
    # Second layer
    W_eq = (diagonals[1] @ weights[1]) @ W_eq
    b_eq = (diagonals[1] @ weights[1]) @ b_eq + (diagonals[1] @ biases[1])
    # Output layer
    W_eq = weights[2] @ W_eq
    b_eq = (weights[2] @ b_eq) + biases[2]
    return W_eq, b_eq

def create_feature_names_list(preprocessor_path):
    preprocessor = joblib.load(preprocessor_path)
    feature_names = preprocessor.get_feature_names_out()
    return feature_names

import numpy as np
import pandas as pd
from PIL import Image
import shap
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrices(target_names, nn_preds, lr_preds, dt_preds, labels):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Confusion Matrices', fontsize=16)

    cm_nn = confusion_matrix(labels, nn_preds)
    display_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=target_names)
    display_nn.plot(ax=axes[0], cmap='Greys')
    axes[0].set_title('Neural Network')

    cm_dtree = confusion_matrix(labels, dt_preds)
    display_dtree = ConfusionMatrixDisplay(confusion_matrix=cm_dtree, display_labels=target_names)
    display_dtree.plot(ax=axes[1], cmap='Greys')
    axes[1].set_title('Decision Tree')

    cm_lr = confusion_matrix(labels, lr_preds)
    display_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=target_names)
    display_lr.plot(ax=axes[2], cmap='Greys')
    axes[2].set_title('Logistic Regression')

    plt.tight_layout()

def combine_comparison_plots(sample_idx):
    img1 = Image.open(f"../figures/shap/local_explanations/local_plot_{sample_idx}.png")
    img2 = Image.open(f"../figures/shap/shap_explanations/shap_plot_{sample_idx}.png")

    width1, height1 = img1.size
    width2, height2 = img2.size

    total_width = width1 + width2
    max_height = max(height1, height2)
    combined_img = Image.new('RGBA', (total_width, max_height), color='white')

    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (width1, 0))

    output_filename = f"../figures/shap/comparisons/comparison_plot_{sample_idx}.png"
    combined_img.save(output_filename)

def plot_passenger_data(passenger_data, sample_idx):
    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.axis('off')

    title = f"Data for Passenger Index: {sample_idx}"
    ax.set_title(title, fontsize=14)

    table = ax.table(
        cellText=passenger_data.values,
        colLabels=passenger_data.columns,
        loc='center',
        cellLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.tight_layout()

def plot_explanation_as_waterfall(x_sample, W_eq, b_eq, feature_names):
    feature_contributions = W_eq * x_sample
    feature_contributions_np = feature_contributions[0].numpy()
    active_contributions = [feature for feature in feature_contributions_np if feature != 0]
    explanation  = shap.Explanation(
        values=np.array(active_contributions),
        base_values=b_eq,
        data=None,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, show=False)

def plot_local_explanation(feature_importance_list, passenger_idx, predicted_label, true_label):
    BLUE, RED = '#008BFB', '#FF0051'
    contributions = pd.Series(dict(feature_importance_list))
    colours_list = [RED if c > 0 else BLUE for c in contributions]
    clean_labels = [label.replace('cts__', '').replace('cat__', '') for label in contributions.index]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(contributions.index, contributions.values, color=colours_list)

    title = (
        f"(Neural Network) Local Feature Contributions for Passenger Index: {passenger_idx}\n"
        "\n"
        f"True Label: {true_label} | Model Prediction: {predicted_label}"
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Contribution to Model Output')
    ax.set_yticks(range(len(contributions)))
    ax.set_yticklabels(clean_labels)
    ax.invert_yaxis()

    _add_value_labels_to_bars(ax, contributions)
    ax.set_xlim(min(contributions) - 0.5, max(contributions) + 0.5)

    legend_elements = [
        Patch(facecolor=RED, edgecolor=RED, label='Pushes prediction towards "Survived"'),
        Patch(facecolor=BLUE, edgecolor=BLUE, label='Pushes prediction towards "Died"')
    ]
    ax.legend(handles=legend_elements, loc='best')
    plt.tight_layout()

def plot_global_explanation(global_feature_importance, local_active_features, passenger_idx, predicted_label, true_label):
    BLUE, RED = '#008BFB', '#FF0051'
    GREY = '#DCDCDC'
    contributions = pd.Series(dict(global_feature_importance))
    clean_labels = [label.replace('cts__', '').replace('cat__', '') for label in contributions.index]
    active_feature_names = [feature for feature, value in local_active_features]
    colours = _create_bar_colours_list(contributions, active_feature_names, RED, BLUE, GREY)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(contributions)), contributions.values, color=colours)

    title = (
        f"(Logistic Regression) Global Feature Contributions for Passenger Index: {passenger_idx}\n"
        "\n"
        f"True Label: {true_label} | Model Prediction: {predicted_label}"
    )
    ax.set_title(title, fontsize=14)
    ax.set_yticks(range(len(contributions)))
    ax.set_yticklabels(clean_labels)
    ax.invert_yaxis()

    _add_value_labels_to_bars(ax, contributions, colours=colours)
    ax.set_xlim(min(contributions) - 0.5, max(contributions) + 0.5)

    legend_elements = [
        Patch(facecolor=GREY, label='Inactive in Local Explanation'),
        Patch(facecolor=RED, label='Pushes prediction towards "Survived"'),
        Patch(facecolor=BLUE, label='Pushes prediction towards "Died"')
    ]
    ax.legend(handles=legend_elements, loc='best')
    plt.tight_layout()

def _add_value_labels_to_bars(ax, contributions, colours=None):
    BLUE, RED = '#008BFB', '#FF0051'
    GREY = '#DCDCDC'
    for i, value in enumerate(contributions):
        if colours is not None:
            if colours[i] == GREY:
                # Omit value labels for inactive features.
                continue
        if value > 0:
            colour = RED
            alignment = 'left'
            x_pos = value + 0.01
        else:
            colour = BLUE
            alignment = 'right'
            x_pos = value - 0.01

        text_label = f'{value:+.2f}'
        ax.text(x_pos, i, text_label, va='center', ha=alignment, color=colour, fontsize=12)

def _create_bar_colours_list(contributions, active_feature_names, RED, BLUE, GREY):
    colours = []
    for feature, value in contributions.items():
        if feature in active_feature_names:
            if value > 0:
                colours.append(RED)
            else:
                colours.append(BLUE)
        else:
            colours.append(GREY)
    return colours
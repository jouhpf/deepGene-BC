import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import combinations
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

def calculate_roc_with_cv(true_labels, prob_scores, cv_folds):
    """
    Calculate ROC curve and AUC under cross-validation, return mean and error bands
    """
    fpr_list = []
    tpr_list = []
    auc_list = []

    unique_folds = np.unique(cv_folds)

    for fold in unique_folds:
        # Get data for current fold
        mask = cv_folds == fold
        y_true_fold = true_labels[mask]
        y_prob_fold = prob_scores[mask]

        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_true_fold, y_prob_fold)
        roc_auc = auc(fpr, tpr)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc)

    # Interpolate to unified base_fpr grid
    base_fpr = np.linspace(0, 1, 1000)
    tpr_interp = []

    for i in range(len(fpr_list)):
        interp_tpr = np.interp(base_fpr, fpr_list[i], tpr_list[i])
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tpr_interp.append(interp_tpr)

    # Calculate mean and standard deviation
    mean_tpr = np.mean(tpr_interp, axis=0)
    std_tpr = np.std(tpr_interp, axis=0)

    # Calculate mean AUC and standard deviation
    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)

    return base_fpr, mean_tpr, std_tpr, mean_auc, std_auc


def plot_one_vs_rest_roc(df, save_path='one_vs_all_roc_combined.png'):
    """
    Plot four one-vs-rest ROC curves on a single figure
    """
    n_classes = 4

    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = ['red', 'blue', 'green', 'orange']
    class_names = ['Basal', 'Her2', 'LumA', 'LumB']
    for i in range(n_classes):
        # Prepare one-vs-rest labels
        y_true_binary = (df['true_label'] == i).astype(int)
        y_prob = df[f'class{i}_prob']
        cv_folds = df['fold'].values

        # Calculate ROC
        base_fpr, mean_tpr, std_tpr, mean_auc, std_auc = calculate_roc_with_cv(
            y_true_binary, y_prob, cv_folds
        )

        # Plot ROC curve and error bands
        ax.plot(base_fpr, mean_tpr, color=colors[i],
                label=f'{class_names[i]} vs Rest (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
                lw=2.5, alpha=0.8)

        # Plot error bands
        ax.fill_between(base_fpr,
                        mean_tpr - std_tpr,
                        mean_tpr + std_tpr,
                        color=colors[i], alpha=0.1)

    # Plot random guess line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Guess')

    # Set figure properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('One-vs-Rest ROC Curves with Cross-Validation Error Bands\n(All Classes)',
                 fontsize=16, fontweight='bold', pad=20)

    # Custom legend
    ax.legend(loc="lower right", fontsize=11)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_pairwise_roc(df, save_path='pairwise_roc.png'):
    """
    Plot ROC curves for pairwise class comparisons (one figure per pair)
    """
    class_pairs = list(combinations([0, 1, 2, 3], 2))
    n_pairs = len(class_pairs)
    class_names = ['Basal', 'Her2', 'LumA', 'LumB']

    # Set figure layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = sns.color_palette("husl", n_pairs)

    for idx, (i, j) in enumerate(class_pairs):
        # Filter samples belonging to class i or j
        mask = df['true_label'].isin([i, j])
        df_pair = df[mask].copy()

        # Convert labels: i as positive class, j as negative class
        y_true_binary = (df_pair['true_label'] == i).astype(int)

        # Use class i probability relative to class j
        y_prob_i = df_pair[f'class{i}_prob'].values
        y_prob_j = df_pair[f'class{j}_prob'].values

        # Normalize probabilities, handle zero probabilities
        prob_sum = y_prob_i + y_prob_j

        # Check and handle zero probabilities
        zero_mask = prob_sum == 0
        if zero_mask.any():
            print(f"Warning: {zero_mask.sum()} samples have zero probability for both classes {i} and {j}")
            # Option 1: Set neutral probability
            y_prob_normalized = np.zeros_like(prob_sum, dtype=float)
            y_prob_normalized[~zero_mask] = y_prob_i[~zero_mask] / prob_sum[~zero_mask]
            y_prob_normalized[zero_mask] = 0.5
        else:
            y_prob_normalized = y_prob_i / prob_sum

        cv_folds = df_pair['fold'].values

        # Check for NaN values
        if np.isnan(y_prob_normalized).any():
            print(f"Error: NaN values still exist for class {i} vs {j} after normalization")
            continue  # Skip this pair

        # Calculate ROC
        try:
            base_fpr, mean_tpr, std_tpr, mean_auc, std_auc = calculate_roc_with_cv(
                y_true_binary, y_prob_normalized, cv_folds
            )
        except ValueError as e:
            print(f"Error computing ROC for {class_names[i]} vs {class_names[j]}: {e}")
            continue

        # Plot
        ax = axes[idx]
        ax.plot(base_fpr, mean_tpr, color=colors[idx],
                label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)
        ax.fill_between(base_fpr,
                        mean_tpr - std_tpr,
                        mean_tpr + std_tpr,
                        color=colors[idx], alpha=0.1,
                        label=f'±1 std. dev.')

        # Random guess line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC: {class_names[i]} vs {class_names[j]}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)

    # Remove unused subplots
    if n_pairs < len(axes):
        for idx in range(n_pairs, len(axes)):
            fig.delaxes(axes[idx])

    plt.suptitle('Pairwise ROC Curves with Cross-Validation Error Bands',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_pr_with_cv(true_labels, prob_scores, cv_folds):
    """
    Calculate PR curve and AP under cross-validation, return mean and error bands
    """
    precision_list = []
    recall_list = []
    ap_list = []

    unique_folds = np.unique(cv_folds)

    for fold in unique_folds:
        # Get data for current fold
        mask = cv_folds == fold
        y_true_fold = true_labels[mask]
        y_prob_fold = prob_scores[mask]

        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(y_true_fold, y_prob_fold)
        ap = average_precision_score(y_true_fold, y_prob_fold)

        precision_list.append(precision)
        recall_list.append(recall)
        ap_list.append(ap)

    # Create unified recall grid (0 to 1)
    base_recall = np.linspace(0, 1, 1000)
    precision_interp = []

    for i in range(len(precision_list)):
        # PR curves have decreasing recall, reverse for interpolation
        recall = recall_list[i]
        precision = precision_list[i]

        # Ensure recall is monotonically increasing for interpolation
        if recall[0] > recall[-1]:
            recall = recall[::-1]
            precision = precision[::-1]

        # Add boundary points to ensure [0,1] coverage
        recall_interp = np.concatenate(([0], recall, [1]))
        precision_interp_vals = np.concatenate(([precision[0]], precision, [0]))

        # Use numpy interpolation (more stable)
        interp_precision = np.interp(base_recall, recall_interp, precision_interp_vals)
        precision_interp.append(interp_precision)

    # Calculate mean and standard deviation (clipped to [0,1])
    mean_precision = np.clip(np.mean(precision_interp, axis=0), 0, 1)
    std_precision = np.clip(np.std(precision_interp, axis=0), 0, 1)

    # Calculate mean AP and standard deviation
    mean_ap = np.mean(ap_list)
    std_ap = np.std(ap_list)

    return base_recall, mean_precision, std_precision, mean_ap, std_ap


def plot_one_vs_rest_pr(df, save_path='one_vs_all_pr_combined.png'):
    """
    Plot four one-vs-rest PR curves on a single figure
    """
    n_classes = 4

    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = ['red', 'blue', 'green', 'orange']
    class_names = ['Basal', 'Her2', 'LumA', 'LumB']
    for i in range(n_classes):
        # Prepare one-vs-all labels
        y_true_binary = (df['true_label'] == i).astype(int)
        y_prob = df[f'class{i}_prob']
        cv_folds = df['fold'].values

        # Calculate PR curve
        base_recall, mean_precision, std_precision, mean_ap, std_ap = calculate_pr_with_cv(
            y_true_binary, y_prob, cv_folds
        )

        # Plot PR curve and error bands
        ax.plot(base_recall, mean_precision, color=colors[i],
                label=f'{class_names[i]} vs Rest (AUPR = {mean_ap:.2f} ± {std_ap:.2f})',
                lw=2.5, alpha=0.8)

        # Plot error bands
        ax.fill_between(base_recall,
                        mean_precision - std_precision,
                        mean_precision + std_precision,
                        color=colors[i], alpha=0.1)

    # Plot random guess lines (positive class rates)
    for i in range(n_classes):
        pos_rate = (df['true_label'] == i).mean()
        ax.axhline(y=pos_rate, color=colors[i], linestyle=':', lw=1.5, alpha=0.5,
                   label=f'Random: {class_names[i]}' if i == 0 else "")

    # Set figure properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('One-vs-Rest PR Curves with Cross-Validation Error Bands\n(All Classes)',
                 fontsize=16, fontweight='bold', pad=20)

    # Custom legend
    ax.legend(loc="lower left", fontsize=11)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.show()

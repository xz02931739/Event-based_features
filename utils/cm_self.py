import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def cm_percentage(cm, labels, title=None, cmap=plt.cm.Blues, percentage=True, save_path=None):
    """
    show a confusion matrix with percentage
    :param cm: confusion matrix
    :param labels: labels
    :param title: title
    :param cmap: color map
    :return: None
    """
    ### fix color bar range 0-100.
    ### let xticks rotate 45 degree

    plt.figure(figsize=(4.2, 2.8))
    if percentage:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # cm = cm * 100
        cm = np.around(cm, 2)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues',
            xticklabels=[f'Predicted {c}' for c in labels],
            yticklabels=[f'True {c}' for c in labels],
            vmin=0, vmax=1, )
    else:
        cm = cm
        sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues',
            xticklabels=[f'Predicted {c}' for c in labels],
            yticklabels=[f'True {c}' for c in labels])
    plt.title(title)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    
    else:
        plt.show()

    return cm


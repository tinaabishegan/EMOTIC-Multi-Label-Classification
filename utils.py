import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report

def test_scikit_ap(cat_preds, cat_labels):
    """
    Compute mean Average Precision (mAP) for categorical predictions.
    """
    ap = np.zeros(26, dtype=np.float32)
    for i in range(26):
        ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
    print('ap', ap, ap.shape, ap.mean())
    return ap.mean()

def test_emotic_vad(cont_preds, cont_labels):
    """
    Compute mean absolute error (mAE) for continuous predictions.
    """
    vad = np.zeros(3, dtype=np.float32)
    for i in range(3):
        vad[i] = np.mean(np.abs(cont_preds[i, :] - cont_labels[i, :]))
    print('vad', vad, vad.shape, vad.mean())
    return vad.mean()

def get_thresholds(cat_preds, cat_labels):
    """
    Compute thresholds for each emotion from the precision-recall curve.
    """
    thresholds = np.zeros(26, dtype=np.float32)
    for i in range(26):
        p, r, t = precision_recall_curve(cat_labels[i, :], cat_preds[i, :])
        for k in range(len(p)):
            if p[k] == r[k]:
                thresholds[i] = t[k]
                break
    np.save('./thresholds.npy', thresholds)
    return thresholds

def plot_loss_curve(train_losses, val_losses, filename="loss_curve.png"):
    """
    Plot training and validation loss over epochs.
    """
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(filename)


def plot_metric_curve(train_metrics, val_metrics, filename="metric_curve.png"):
    """
    Plot training and validation metric (mAP or mAE) over epochs.
    """
    plt.figure()
    plt.plot(train_metrics, label="Train Metric")
    plt.plot(val_metrics, label="Validation Metric")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Metric Curve")
    plt.legend()
    plt.savefig(filename)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', filename= "confusion_matrix.png",cmap=plt.cm.Blues):
    """
    Plot a confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=6)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)

def compute_classification_metrics(y_true, y_pred):
    """
    Compute and return classification metrics: precision, recall, F1-score, and support.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    return report

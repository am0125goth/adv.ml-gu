import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error

def plot_training_curve(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation loss")
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def eval_threshold(y_true, y_prob, thresholds=0.5):
    if not isinstance(thresholds, (list, np.ndarray)):
        thresholds = [thresholds]

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_true = np.squeeze(y_true)
    y_prob = np.squeeze(y_prob)
    
    results = []
    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        results.append({"threshold": float(t),
                        "precision": precision_score(y_true.flatten(), y_pred.flatten()),
        "recall": recall_score(y_true.flatten(), y_pred.flatten()),
        "f1": f1_score(y_true.flatten(), y_pred.flatten()),
        "accuracy": accuracy_score(y_true.flatten(), y_pred.flatten())
    })
    return results
        

def iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection/union if union > 0 else 0

def dice(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2. * intersection)/(y_true.sum() + y_pred.sum() + 1e8)

def eval_testset(test_preds, test_masks, threshold=0.5):
    y_pred_bin = (test_preds > threshold).astype(int)
    iou_scores = [iou(gt, pr) for gt, pr in zip(test_masks, y_pred_bin)]
    dice_scores = [dice(gt, pr) for gt, pr in zip(test_masks, y_pred_bin)]
    mse_scores = [mean_squared_error(gt.flatten(), pr.flatten()) for gt, pr in zip(test_masks, test_preds)]
    return {
        "mean IoU": np.mean(iou_scores),
        "mean Dice": np.mean(dice_scores),
        "mean MSE": np.mean(mse_scores)
    }

def visualize_pred(img, mask, pred_prob, threshold=0.5):
    pred_mask = (pred_prob > threshold).astype(int)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img)
    ax[0].set_title("original Image")

    ax[1].imshow(mask.squeeze(), cmap="gray")
    ax[1].set_title("Ground Truth")

    ax[2].imshow(img, alpha=0.6)
    ax[2].imshow(pred_prob.squeeze(), cmap="jet", alpha=0.4)
    ax[2].set_title("Prediction overlay")

    plt.show()


    
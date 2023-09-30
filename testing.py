import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt


def test(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)
    lb = LabelBinarizer().fit(y_test)
    y_label = lb.transform(y_test)
    n_class = 10
    fpr = []
    tpr = []
    thresholds = []
    # Calculate ROC curve
    for i in range(n_class):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_label, y_pred[:,i], pos_label=i)
        roc_auc = roc_auc_score(y_label, y_pred[:,i], pos_label=i)

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test AUC Score: {roc_auc * 100:.2f}%")
    return test_loss, test_accuracy, fpr, tpr


def get_plot(history, fpr, tpr):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.plot(fpr[0], tpr[0], linestyle='--')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')

    plt.show()
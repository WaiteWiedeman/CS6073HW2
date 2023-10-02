import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay, f1_score
from itertools import cycle
import matplotlib.pyplot as plt


# defining test function
# takes model, testing data, and number of classes as input
# returns test loss, test accuracy, prediction probabilities, and labels
def test(model, X_test, y_test, n_classes):
    # model evaluate function to get test loss and accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    # get y predictions for AUC plot
    y_pred = model.predict(X_test)
    # get binary y prediction for f1 score
    threshold = 0.5  # threshold for binary conversion
    y_pred_binary = (y_pred > threshold).astype(int)
    # binarize labels for AUC plot
    lb = LabelBinarizer().fit(y_test)
    y_label = lb.transform(y_test)
    # calculate f1 score
    f1 = f1_score(y_test, y_pred_binary, average=None)
    print(f"F1 Scores: {f1}")  # print f1 scores

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")  # print test accuracy

    return test_loss, test_accuracy, y_pred, y_label


# define function to plot model performance
# takes model history, number of classes, y predictions, and y labels as input
def get_plot(history, n_classes, y_pred, y_label):
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
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'fuchsia',
                    'red', 'brown', 'green', 'cyan', 'gray', 'purple'])
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_label[:, i],
            y_pred[:, i],
            name=f"ROC curve for {target_names[i]}",
            color=color,
            ax=ax,
            plot_chance_level=(i == 2),
        )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    plt.show()

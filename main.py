# import packages
from models import ResNet18, CNN, DNN
from getdata import get_data
from training import training
from testing import test, get_plot
from modelvis import model_visualization


learn_rate = 0.001  # model learning rate
epochs = 10  # number of epochs for training model
n_classes = 10  # number of classes in dataset

data = get_data()  # call get_data function to get processed data set
# split dataset between training/testing features and classes
X_train = data[0]
y_train = data[1]
X_test = data[2]
y_test = data[3]
# reshape training and testing features for deep neural network
X_train_dnn = X_train.reshape(-1,28*28)
X_test_dnn = X_test.reshape(-1, 28*28)
# get models
res = ResNet18()
cnn = CNN()
dnn = DNN()
# train, test, and plot models

res_history = training(res, X_train, y_train, learn_rate, epochs)
res_test_loss, res_test_accuracy, res_y_pred, res_y_label = test(res, X_test, y_test, n_classes)
get_plot(res_history, n_classes, res_y_pred, res_y_label)
res.save('ResNet.keras')
model_visualization(res)

cnn_history = training(cnn, X_train, y_train, learn_rate, epochs)
cnn_test_loss, cnn_test_accuracy, cnn_y_pred, cnn_y_label = test(cnn, X_test, y_test, n_classes)
get_plot(cnn_history, n_classes, cnn_y_pred, cnn_y_label)
cnn.save('CNN.keras')
model_visualization(cnn)

dnn_history = training(dnn, X_train_dnn, y_train, learn_rate, epochs)
dnn_test_loss, dnn_test_accuracy, dnn_y_pred, dnn_y_label = test(dnn, X_test_dnn, y_test, n_classes)
get_plot(dnn_history, n_classes, dnn_y_pred, dnn_y_label)
dnn.save('DNN.keras')
model_visualization(dnn)

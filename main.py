from models import ResNet18, CNN, DNN
from getdata import get_data
from training import training
from testing import test, get_plot


learn_rate = 0.001
epochs = 1

data = get_data()

X_train = data[0]
y_train = data[1]
X_test = data[2]
y_test = data[3]
X_train_dnn = X_train.reshape(-1,28*28)
X_test_dnn = X_test.reshape(-1, 28*28)

res = ResNet18()
cnn = CNN()
dnn = DNN()
''''
res_history = training(res, X_train, y_train, learn_rate, epochs)
res_test_loss, res_test_accuracy, res_fpr, res_tpr = test(res, X_test, y_test)
get_plot(res_history, res_fpr, res_tpr)

cnn_history = training(cnn, X_train, y_train, learn_rate, epochs)
cnn_test_loss, cnn_test_accuracy, cnn_fpr, cnn_tpr = test(cnn, X_test, y_test)
get_plot(cnn_history, cnn_fpr, cnn_tpr)
'''''
dnn_history = training(dnn, X_train_dnn, y_train, learn_rate, epochs)
dnn_test_loss, dnn_test_accuracy, dnn_fpr, dnn_tpr = test(dnn, X_test_dnn, y_test)
get_plot(dnn_history, dnn_fpr, dnn_tpr)

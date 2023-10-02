from keras.utils import plot_model
import pydot
import pydotplus
import graphviz
from pydotplus import graphviz


def feature_visualization(model):
    plot_model(model, to_file=f'{model}.png')



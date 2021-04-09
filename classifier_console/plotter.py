# For plot
import matplotlib.pyplot as plt

# For make confusion matrix
from sklearn.metrics import confusion_matrix

import numpy as np

class Plotter:

    @staticmethod
    def plot_lstm(history, num_tickets):
        import matplotlib.pyplot as plt

        # График точности модели
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig("Acc_" + num_tickets + ".png")

        # График оценки loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig("Loss_" + num_tickets + ".png")

    @staticmethod
    def plot_train_result(epochs, history, num_tickets):

        # %matplotlib inline
        plt.style.use("ggplot")
        plt.figure()
        N = epochs
        plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
        plt.title("Эффективность обучения")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.show()

        plt.savefig("Locc_Acc_" + num_tickets + ".png")

    @staticmethod
    def make_confusion_matrix(model, X_test, y_test, label_categories, num_tickets):
        y_softmax = model.predict(X_test)

        y_test_1d = []
        y_pred_1d = []

        for i in range(len(y_test)):
            probs = y_test[i]
            index_arr = np.nonzero(probs)
            one_hot_index = index_arr[0].item(0)
            y_test_1d.append(one_hot_index)

        for i in range(0, len(y_softmax)):
            probs = y_softmax[i]
            predicted_index = np.argmax(probs)
            y_pred_1d.append(predicted_index)

        cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
        plt.figure(figsize=(48, 40))
        Plotter.plot_confusion_matrix(cnf_matrix, classes=label_categories, title="Confusion matrix")
        plt.show()

        plt.savefig("Confusion_Matrix_" + num_tickets + ".png")

        del model
        import tensorflow.keras.backend as K
        K.clear_session()  # removing session, it will instance another
        import gc
        gc.collect()

    @staticmethod
    def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, normalize=True):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        import itertools

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=30)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
        plt.yticks(tick_marks, classes, fontsize=22)

        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Правильная категория', fontsize=25)
        plt.xlabel('Определенная моделью категория', fontsize=25)

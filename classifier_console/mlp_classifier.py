import pandas as pd
import numpy as np
from numpy import unicode

import pymorphy2 as pm
import re

# For ML
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.utils

# import tensorflow.keras.utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing import sequence

# For plot
import matplotlib.pyplot as plt

# For make confusion matrix
from sklearn.metrics import confusion_matrix

class MLP_classifier:

    @staticmethod
    def clean_text(text):
        text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
        text = text.lower()
        text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)  # deleting newlines and line-breaks
        text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # deleting symbols
        text = " ".join(pm.MorphAnalyzer().parse(unicode(word))[0].normal_form for word in text.split())
        text = ' '.join(word for word in text.split() if len(word) > 3)
        return text

    @staticmethod
    def get_data_from_db(self, num_tickets):
        stmt = "SELECT * FROM {database}.{table} WHERE id < {num_tickets}".format(
            database=self.dbinfo['database'],
            table=self.dbinfo['table'],
            num_tickets=int(num_tickets) + 1)

        # Read SQL data
        df = pd.read_sql(stmt, self.connection)
        print("Текст до обработки")
        print(df[['ticket_queue_name', 'article_a_subject', 'article_a_body']])

        # Clean text in SQL data
        # df['article_a_body'] = df.apply(lambda x: LSTM_classifier.clean_text(x[u'article_a_body']), axis=1)
        # print("Текст после обработки")
        # print(df[['ticket_queue_name', 'article_a_subject', 'article_a_body']])

        # создадим массив, содержащий уникальные категории из нашего DataFrame
        categories = {}
        for key, value in enumerate(df[u'ticket_queue_name'].unique()):
            categories[value] = key + 1

        label_categories = df['ticket_queue_name'].unique()
        print("Список категорий заявок из БД для классификации: ", label_categories)

        # Запишем в новую колонку числовое обозначение категории
        df['ticket_queue_name'] = df[u'ticket_queue_name'].map(categories)

        total_categories = len(df[u'ticket_queue_name'].unique())
        print('Всего категорий: {}'.format(total_categories))

        # df = df.sample(frac=1).reset_index(drop=True)

        return df, total_categories, label_categories

    @staticmethod
    def get_1_ticket_from_db(self, num_tickets, label_categories_const):
        stmt = "SELECT * FROM {database}.{table} WHERE id = {num_tickets}".format(
            database=self.dbinfo['database'],
            table=self.dbinfo['table'],
            num_tickets=num_tickets)

        # Read SQL data
        ticket = pd.read_sql(stmt, self.connection)
        print("ticket for classify  = \n", ticket[['ticket_id', 'ticket_queue_name', 'article_a_body']])

        # X prepare ==============================================================
        descriptions = ticket['article_a_body']

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(descriptions.tolist())
        textSequences = tokenizer.texts_to_sequences(descriptions.tolist())
        # print("textSequences = ", textSequences)

        data_size = len(textSequences)
        X_train = textSequences[:data_size]
        # print("X_train = ", X_train)

        # Максимальное количество слов в самом длинном описании заявки
        max_words = 0
        for desc in descriptions.tolist():
            words = len(desc.split())
            if words > max_words:
                max_words = words

        total_unique_words = len(tokenizer.word_counts)

        tokenizer = Tokenizer(num_words=total_unique_words)
        tokenizer.fit_on_texts(ticket)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_train = sequence.pad_sequences(X_train, maxlen=max_words)

        # Y prepare ==============================================================
        ind = np.where(label_categories_const == ticket[u'ticket_queue_name'].tolist()[0])
        y_train = to_categorical(np.arange(8), len(np.arange(8)))

        return X_train, y_train

    @staticmethod
    def data_classify_mlp(self):
        label_categories_const = np.array(
            ["Диспетчер ЭДО", "Техотдел", "TechSupport", "Подрядчики", "VideoControl", "ОКФ", "Network_Dept",
             "Develop_Dept"])

        mode = ""
        while len(mode) == 0:
            mode = input("Вы хотите протестировать нейронку (0) или обучить (1)?\t")
            if int(mode) == 0:
                num_ticket = input("Укажите номер заявки в БД для классификации:\t")
                # Достаём заявку из БД и готовим её к подачи в нейронку
                ticket_text, labels = MLP_classifier.get_1_ticket_from_db(self, num_ticket, label_categories_const)

                MLP_classifier.model_predict_without_train(label_categories_const, ticket_text, labels)

            elif int(mode) == 1:
                num_tickets = input("Укажите количество заявок для обучения:\t")

                # Достаём данные из БД для подготовки их к подаче в нейронку
                df, total_categories, label_categories = \
                    MLP_classifier.get_data_from_db(self, num_tickets)

                # Подготовка входных для нейронной сети
                MLP_classifier.model_training(self, df, num_tickets, label_categories)

            else:
                print("Выберите один из предложенных режимов работы.")
                mode = ""

    @staticmethod
    def model_training(self, df, num_tickets, label_categories):

        # Sampling
        descriptions = df['article_a_body']
        categories = df[u'ticket_queue_name']

        # Keras
        # создаем единый словарь (слово -> число) для преобразования
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(descriptions.tolist())

        # Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
        textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

        X_train, y_train, X_test, y_test = self.load_data_from_arrays(textSequences, categories, train_test_split=0.8)

        # Посчитаем максимальную длинну текста описания в словах
        max_words = 0
        for desc in descriptions:
            words = len(desc.split())
            if words > max_words:
                max_words = words
        print('Максимальная длина описания: {} слов'.format(max_words))
        maxSequenceLength = max_words

        total_words = len(tokenizer.word_index)
        print('В словаре {} слов'.format(total_words))

        # количество наиболее часто используемых слов
        num_words = total_words

        print(u'Преобразуем описания заявок в векторы чисел...')
        tokenizer = Tokenizer(num_words=num_words)
        X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
        X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
        print('Размерность X_train:', X_train.shape)
        print('Размерность X_test:', X_test.shape)

        print(u'Преобразуем категории в матрицу двоичных чисел '
              u'(для использования categorical_crossentropy)')

        num_classes = np.max(y_train) + 1

        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)


        # MLP многослойный перцептрон -------------------------------------------------------
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import Dropout

        with tf.device('/gpu:1'):
            # количество эпох\итераций для обучения
            epochs = 3

            print(u'Собираем модель...')
            model = Sequential()
            model.add(Dense(512, input_shape=(num_words,)))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Dense(num_classes))
            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            print(model.summary())

            # Learning MLP
            history = model.fit(X_train, y_train,
                                batch_size=32,
                                epochs=epochs,
                                verbose=1,
                                validation_split=0.1)

            score = model.evaluate(X_test, y_test,
                                   batch_size=32, verbose=1)

            print(u'Оценка теста: {}'.format(score[0]))
            print(u'Оценка точности модели: {}'.format(score[1]))

            print('Сохраняем модель')
            model.save("mlp_" + str(num_tickets) + ".h5")
            model.save_weights('./checkpoints/mpl/2000')
            print("Сохранение завершено!")

        MLP_classifier.plot_mlp(history)
        # При желании снова можно построить график эффективности процесса обучения:
        # Посмотрим на эффективность обучения
        MLP_classifier.plot_train_result(epochs, history)
        MLP_classifier.make_confusion_matrix(model, X_test, y_test, label_categories)

        # Тестирование модели нейронки
        MLP_classifier.model_predict_with_train(model, X_test, y_test, label_categories)

    @staticmethod
    def model_predict_with_train(model, X_test, y_test, label_categories):
        print("Категории: ", label_categories)
        for i in range(0, 30):
            prediction = model.predict(np.array([X_test[i]]))
            print("prediction ", prediction)
            predicted_label = label_categories[(np.argmax(prediction))]
            print(X_test[i][50], "...")
            print('Правильная категория: {}'.format(label_categories[np.argmax(y_test, axis=1)[i]]))
            print("Определенная моделью категория: {}".format(predicted_label))

    @staticmethod
    def model_predict_without_train(label_categories_const, ticket_text, labels):
        print("ticket:  ", ticket_text)

        # Загружаем обученную модель
        model_loaded = tensorflow.keras.models.load_model('mlp_1000.h5')
        # Покажем архитектуру модели
        # model_loaded.summary()

        # loss, acc = model_loaded.evaluate(ticket_text, labels, batch_size=32, verbose=1)
        # print("Точность восстановленной модели: {:5.2f}%".format(100 * acc))

        prediction = model_loaded.predict(np.array(ticket_text))
        predicted_label = label_categories_const[(np.argmax(prediction))]
        print("Категории: ", label_categories_const)
        print("prediction: ", prediction)
        print("\n\tОпределенная моделью категория: {}".format(predicted_label))

    @staticmethod
    def plot_mlp(history):
        import matplotlib.pyplot as plt

        # График точности модели
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # График оценки loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    @staticmethod
    def plot_train_result(epochs, history):

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

    @staticmethod
    def make_confusion_matrix(model, X_test, y_test, label_categories):
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
        MLP_classifier.plot_confusion_matrix(cnf_matrix, classes=label_categories, title="Confusion matrix")
        plt.show()

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

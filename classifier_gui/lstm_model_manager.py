import os
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing import sequence

from mysql.script.setter import Setter
from plotter import Plotter

from interface.ui_manager import *

class LSTM_model_manager:

    def __init__(self, classifier_object):

        self.classifier_object = classifier_object
        # self.comm = communicator
        self.comm = classifier_object.comm
        self.label_categories_const = np.array(["Диспетчер ЭДО", "Техотдел", "TechSupport", "Подрядчики",
                                                "VideoControl", "ОКФ", "Network_Dept", "Develop_Dept"])
        self.logger = classifier_object.get_logger()

    def __del__(self):
        # Clear memory -----------------------------------------------------------------------
        import tensorflow.keras.backend as K
        K.clear_session()  # removing session, it will instance another
        import gc
        gc.collect()

    def set_connect(self):

        self.comm.window_log_message[str].connect(self.classifier_object.ui.log_writer)

        self.comm.thread_start.connect(self.classifier_object.ui.log_writer)
        self.comm.thread_end.connect(self.classifier_object.ui.log_writer)

    # Train model ----------------------------------------------------------------------------
    def prepare_data_for_trainig(self, df):

        # Sampling
        descriptions = df['article_a_body']
        categories = df[u'ticket_queue_name']

        # создаем единый словарь (слово -> число) для преобразования
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(descriptions.tolist())

        # Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
        textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

        X_train, y_train, X_test, y_test = self.classifier_object.load_data_from_arrays(textSequences, categories, train_test_split=0.8)

        # Максимальное количество слов в самом длинном описании заявки
        max_words = 0
        for desc in descriptions.tolist():
            words = len(desc.split())
            if words > max_words:
                max_words = words
        print('Максимальное количество слов в самом длинном описании заявки: {} слов'.format(max_words))
        self.comm.window_log_message.emit("Максимальное количество слов в самом длинном описании заявки: " + str(max_words) + " слов")

        total_unique_words = len(tokenizer.word_counts)
        print('Всего уникальных слов в словаре: {}'.format(total_unique_words))
        self.comm.window_log_message.emit("Всего уникальных слов в словаре: " + str(total_unique_words))

        maxSequenceLength = max_words

        # Для уменьшения количества расчетов в модели уменьшим общий словарь, оставив в нем только 10% наиболее популярных слов:
        vocab_size = round(total_unique_words / 10)
        # vocab_size = total_unique_words

        # Далее преобразуем данные для тренировки и тестирования в нужный нам формат:
        # print(u'Преобразуем описания заявок в векторы чисел...')
        # tokenizer = Tokenizer(num_words=vocab_size)
        # tokenizer.fit_on_texts(descriptions)
        # X_train = tokenizer.texts_to_sequences(descriptions)
        # X_test = tokenizer.texts_to_sequences(descriptions)

        X_train = sequence.pad_sequences(X_train, maxlen=maxSequenceLength)
        X_test = sequence.pad_sequences(X_test, maxlen=maxSequenceLength)

        print('Размерность X_train:', X_train.shape)
        print('Размерность X_test:', X_test.shape)
        self.comm.window_log_message.emit("Размерность X_train: " + str(X_train.shape))
        self.comm.window_log_message.emit("Размерность X_test: " + str(X_test.shape))

        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train = encoder.transform(y_train)
        y_test = encoder.transform(y_test)

        # Считаем количество категорий:
        num_classes = np.max(y_train) + 1
        print('Количество категорий для классификации: {}'.format(num_classes))
        self.comm.window_log_message.emit('Количество категорий для классификации: ' + str(num_classes))

        print(u'Преобразуем категории в матрицу двоичных чисел '
              u'(для использования categorical_crossentropy)')
        self.comm.window_log_message.emit(u'Преобразуем категории в матрицу двоичных чисел '
                               u'(для использования categorical_crossentropy)' + str(num_classes))

        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        return X_train, y_train, X_test, y_test, num_classes, maxSequenceLength, vocab_size

    def model_training(self, num_tickets, label_categories,
                       X_train, y_train,
                       X_test, y_test,
                       num_classes, maxSequenceLength, vocab_size,
                       batch_size, gpu_memory_fraction):

        # self.emit(SIGNAL('model_train_begin()'))

        # Определяем LSTM модель для обучения: =============================================
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Embedding, LSTM
        from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D

        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        TF_CONFIG_ = tf.compat.v1.ConfigProto(allow_soft_placement=True,  log_device_placement=True)
        TF_CONFIG_.gpu_options.allow_growth = True
        TF_CONFIG_.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction # 0.5
        TF_CONFIG_.gpu_options.visible_device_list = "0"

        with tf.compat.v1.Session(config=TF_CONFIG_) as sess:
            # максимальное количество слов для анализа
            max_features = vocab_size + 1

            print(u'Собираем модель...')
            self.comm.window_log_message.emit(u'Собираем модель...')

            model = Sequential()
            model.add(Embedding(max_features, maxSequenceLength))
            # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
            # model.add(MaxPooling1D(pool_size=3))
            # model.add(Dropout(0.2))

            # model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
            # model.add(MaxPooling1D(pool_size=2))
            # model.add(Dropout(0.3))
            # model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
            # model.add(MaxPooling1D(pool_size=2))
            # model.add(Dropout(0.3))
            # model.add(Conv1D(filters=32, kernel_size=7, padding='same', activation='relu'))
            # model.add(MaxPooling1D(pool_size=2))
            # model.add(Dropout(0.3))
            # model.add(Conv1D(filters=32, kernel_size=9, padding='same', activation='relu'))
            # model.add(MaxPooling1D(pool_size=2))
            # model.add(Dropout(0.3))
            # model.add(Conv1D(filters=32, kernel_size=12, padding='same', activation='relu'))
            # model.add(MaxPooling1D(pool_size=2))
            # model.add(Dropout(0.3))
            # model.add(Conv1D(filters=32, kernel_size=15, padding='same', activation='relu'))
            # model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(num_classes, activation='sigmoid'))

            with tf.device('/cpu:0'):
                sess.run(tf.compat.v1.global_variables_initializer())

                model.compile(loss='binary_crossentropy',
                                    optimizer='adam',
                                    metrics=['accuracy'])

                print(model.summary())
                self.comm.window_log_message.emit(str(model.summary()))

                # Обучаем
                # batch_size = 4
                epochs = 3

                print(u'Тренируем модель...')
                self.comm.window_log_message.emit(u'Тренируем модель...')

                history = model.fit(X_train, y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_data=(X_test, y_test))

                # По окончанию процесса обучения оценим его результаты:
                score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

                print(u'Оценка теста: {}'.format(score[0]))
                print(u'Оценка точности модели: {}'.format(score[1]))

                self.comm.window_log_message.emit(u'Оценка теста: {}'.format(score[0]))
                self.comm.window_log_message.emit(u'Оценка точности модели: {}'.format(score[1]))

                print('Сохраняем модель')
                self.comm.window_log_message.emit('Сохраняем модель')

                model.save("lstm_" + str(num_tickets) + ".h5")
                # model.save_weights('./checkpoints/lstm/2000')
                print("Сохранение завершено!")
                self.comm.window_log_message.emit('Сохранение завершено!')

                # При желании снова можно построить график эффективности процесса обучения:
                # Посмотрим на эффективность обучения

                # Раздельные графики
                Plotter.plot_lstm(history, num_tickets)
                # Обьединённые графики
                Plotter.plot_train_result(epochs, history, num_tickets)
                # Конфусионная матрица
                Plotter.make_confusion_matrix(model, X_test, y_test, label_categories, num_tickets)

                # Тестирование модели нейронки
                LSTM_model_manager.model_predict_with_train(model, X_test, y_test, label_categories)

        # self.emit(SIGNAL('model_train_end()'))
        return model

    def model_predict_with_train(self, model, X_test, y_test, label_categories):

        # Тестовые данные для нейронки
        # text_labels = encoder.classes_
        # print("Категории: ", text_labels)

        # for i in range(20):
        #     prediction = model.predict(np.array([X_test[i]]))
        #     predicted_label = text_labels[np.argmax(prediction)]
        #     print(X_test.iloc[i][:50], "...")
        #     print('Правильная категория: {}'.format(y_test.iloc[i]))
        #     print("Определенная моделью категория: {}".format(predicted_label))

        print("Категории: ", label_categories)
        self.comm.window_log_message.emit("Категории: " + str(label_categories))

        for i in range(0, 30):
            prediction = model.predict(np.array([X_test[i]]))
            # print("argmax_predict ", np.argmax(prediction))
            # print("argmax_ytest", np.argmax(y_test, axis=1))
            predicted_label = label_categories[(np.argmax(prediction))]

            msg_cmd = "\n* prediction: " + str(prediction) + \
                  "\n* X_test:  " + str(X_test[i][50], "...") + \
                  "\n* Категория из БД: " + str(label_categories[np.argmax(y_test, axis=1)[i]]) + \
                  "\n* Определенная моделью категория: " + str(predicted_label)

            msg_ui = "\n* prediction: " + str(prediction) + \
                  "\n* Категория из БД: " + str(label_categories[np.argmax(y_test, axis=1)[i]]) + \
                  "\n* Определенная моделью категория: " + str(predicted_label)

            print(msg_cmd)
            self.comm.window_log_message.emit(msg_ui)

        return 0

    def model_predict_without_train(self, num_ticket):

        print("=== model_predict_without_train ===")

        model_loaded = tensorflow.keras.models.load_model('lstm_30000.h5')
        ticket_text, ticket = Getter.get_1_ticket_from_db(self.classifier_object, num_ticket)
        prediction = model_loaded.predict(np.array(ticket_text))
        predicted_label = self.label_categories_const[(np.argmax(prediction))]

        # Покажем архитектуру модели и оценим точность
        # model_loaded.summary()
        # loss, acc = model_loaded.evaluate(ticket_text, labels, batch_size=32, verbose=1)
        # print("Точность восстановленной модели: {:5.2f}%".format(100 * acc))

        msg =   "\n* prediction: " + str(prediction) + \
                "\n* ticket_id:  " + str(ticket.iloc[0]['ticket_id']) + \
                "\n* ticket text:  " + str(ticket.iloc[0]['article_a_body']) + \
                "\n* Категория из БД: " + str(ticket.iloc[0]['ticket_queue_name']) + \
                "\n* Определенная моделью категория: " + str(predicted_label)

        print(msg)
        self.comm.window_log_message.emit(msg)
        return 0

    def model_predict_without_train_auto(self, begin_num, end_num):

        model_loaded = tensorflow.keras.models.load_model('lstm_30000.h5')

        # begin_num = int(self.classifier_object.ui.window.beginTicketIndex)
        # end_num = int(self.classifier_object.ui.window.endTicketIndex)

        for i in range(begin_num, end_num):
            ticket_text, ticket = Getter.get_1_ticket_from_db(self.classifier_object, i)
            if ticket_text.size == 0:
                continue
            prediction = model_loaded.predict(np.array(ticket_text))
            predicted_label = self.label_categories_const[(np.argmax(prediction))]

            msg = "\n* prediction: " + str(prediction) + \
                  "\n* ticket_id:  " + str(ticket.iloc[0]['ticket_id']) + \
                  "\n* ticket text:  " + str(ticket.iloc[0]['article_a_body']) + \
                  "\n* Категория из БД: " + str(ticket.iloc[0]['ticket_queue_name']) + \
                  "\n* Определенная моделью категория: " + str(predicted_label)

            print(msg)
            self.comm.window_log_message.emit(msg)

            Setter.update_ticket_on_db(self.classifier_object, ticket, predicted_label)
        return 0

    # For show tickets nums with correct prediction (dev) -------------------------------------------
    def check_correct_predict_for_all_tickets(self, model_loaded, num_tickets):
        strnums = [ ]
        count = 0
        for i in range(1, int(num_tickets)):
            ticket_text, ticket = Getter.get_1_ticket_from_db(self.classifier_object, i)
            if ticket_text.size == 0:
                continue
            prediction = model_loaded.predict(np.array(ticket_text))
            predicted_label = self.label_categories_const[(np.argmax(prediction))]
            # Сохранение в файл номеров заявок, у которых предсказание сошлось с тем, что было в БД
            if ticket.iloc[0]['ticket_queue_name'] == predicted_label:
                strnums.append(ticket.iloc[0]['ticket_id'])
                with open('correct_tickets.txt', 'w+') as f:
                    f.write("\nCount: %s\t" % count)
                    f.write("\nTicket_ID: %s\t" % ticket.iloc[0]['ticket_id'])
                    f.write("\tВерная категория: {}".format(ticket.iloc[0]['ticket_queue_name']))
                    f.write("\tОпределенная моделью категория: {}".format(predicted_label))
                count+=1

        with open('correct_ids.txt', 'w+') as f:
            f.write("Количество верных: %s" % count)
            for item in strnums:
                f.write("\n%s" % item)

        return  0

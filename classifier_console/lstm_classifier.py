import numpy as np

# For ML
import tensorflow as tf
import os
import tensorflow.keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing import sequence

from mysql.script.getter import Getter
from mysql.script.setter import Setter
from mysql.script.excel_db import Excel
from mysql.script.db_format import DB_formalizer
from plotter import Plotter


class LSTM_classifier:

    @staticmethod
    def data_classify_lstm(self):
        # Filter out messages ----------------------------------------------------------------
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('INFO')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        # Clear memory -----------------------------------------------------------------------
        import tensorflow.keras.backend as K
        K.clear_session()  # removing session, it will instance another
        import gc
        gc.collect()

        # Logger -----------------------------------------------------------------------------
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)  # process everything, even if everything isn't printed
        # logger.setLevel(logging.DEBUG)  # process everything, even if everything isn't printed

        file_log_handler = logging.FileHandler('logfile.log')
        logger.addHandler(file_log_handler)

        stderr_log_handler = logging.StreamHandler()
        logger.addHandler(stderr_log_handler)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)

        # logger.info('Info message')
        # logger.error('Error message')

        # Menu ------------------------------------------------------------------------------
        label_categories_const = np.array(
            ["Диспетчер ЭДО", "Техотдел", "TechSupport", "Подрядчики", "VideoControl", "ОКФ", "Network_Dept",
             "Develop_Dept"])

        mode = ""
        while len(mode) == 0:
            mode = input("\nВыберите режим работы: "
                         "\n 0 - обучить модель, вывести графики эффективности, потерь и матрицы потерь"
                         "\nОбработка данных в автоматическом режиме:"
                         "\n 1 - классифицировать все заявки за указанные даты"
                         "\nОбработка данных в ручном режиме:"
                         "\n 2 - классифицировать одну заявку"
                         "\n 3 - классифицировать несколько заявок"
                         "\n 4 - экспортировать заявки из БД OTRS"
                         "\n 5 - форматировать текст"
                         # "\n4 - подсчитать верные предсказания в датасете"
                         "\nВыбраный режим: \t")

            if int(mode) == 0: # Обучить -------------------------------------------------------
                num_tickets = int(input("Укажите количество заявок для обучения:\t"))
                batch_size = int(input("Размер порции (количество заявок, подаваемых одновременно): \t"))
                gpu_memory_fraction = float(
                    input("Доля видепамяти под нейронку (0-1, где 1 = вся доступная видеопамять): \t"))

                # Достаём данные из БД для подготовки их к подаче в нейронку
                df, total_categories, label_categories = \
                    Getter.get_data_from_db(self, num_tickets)

                # Подготовка входных для нейронной сети
                X_train, y_train, X_test, y_test, num_classes, maxSequenceLength, vocab_size = \
                    LSTM_classifier.prepare_data_for_trainig(self, df)

                # Создание и обучение модели нейронной сети
                model = LSTM_classifier.model_training(
                    # Кол-во заявок и их категории
                    num_tickets, label_categories,
                    X_train, y_train, X_test, y_test, num_classes, maxSequenceLength, vocab_size,
                    batch_size, gpu_memory_fraction)

                del model
                import tensorflow.keras.backend as K
                K.clear_session()  # removing session, it will instance another
                import gc
                gc.collect()

            elif int(mode) == 1: # Классифицировать все заявки по датам --------------------------

                model_loaded = tensorflow.keras.models.load_model('lstm_30000.h5')
                begin_data = input("Введите начальную дату в формате гггг-мм-дд: ")
                end_data = input("Введите конечную дату в формате гггг-мм-дд: ")

                # Экспортируем из БД OTRS в Excel
                num_tickets = Excel.db_to_excel(self, begin_data, end_data)
                print("Количество заявок: ", num_tickets)

                # Импортируем в свою таблицу из Excel
                Excel.excel_to_db(self)

                # Нормализуем текст заявок и очередей
                begin = 0
                end = num_tickets
                DB_formalizer.normalise_body_on_db(self, begin, end)

                for i in range(1,num_tickets):
                    LSTM_classifier.model_predict_without_train(
                        self, model_loaded, label_categories_const, num_tickets)

            elif int(mode) == 2: # Протестировать -----------------------------------------------
                # Загружаем обученную модель
                # num_tickets = input("\nНа скольких записях обучена модель, которую хотите протестить? : ")
                # model_loaded = tensorflow.keras.models.load_model('lstm_' + num_tickets + '.h5')

                print("Категории: ", label_categories_const)
                model_loaded = tensorflow.keras.models.load_model('lstm_30000.h5')
                num_ticket = -1
                while (num_ticket != "exit"):
                    num_ticket = input("\nУкажите номер заявки в БД для классификации:\t")
                    LSTM_classifier.model_predict_without_train(self, model_loaded, label_categories_const, num_ticket)

            elif int(mode) == 3: # Классифицировать все заявки --------------------------
                print("Категории: ", label_categories_const)
                num_tickets = input("\nНа скольких записях обучена используемая модель классификатора? : ")
                # count_for_test = input("\nНа сколько записей из датасета сделать предсказания? : ")
                model_loaded = tensorflow.keras.models.load_model('lstm_' + num_tickets + '.h5')
                LSTM_classifier.model_predict_without_train_auto(self, model_loaded, label_categories_const, num_tickets)

            elif int(mode) == 4:  # Экспорт из БД OTRS в нашу БД ---------------------------------

                export_mode = input("\nВыберите режим экспорта: "
                                    "\n1 - Из БД OTRS в Excel"
                                    "\n2 - Из Excel в нашу БД"
                                    "\nВыбраный режим: \t")
                if int(export_mode) == 1:
                    begin_data = input("\tВведите начальную дату в формате гггг-мм-дд: ")
                    end_data = input("\tВведите конечную дату в формате гггг-мм-дд: ")

                    try:
                        num_tickets = Excel.db_to_excel(self, begin_data, end_data)
                        print("Заявки экспортированы")
                        print("Количество экспортированых заявок: ", num_tickets)
                    except:
                        print("Произошла ошибка. Заявки не экспортированы.")

                elif int(export_mode) == 2:
                    Excel.excel_to_db(self)

            elif int(mode) == 5: # Форматировать заявки в БД -------------------------------------
                begin = int(input("\nC какого ID заявки начать? : "))
                end = int(input("На каком ID заявки остановиться? : "))

                DB_formalizer.normalise_body_on_db(self, begin, end)

            else:
                print("Выберите один из предложенных режимов работы.")
                mode = ""

    @staticmethod
    def prepare_data_for_trainig(self, df):

        # Sampling
        descriptions = df['article_a_body']
        categories = df[u'ticket_queue_name']

        # создаем единый словарь (слово -> число) для преобразования
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(descriptions.tolist())

        # Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
        textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

        X_train, y_train, X_test, y_test = self.load_data_from_arrays(textSequences, categories, train_test_split=0.8)

        # Максимальное количество слов в самом длинном описании заявки
        max_words = 0
        for desc in descriptions.tolist():
            words = len(desc.split())
            if words > max_words:
                max_words = words
        print('Максимальное количество слов в самом длинном описании заявки: {} слов'.format(max_words))

        total_unique_words = len(tokenizer.word_counts)
        print('Всего уникальных слов в словаре: {}'.format(total_unique_words))

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

        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train = encoder.transform(y_train)
        y_test = encoder.transform(y_test)

        # Считаем количество категорий:
        num_classes = np.max(y_train) + 1
        print('Количество категорий для классификации: {}'.format(num_classes))

        print(u'Преобразуем категории в матрицу двоичных чисел '
              u'(для использования categorical_crossentropy)')
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        return X_train, y_train, X_test, y_test, num_classes, maxSequenceLength, vocab_size

    @staticmethod
    def model_training(num_tickets, label_categories,
                       X_train, y_train,
                       X_test, y_test,
                       num_classes, maxSequenceLength, vocab_size,
                       batch_size, gpu_memory_fraction):

        # gpu_options = tf.GPUOptions(allow_growth=True)
        # session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        # Определяем LSTM модель для обучения: =============================================
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Embedding, LSTM
        from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D

        # CUDA_VISIBLE_DEVICES = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        TF_CONFIG_ = tf.ConfigProto()
        TF_CONFIG_.gpu_options.allow_growth = True
        TF_CONFIG_.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction # 0.5
        TF_CONFIG_.gpu_options.visible_device_list = "0"

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        # TF_CONFIG_ = tf.ConfigProto(gpu_options=gpu_options)
        # TF_CONFIG_ = tf.ConfigProto()
        # TF_CONFIG_.gpu_options.allow_growth = True
        # options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        with tf.Session(config=TF_CONFIG_) as sess:
            # максимальное количество слов для анализа
            max_features = vocab_size + 1

            print(u'Собираем модель...')
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

            with tf.device('/GPU:0'):
                sess.run(tf.global_variables_initializer())

                model.compile(loss='binary_crossentropy',
                                    optimizer='adam',
                                    metrics=['accuracy'])

                print(model.summary())

                # Обучаем
                # batch_size = 4
                epochs = 3

                print(u'Тренируем модель...')
                history = model.fit(X_train, y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_data=(X_test, y_test))

                # По окончанию процесса обучения оценим его результаты:
                score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

                print(u'Оценка теста: {}'.format(score[0]))
                print(u'Оценка точности модели: {}'.format(score[1]))

                print('Сохраняем модель')
                model.save("lstm_" + str(num_tickets) + ".h5")
                # model.save_weights('./checkpoints/lstm/2000')
                print("Сохранение завершено!")

                # При желании снова можно построить график эффективности процесса обучения:
                # Посмотрим на эффективность обучения

                # Раздельные графики
                Plotter.plot_lstm(history, num_tickets)
                # Обьединённые графики
                Plotter.plot_train_result(epochs, history, num_tickets)
                # Конфусионная матрица
                Plotter.make_confusion_matrix(model, X_test, y_test, label_categories, num_tickets)

                # Тестирование модели нейронки
                LSTM_classifier.model_predict_with_train(model, X_test, y_test, label_categories)

        return model

    @staticmethod
    def model_predict_with_train(model, X_test, y_test, label_categories):
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
        for i in range(0, 30):
            prediction = model.predict(np.array([X_test[i]]))
            print("prediction ", prediction)
            # print("argmax_predict ", np.argmax(prediction))
            # print("argmax_ytest", np.argmax(y_test, axis=1))
            predicted_label = label_categories[(np.argmax(prediction))]
            print(X_test[i][50], "...")
            print('Правильная категория: {}'.format(label_categories[np.argmax(y_test, axis=1)[i]]))
            print("Определенная моделью категория: {}".format(predicted_label))

    @staticmethod
    def model_predict_without_train(self, model_loaded, label_categories_const, num_ticket):
        ticket_text, ticket = Getter.get_1_ticket_from_db(self, num_ticket)
        prediction = model_loaded.predict(np.array(ticket_text))
        predicted_label = label_categories_const[(np.argmax(prediction))]

        # Покажем архитектуру модели и оценим точность
        # model_loaded.summary()
        # loss, acc = model_loaded.evaluate(ticket_text, labels, batch_size=32, verbose=1)
        # print("Точность восстановленной модели: {:5.2f}%".format(100 * acc))

        print("\tprediction: ", prediction)
        print("\tticket_id:  ", ticket.iloc[0]['ticket_id'])
        print("\tticket text:  ", ticket.iloc[0]['article_a_body'])
        print("\t\tВерная категория: ", ticket.iloc[0]['ticket_queue_name'])
        print("\t\tОпределенная моделью категория: {}".format(predicted_label))

    @staticmethod
    def model_predict_without_train_auto(self, model_loaded, label_categories_const, num_tickets):
        # f = open('correct_ids1.txt', 'w')
        # f1 = open('correct_ids2.txt', 'w')
        # f.close()
        # f1.close()

        begin_num = int(input("Введите номер начальной заявки: "))
        end_num = int(input("Введите номер конечной заявки: "))

        strnums = [ ]
        count = 0
        # for i in range(1, int(num_tickets)):
        for i in range(begin_num, end_num):
            ticket_text, ticket = Getter.get_1_ticket_from_db(self, i)
            if ticket_text.size == 0:
                continue
            prediction = model_loaded.predict(np.array(ticket_text))
            predicted_label = label_categories_const[(np.argmax(prediction))]

            print("\n\tprediction: ", prediction)
            print("\tticket_id:  ", ticket.iloc[0]['ticket_id'])
            print("\tticket text:  ", ticket.iloc[0]['article_a_body'])
            print("\t\tВерная категория: ", ticket.iloc[0]['ticket_queue_name'])
            print("\t\tОпределенная моделью категория: {}".format(predicted_label))

            # Setter.update_ticket_on_db(self, ticket, predicted_label)

        #     # Сохранение в файл номеров заявок, у которых предсказание сошлось с тем, что было в БД
        #     if ticket.iloc[0]['ticket_queue_name'] == predicted_label:
        #         strnums.append(ticket.iloc[0]['ticket_id'])
        #         with open('correct_tickets.txt', 'a') as f:
        #             f.write("\nCount: %s\t" % count)
        #             f.write("\nTicket_ID: %s\t" % ticket.iloc[0]['ticket_id'])
        #             f.write("\tВерная категория: {}".format(ticket.iloc[0]['ticket_queue_name']))
        #             f.write("\tОпределенная моделью категория: {}".format(predicted_label))
        #         count+=1
        #
        # with open('correct_ids.txt', 'a') as f:
        #     f.write("Количество верных: %s" % count)
        #     for item in strnums:
        #         f.write("\n%s" % item)


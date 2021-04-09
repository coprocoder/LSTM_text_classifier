import pymysql.cursors
# from lstm_model_manager import *
from interface.ui_manager import *

from PyQt5.QtCore import Qt, QObject, pyqtSignal

class Comunicator(QObject):

    thread_start = pyqtSignal(str)
    thread_end = pyqtSignal(str)

    window_log_message = pyqtSignal(str)

class Classifier:

    def __init__(self):
        self.connection = self.get_connection()
        self.dbinfo = {
            'host': 'localhost',
            'username': 'root',
            'password': '1',
            'database': 'OTRS',
            'table': 'tickets_normalise',
            'body_field' : 'article_a_body',
            'queue_field' : 'ticket_queue_name'
        }

    @staticmethod
    def get_connection():
        connection = pymysql.connect(host='localhost',
                                     user='root',
                                     password='1',
                                     db='OTRS',
                                     charset='utf8mb4',
                                     cursorclass=pymysql.cursors.DictCursor,
                                     autocommit=True)
        return connection

    @staticmethod
    def load_data_from_arrays(strings, labels, train_test_split=0.9):
        data_size = len(strings)
        test_size = int(data_size - round(data_size * train_test_split))
        print("Test size: {}".format(test_size))

        print("\nTraining set:")
        x_train = strings[test_size:]
        print("\t - x_train: {}".format(len(x_train)))
        y_train = labels[test_size:]
        print("\t - y_train: {}".format(len(y_train)))

        print("\nTesting set:")
        x_test = strings[:test_size]
        print("\t - x_test: {}".format(len(x_test)))
        y_test = labels[:test_size]
        print("\t - y_test: {}".format(len(y_test)))

        return x_train, y_train, x_test, y_test

    def get_logger(self):

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

        self.logger = logger
        return logger

    def data_classify(self):

        # Create obj
        self.comm = Comunicator()
        self.lstm_classifier = LSTM_model_manager(self)
        self.ui = Ui_manager(self)

        # Init base in obj
        self.lstm_classifier.set_connect()
        self.ui.ui_init()








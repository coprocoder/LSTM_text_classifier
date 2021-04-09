import pymysql.cursors

from mlp_classifier import MLP_classifier
from lstm_classifier import LSTM_classifier

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
        self.dbotrs = {
            'username': 'mog.ie',
            'password': 'F0reGn+d'
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

    def data_classify(self):
        # logger —----------------------------------------------------------------------------------------------

        # print("\n-------------------- MLP --------------------\n")
        # MLP_classifier.data_classify(self)

        # print("\n-------------------- LSTM --------------------\n")
        LSTM_classifier.data_classify_lstm(self)

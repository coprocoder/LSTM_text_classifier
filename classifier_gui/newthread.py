import threading
import time
from lstm_model_manager import LSTM_model_manager

class NewThread(threading.Thread):
    def __init__(self, classifier_manager, arg1, arg2):
        threading.Thread.__init__(self)
        self.classifier_manager = classifier_manager
        self.lstm_obj = LSTM_model_manager(classifier_manager)
        self.arg1 = arg1
        self.arg2 = arg2

    # def __del__(self):
    #     #     del self.lstm_obj
    #     #     del self

    def run(self):
        if self.getName() == "classify_one":
            # self.lstm_obj.set_connect()
            self.lstm_obj.model_predict_without_train(self.arg1)
        if self.getName() == "classify_several" or self.getName() == "classify_all":
            # self.lstm_obj.set_connect()
            self.lstm_obj.model_predict_without_train_auto(self.arg1, self.arg2)

        time.sleep(2)

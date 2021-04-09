import sys  # sys нужен для передачи argv в QApplication

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
import threading

from interface.py import mainmenu
from interface.py import train
from interface.py import export
from interface.py import format
from interface.py import classify

from mysql.script.excel_db import Excel
from mysql.script.db_format import DB_formalizer
from mysql.script.getter import Getter

import resource_manager

from lstm_model_manager import *
from newthread import NewThread


class Ui_manager:
    def __init__(self, classifier):
        self.classifier_manager = classifier

    def ui_init(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.app.setWindowIcon(QtGui.QIcon('.\images\main_icon.png'))
        self.window = Ui_mainmenu(self)
        self.window.show()  # Показываем окно
        self.app.exec_()  # и запускаем приложение

    def menu_init(self, parameter_from_ui):

        # menu_cases = {
        #     'train': self.menu_train(),
        #     'classify_all': self.menu_classify_all(),
        #     'classify_one': self.menu_classify_one(),
        #     'classify_several': self.menu_classify_several(),
        #     'export': self.menu_export(),
        #     'import': self.menu_import(),
        #     'format_text': self.menu_format_text()
        # }
        # try:
        #     menu_cases[parameter_from_ui]
        # except KeyError as e:
        #     # присвоить значение по умолчанию вместо бросания исключения
        #     raise ValueError('Undefined unit: {}'.format(e.args[0]))

        if parameter_from_ui == 'train': self.menu_train()
        elif parameter_from_ui == 'classify_all': self.menu_classify_all()
        elif parameter_from_ui == 'classify_one': self.menu_classify_one()
        elif parameter_from_ui == 'classify_several': self.menu_classify_several()
        elif parameter_from_ui == 'export': self.menu_export()
        elif parameter_from_ui == 'import': self.menu_import()
        elif parameter_from_ui == 'format_text': self.menu_format_text()

        return 0

    # Menu ---------------------------------------------------------------------------------
    def menu_train(self):  # Обучить -------------------------------------------------------
        resource_manager.resource_manage(int(self.window.ram_limit))

        gpu_memory_fraction = float(self.window.gpu_mem_limit)
        num_tickets = int(self.window.num_tickets)
        batch_size = int(self.window.batch_size)

        # Достаём данные из БД для подготовки их к подаче в нейронку
        df, total_categories, label_categories = \
            Getter.get_data_from_db(self.classifier_manager, num_tickets)

        # Подготовка входных для нейронной сети
        X_train, y_train, X_test, y_test, num_classes, maxSequenceLength, vocab_size = \
            self.classifier_manager.lstm_classifier.prepare_data_for_trainig(df)

        # Создание и обучение модели нейронной сети
        model = self.classifier_manager.lstm_classifier.model_training(
            num_tickets, label_categories, # Кол-во заявок и их категории
            X_train, y_train, X_test, y_test, # Выборки
            num_classes, maxSequenceLength, vocab_size, # Кол-во категорий, макс. длина заявки, размер словаря
            batch_size, gpu_memory_fraction) # Размер порции и доля видеопамяти

        # После обучения модель сохраняется в файл и RAM/GPU очищается
        del model
        import tensorflow.keras.backend as K
        K.clear_session()  # removing session, it will instance another
        import gc
        gc.collect()

        return 0

    def menu_classify_all(self): # Классифицировать все заявки по датам

        begin_data = int(self.window.beginDate)
        end_data = int(self.window.endDate)

        # Экспортируем из БД OTRS в Excel
        num_tickets = Excel.db_to_excel(self.classifier_manager, begin_data, end_data)
        print("Количество заявок: ", num_tickets)
        self.window.log.append("Количество заявок: " + str(num_tickets))

        # Импортируем в свою таблицу из Excel
        Excel.excel_to_db(self.classifier_manager)

        # Нормализуем текст заявок и очередей
        begin = 0
        end = num_tickets
        DB_formalizer.normalise_body_on_db(self.classifier_manager, begin, end)

        '''
            Тут надо передавать номер заявки последовательно в поток
        '''
        thread = NewThread(self.classifier_manager, 1, num_tickets)
        thread.setName("classify_all")
        thread.start()

        # for i in range(1, num_tickets):
        #     # self.classifier_manager.lstm_classifier.model_predict_without_train(num_tickets)
        #     thread = NewThread(self.classifier_manager, i, i)
        #     thread.setName("classify_all")
        #     thread.start()
        return 0

    def menu_classify_one(self):
        # Загружаем обученную модель
        # num_tickets = input("\nНа скольких записях обучена модель, которую хотите протестить? : ")
        # model_loaded = tensorflow.keras.models.load_model('lstm_' + num_tickets + '.h5')

        num_ticket = int(self.window.countTickets) - 1
        # self.classifier_manager.lstm_classifier.model_predict_without_train(num_ticket)

        thread = NewThread(self.classifier_manager, num_ticket, num_ticket)
        thread.setName("classify_one")
        thread.start()

        return 0

    def menu_classify_several(self):
        print("Категории: ", self.classifier_manager.lstm_classifier.label_categories_const)
        self.window.log.append("Категории: " + str(self.classifier_manager.lstm_classifier.label_categories_const))

        # self.classifier_manager.lstm_classifier.model_predict_without_train_auto()

        thread = NewThread(self.classifier_manager, int(self.window.beginTicketIndex), int(self.window.endTicketIndex))
        thread.setName("classify_several")
        thread.start()

        return 0

    def menu_export(self):
        begin_data = int(self.window.beginDate)
        end_data = int(self.window.endDate)

        try:
            num_tickets = Excel.db_to_excel(self.classifier_manager, begin_data, end_data)

            print("Количество экспортированых заявок: ", num_tickets)
            print("Заявки экспортированы")

            self.window.log.append("Заявки экспортированы")
            self.window.log.append("Количество экспортированых заявок: " + str(num_tickets))
            return 0
        except:
            print("Произошла ошибка. Заявки не экспортированы.")
            self.window.log.append("Произошла ошибка. Заявки не экспортированы.")
            return -1

    def menu_import(self):
        Excel.excel_to_db(self.classifier_manager)
        return 0

    def menu_format_text(self):
        beginTicketIndex = int(self.window.beginTicketIndex)
        endTicketIndex = int(self.window.endTicketIndex)

        DB_formalizer.normalise_body_on_db(self.classifier_manager, beginTicketIndex, endTicketIndex)
        return 0

    @QtCore.pyqtSlot(str)
    def signal_catcher(self, msg):
        print("Signal catched, hueh... " + msg)

    def log_writer(self, msg):
        self.window.log.append(msg)

class Ui_mainmenu(QtWidgets.QMainWindow, mainmenu.Ui_MainWindow):
    def __init__(self, ui_manager):

        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.setWindowTitle('Главное меню')
        # self.setWindowIcon(QIcon('main_icon.png'))
        self.ui_manager = ui_manager

        # Connect signal-slot
        self.trainButton.clicked.connect(self.show_train_window)
        self.classifyAllButton.clicked.connect(self.show_classify_all_window)
        self.classifyOneButton.clicked.connect(self.show_classify_one_window)
        self.classifySeveralButton.clicked.connect(self.show_classify_several_window)
        self.exportTicketsButton.clicked.connect(self.show_export_window)
        self.importTicketsButton.clicked.connect(self.show_import_window)
        self.formatTextButton.clicked.connect(self.show_format_window)

    def show_train_window(self):
        self.ui_manager.window = Ui_train_window(self.ui_manager)
        self.ui_manager.window.show()
        # self.ui_manager.window.closeButton.setEnabled(0)
        pass

    def show_classify_one_window(self):
        self.ui_manager.window = Ui_classify_window(self.ui_manager)
        self.ui_manager.window.mode = 'classify_one'
        self.ui_manager.window.set_text_on_ui()
        # self.ui_manager.window.closeButton.setEnabled(0)
        self.ui_manager.window.show()
        pass

    def show_classify_all_window(self):
        self.ui_manager.window = Ui_export_window(self.ui_manager)
        self.ui_manager.window.mode = 'classify_all'
        self.ui_manager.window.set_text_on_ui()
        # self.ui_manager.window.closeButton.setEnabled(0)
        self.ui_manager.window.show()
        pass

    def show_export_window(self):
        self.ui_manager.window = Ui_export_window(self.ui_manager)
        self.ui_manager.window.mode = 'export'
        self.ui_manager.window.set_text_on_ui()
        # self.ui_manager.window.closeButton.setEnabled(0)
        self.ui_manager.window.show()
        pass

    def show_import_window(self):
        self.ui_manager.window = Ui_export_window(self.ui_manager)
        self.ui_manager.window.mode = 'import'
        self.ui_manager.window.set_text_on_ui()
        # self.ui_manager.window.closeButton.setEnabled(0)
        self.ui_manager.window.show()
        pass

    def show_classify_several_window(self):
        self.ui_manager.window = Ui_format_window(self.ui_manager)
        self.ui_manager.window.mode = 'classify_several'
        self.ui_manager.window.set_text_on_ui()
        # self.ui_manager.window.closeButton.setEnabled(0)
        self.ui_manager.window.show()
        pass

    def show_format_window(self):
        self.ui_manager.window = Ui_format_window(self.ui_manager)
        self.ui_manager.window.mode = 'format_text'
        self.ui_manager.window.set_text_on_ui()
        # self.ui_manager.window.closeButton.setEnabled(0)
        self.ui_manager.window.show()
        pass

class Ui_train_window(QtWidgets.QMainWindow, train.Ui_MainWindow):
    def __init__(self, ui_manager):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Обучение модели')
        self.ui_manager = ui_manager

        # Connect signal-slot
        self.trainButton.clicked.connect(self.get_settings)
        self.closeButton.clicked.connect(self.close_window)

    def get_settings(self):
        # self.ui_manager.window.closeButton.setEnabled(0)
        self.ram_limit = self.ram_size_line.text()
        self.gpu_mem_limit = self.gpu_mem_size_line.text()
        self.num_tickets = self.num_tickets_line.text()
        self.batch_size = self.batch_size_line.text()
        self.ui_manager.menu_init('train')
        pass

    def close_window(self):
        self.ui_manager.window = Ui_mainmenu(self.ui_manager)
        self.ui_manager.window.show()
        self.close()
        pass

class Ui_classify_window(QtWidgets.QMainWindow, classify.Ui_MainWindow):
    def __init__(self, ui_manager):
        super().__init__()
        self.setupUi(self)
        self.ui_manager = ui_manager
        self.mode = ''

        # Connect signal-slot
        self.applyButton.clicked.connect(self.get_settings)
        self.closeButton.clicked.connect(self.close_window)

    def set_text_on_ui(self):
        if self.mode == 'classify_one':
            self.setWindowTitle(u'Классификация одной заявки')
            self.label.setText(u'Классифицировать заявку №')

    def get_settings(self):
        # self.ui_manager.window.closeButton.setEnabled(0)
        self.countTickets = self.lineEdit.text()
        self.ui_manager.menu_init(self.mode)

    def close_window(self):
        self.ui_manager.window = Ui_mainmenu(self.ui_manager)
        self.ui_manager.window.show()
        self.close()
        pass

class Ui_export_window(QtWidgets.QMainWindow, export.Ui_MainWindow):
    def __init__(self, ui_manager):
        super().__init__()
        self.setupUi(self)
        self.ui_manager = ui_manager
        self.mode = ''

        # Connect signal-slot
        self.startButton.clicked.connect(self.get_settings)
        self.closeButton.clicked.connect(self.close_window)

    def set_text_on_ui(self):
        if self.mode == 'classify_all':
            self.setWindowTitle(u'Классификация всех заяок')
            # self.label_2.setText(u'заявку')
        elif self.mode == 'export':
            self.setWindowTitle(u'Экспорт из исходной таблицы БД в Excel')
            # self.label_2.setText(u'заявок')
        elif self.mode == 'import':
            self.setWindowTitle(u'Импорт из Excel в новую таблицу БД')
            self.label.hide()
            self.label_2.hide()
            self.beginDate.hide()
            self.endDate.hide()

    def get_settings(self):
        # self.closeButton.setEnabled(0)
        self.beginDate = self.beginDate.text()
        self.endDate = self.endDate.text()
        self.ui_manager.menu_init(self.mode)

    def close_window(self):
        self.ui_manager.window = Ui_mainmenu(self.ui_manager)
        self.ui_manager.window.show()
        self.close()
        pass

class Ui_format_window(QtWidgets.QMainWindow, format.Ui_MainWindow):
    def __init__(self, ui_manager):
        super().__init__()
        self.setupUi(self)
        self.ui_manager = ui_manager
        self.mode = ''

        # Connect signal-slot
        self.startButton.clicked.connect(self.get_settings)
        self.closeButton.clicked.connect(self.close_window)

    def set_text_on_ui(self):
        if self.mode == 'format_text':
            self.setWindowTitle(u'Форматирование текста заявок')
            self.startButton.setText('Запустить форматирование')
        elif self.mode == 'classify_several':
            self.setWindowTitle(u'Классификация нескольких заявок')
            self.startButton.setText('Запустить классификацию')

    def get_settings(self):
        # self.ui_manager.window.closeButton.setEnabled(0)
        self.beginTicketIndex = self.beginTicketIndex.text()
        self.endTicketIndex = self.endTicketIndex.text()
        self.ui_manager.menu_init(self.mode)

    def close_window(self):
        self.ui_manager.window = Ui_mainmenu(self.ui_manager)
        self.ui_manager.window.show()
        self.close()
        pass

# def main():
#     app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
#     window = Ui_mainmenu()  # Создаём объект класса ExampleApp
#     window.show()  # Показываем окно
#     app.exec_()  # и запускаем приложение
#
# if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
#     main()  # то запускаем функцию main()


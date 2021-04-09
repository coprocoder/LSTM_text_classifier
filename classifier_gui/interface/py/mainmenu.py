# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\ILYA\Desktop\Python\OTRS\classifier\interface\ui\mainmenu.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(460, 300)
        MainWindow.setMaximumSize(QtCore.QSize(460, 300))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(180, 10, 141, 20))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 30, 61, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 80, 121, 16))
        self.label_3.setObjectName("label_3")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(10, 130, 81, 16))
        self.label_6.setObjectName("label_6")
        self.trainButton = QtWidgets.QPushButton(self.centralwidget)
        self.trainButton.setGeometry(QtCore.QRect(10, 50, 441, 23))
        self.trainButton.setObjectName("trainButton")
        self.classifyAllButton = QtWidgets.QPushButton(self.centralwidget)
        self.classifyAllButton.setGeometry(QtCore.QRect(10, 100, 441, 23))
        self.classifyAllButton.setObjectName("classifyAllButton")
        self.classifyOneButton = QtWidgets.QPushButton(self.centralwidget)
        self.classifyOneButton.setGeometry(QtCore.QRect(10, 150, 441, 23))
        self.classifyOneButton.setObjectName("classifyOneButton")
        self.classifySeveralButton = QtWidgets.QPushButton(self.centralwidget)
        self.classifySeveralButton.setGeometry(QtCore.QRect(10, 180, 441, 23))
        self.classifySeveralButton.setObjectName("classifySeveralButton")
        self.exportTicketsButton = QtWidgets.QPushButton(self.centralwidget)
        self.exportTicketsButton.setGeometry(QtCore.QRect(10, 210, 441, 23))
        self.exportTicketsButton.setObjectName("exportTicketsButton")
        self.formatTextButton = QtWidgets.QPushButton(self.centralwidget)
        self.formatTextButton.setGeometry(QtCore.QRect(10, 270, 441, 23))
        self.formatTextButton.setObjectName("formatTextButton")
        self.importTicketsButton = QtWidgets.QPushButton(self.centralwidget)
        self.importTicketsButton.setGeometry(QtCore.QRect(10, 240, 441, 23))
        self.importTicketsButton.setObjectName("importTicketsButton")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Выберите режим работы"))
        self.label_2.setText(_translate("MainWindow", "Обучение"))
        self.label_3.setText(_translate("MainWindow", "Автоматический режим"))
        self.label_6.setText(_translate("MainWindow", "Ручной режим"))
        self.trainButton.setText(_translate("MainWindow", " Обучить модель, вывести графики эффективности, потерь и матрицы потерь"))
        self.classifyAllButton.setText(_translate("MainWindow", "Классифицировать все заявки за указанные даты"))
        self.classifyOneButton.setText(_translate("MainWindow", "Классифицировать одну заявку"))
        self.classifySeveralButton.setText(_translate("MainWindow", " Классифицировать несколько заявок"))
        self.exportTicketsButton.setText(_translate("MainWindow", "Экспортировать заявки из исходной таблицы БД в документ Excel"))
        self.formatTextButton.setText(_translate("MainWindow", "Форматировать текст заявок"))
        self.importTicketsButton.setText(_translate("MainWindow", "Импортировать заявки из документа Excel в новую таблицу БД"))


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\ILYA\Desktop\Python\OTRS\classifier\interface\ui\format.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(480, 223)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.beginTicketIndex = QtWidgets.QLineEdit(self.centralwidget)
        self.beginTicketIndex.setObjectName("beginTicketIndex")
        self.horizontalLayout_2.addWidget(self.beginTicketIndex)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.endTicketIndex = QtWidgets.QLineEdit(self.centralwidget)
        self.endTicketIndex.setObjectName("endTicketIndex")
        self.horizontalLayout_3.addWidget(self.endTicketIndex)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 1, 1, 1)
        self.log = QtWidgets.QTextEdit(self.centralwidget)
        self.log.setObjectName("log")
        self.gridLayout.addWidget(self.log, 1, 0, 1, 2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setObjectName("startButton")
        self.horizontalLayout.addWidget(self.startButton)
        self.closeButton = QtWidgets.QPushButton(self.centralwidget)
        self.closeButton.setObjectName("closeButton")
        self.horizontalLayout.addWidget(self.closeButton)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "Начальный индекс в БД"))
        self.label_3.setText(_translate("MainWindow", "Конечный индекс в БД"))
        self.startButton.setText(_translate("MainWindow", "Запустить форматирование"))
        self.closeButton.setText(_translate("MainWindow", "Вернуться в главное меню"))


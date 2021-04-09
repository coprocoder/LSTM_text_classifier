# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\ILYA\Desktop\Python\OTRS\classifier\interface\ui\classify.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(360, 250)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.closeButton = QtWidgets.QPushButton(self.centralwidget)
        self.closeButton.setObjectName("closeButton")
        self.gridLayout.addWidget(self.closeButton, 2, 0, 1, 3)
        self.log = QtWidgets.QTextEdit(self.centralwidget)
        self.log.setObjectName("log")
        self.gridLayout.addWidget(self.log, 1, 0, 1, 3)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.applyButton = QtWidgets.QPushButton(self.centralwidget)
        self.applyButton.setObjectName("applyButton")
        self.gridLayout.addWidget(self.applyButton, 0, 2, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.closeButton.setText(_translate("MainWindow", "Вернуться в главное меню"))
        self.label.setText(_translate("MainWindow", "Классифицировать заявку №"))
        self.applyButton.setText(_translate("MainWindow", "Начать"))


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\ILYA\Desktop\Python\OTRS\classifier\interface\ui\export.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 204)
        MainWindow.setMinimumSize(QtCore.QSize(400, 190))
        MainWindow.setMaximumSize(QtCore.QSize(9999, 9999))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.beginDate = QtWidgets.QDateEdit(self.centralwidget)
        self.beginDate.setObjectName("beginDate")
        self.horizontalLayout_2.addWidget(self.beginDate)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.endDate = QtWidgets.QDateEdit(self.centralwidget)
        self.endDate.setObjectName("endDate")
        self.horizontalLayout_3.addWidget(self.endDate)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 1, 1, 1)
        self.log = QtWidgets.QTextEdit(self.centralwidget)
        self.log.setObjectName("log")
        self.gridLayout.addWidget(self.log, 1, 0, 1, 2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setObjectName("startButton")
        self.horizontalLayout_4.addWidget(self.startButton)
        self.closeButton = QtWidgets.QPushButton(self.centralwidget)
        self.closeButton.setObjectName("closeButton")
        self.horizontalLayout_4.addWidget(self.closeButton)
        self.gridLayout.addLayout(self.horizontalLayout_4, 2, 0, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Начальная дата"))
        self.label_2.setText(_translate("MainWindow", "Конечная дата"))
        self.startButton.setText(_translate("MainWindow", "Начать экспорт"))
        self.closeButton.setText(_translate("MainWindow", "Вернуться в главное меню"))


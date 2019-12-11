# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DetectLandmarksPopup.ui',
# licensing of 'DetectLandmarksPopup.ui' applies.
#
# Created: Sat Dec  7 18:07:09 2019
#      by: PyQt5-uic  running on PyQt5 5.13.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(389, 191)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 10, 341, 171))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.all_frames_radio = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.all_frames_radio.setObjectName("all_frames_radio")
        self.horizontalLayout_6.addWidget(self.all_frames_radio)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.some_frames_radio = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.some_frames_radio.setObjectName("some_frames_radio")
        self.horizontalLayout_7.addWidget(self.some_frames_radio)
        self.some_frames_input = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.some_frames_input.setObjectName("some_frames_input")
        self.horizontalLayout_7.addWidget(self.some_frames_input)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.go_button = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.go_button.setObjectName("go_button")
        self.horizontalLayout_3.addWidget(self.go_button)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem4)
        self.time_estimate = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.time_estimate.setObjectName("time_estimate")
        self.verticalLayout_3.addWidget(self.time_estimate)
        self.progress = QtWidgets.QProgressBar(self.horizontalLayoutWidget)
        self.progress.setMaximum(1000)
        self.progress.setProperty("value", 0)
        self.progress.setObjectName("progress")
        self.verticalLayout_3.addWidget(self.progress)
        self.horizontalLayout_4.addLayout(self.verticalLayout_3)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem5)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Form", None, -1))
        self.all_frames_radio.setText(QtWidgets.QApplication.translate("Form", "All Frames", None, -1))
        self.some_frames_radio.setText(QtWidgets.QApplication.translate("Form", "Some Frames", None, -1))
        self.go_button.setText(QtWidgets.QApplication.translate("Form", "Go", None, -1))
        self.time_estimate.setText(QtWidgets.QApplication.translate("Form", "Estimated Time Left:", None, -1))


# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MetricViewerPopup.ui',
# licensing of 'MetricViewerPopup.ui' applies.
#
# Created: Sat Dec  7 18:07:33 2019
#      by: PyQt5-uic  running on PyQt5 5.13.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(673, 854)
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 611, 801))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.main_container = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.main_container.setContentsMargins(0, 0, 0, 0)
        self.main_container.setObjectName("main_container")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.normalize_checkbox = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.normalize_checkbox.setObjectName("normalize_checkbox")
        self.horizontalLayout.addWidget(self.normalize_checkbox)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.main_container.addLayout(self.horizontalLayout)
        self.scrollArea = QtWidgets.QScrollArea(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setMaximumSize(QtCore.QSize(16777215, 120))
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 607, 118))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.metric_container = QtWidgets.QVBoxLayout()
        self.metric_container.setObjectName("metric_container")
        self.verticalLayout_2.addLayout(self.metric_container)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.main_container.addWidget(self.scrollArea)
        self.line = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.main_container.addWidget(self.line)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Form", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("Form", "Select Metrics:", None, -1))
        self.normalize_checkbox.setText(QtWidgets.QApplication.translate("Form", "Normalize", None, -1))


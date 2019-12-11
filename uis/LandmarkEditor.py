# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LandmarkEditor.ui',
# licensing of 'LandmarkEditor.ui' applies.
#
# Created: Sat Dec  7 18:06:46 2019
#      by: PyQt5-uic  running on PyQt5 5.13.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(265, 112)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 10, 241, 89))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem = QtWidgets.QSpacerItem(5, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_3.addItem(spacerItem)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.group_label = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.group_label.setEnabled(True)
        self.group_label.setMargin(5)
        self.group_label.setObjectName("group_label")
        self.horizontalLayout.addWidget(self.group_label)
        self.group_dropdown = QtWidgets.QComboBox(self.horizontalLayoutWidget_3)
        self.group_dropdown.setObjectName("group_dropdown")
        self.horizontalLayout.addWidget(self.group_dropdown)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.line = QtWidgets.QFrame(self.horizontalLayoutWidget_3)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.position_label = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.position_label.setMargin(5)
        self.position_label.setObjectName("position_label")
        self.horizontalLayout_2.addWidget(self.position_label)
        self.position = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.position.setMargin(5)
        self.position.setObjectName("position")
        self.horizontalLayout_2.addWidget(self.position)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        spacerItem1 = QtWidgets.QSpacerItem(5, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_3.addItem(spacerItem1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Form", None, -1))
        self.group_label.setText(QtWidgets.QApplication.translate("Form", "Group:", None, -1))
        self.position_label.setText(QtWidgets.QApplication.translate("Form", "Position:", None, -1))
        self.position.setText(QtWidgets.QApplication.translate("Form", "Placeholder Position", None, -1))


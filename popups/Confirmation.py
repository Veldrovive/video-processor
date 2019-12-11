from PyQt5 import QtWidgets, QtCore
from typing import Tuple, List

class Confirmation(QtWidgets.QMessageBox):
    conf_args: Tuple
    deny_args: Tuple

    def __init__(self, title: str, message: str, on_conf=None, on_deny=None, conf_args=(), deny_args=()):
        super(Confirmation, self).__init__()
        if isinstance(conf_args, (tuple, list)):
            self.conf_args = conf_args
        else:
            self.conf_args = tuple([conf_args])
        if isinstance(deny_args, (tuple, list)):
            self.deny_args = deny_args
        else:
            self.deny_args = tuple([deny_args])

        if on_conf is not None:
            self.on_conf = on_conf
        if on_deny is not None:
            self.on_deny = on_deny
        self.setText(title)
        self.setInformativeText(message)
        self.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        self.setDefaultButton(QtWidgets.QMessageBox.Ok)
        self.buttonClicked.connect(self.on_clicked)
        self.show()

    def on_conf(self):
        print("User Confirmed")

    def on_deny(self):
        print("User Denied")

    @QtCore.pyqtSlot(QtWidgets.QAbstractButton)
    def on_clicked(self, button: QtWidgets.QAbstractButton):
        if button.text() == "Cancel":
            self.on_deny(*self.deny_args)
        else:
            self.on_conf(*self.conf_args)

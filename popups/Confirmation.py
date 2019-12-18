from PyQt5 import QtWidgets, QtCore
from typing import Tuple

"""
This is used to prompt the user with simple 'confirm' or 'deny' and handles
a callback for simplicity
"""

class Confirmation(QtWidgets.QMessageBox):
    conf_args: Tuple
    deny_args: Tuple
    parent: QtWidgets.QWidget

    def __init__(self, title: str, message: str, parent = None, on_conf=None, on_deny=None, conf_args=(), deny_args=(), can_deny=True):
        super(Confirmation, self).__init__()
        self.parent = parent
        # If the user did not give a tuple as callback arguments, assume it is
        # one argument
        if isinstance(conf_args, tuple):
            self.conf_args = conf_args
        else:
            self.conf_args = tuple([conf_args])
        if isinstance(deny_args, tuple):
            self.deny_args = deny_args
        else:
            self.deny_args = tuple([deny_args])

        # Since only one callback is needed, we do not use signals
        if on_conf is not None:
            self.on_conf = on_conf
        if on_deny is not None:
            self.on_deny = on_deny
        self.setText(title)
        self.setInformativeText(message)

        # If only information is needed, done display a cancel option
        if can_deny:
            self.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        else:
            self.setStandardButtons(QtWidgets.QMessageBox.Ok)
            self.setDefaultButton(QtWidgets.QMessageBox.Ok)
        self.buttonClicked.connect(self.on_clicked)
        self.show()

    def on_conf(self):
        """
        Placeholder for the user defined callback
        """
        pass

    def on_deny(self):
        """
        Placeholder for the user defined callback
        """
        pass

    @QtCore.pyqtSlot(QtWidgets.QAbstractButton)
    def on_clicked(self, button: QtWidgets.QAbstractButton):
        # Checks which button the user pressed
        if button.text() == "Cancel":
            self.on_deny(*self.deny_args)
        else:
            self.on_conf(*self.conf_args)
        if self.parent is not None:
            self.parent.raise_()
            self.parent.activateWindow()

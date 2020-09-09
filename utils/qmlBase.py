import os
import sys

from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5 import QtQuick, QtCore
from PyQt5.QtQml import qmlRegisterType, QQmlComponent, QQmlEngine, QQmlContext

from utils.Globals import Globals

try:
   wd = sys._MEIPASS
except AttributeError:
   wd = os.getcwd()

class WindowHandler(QtCore.QObject):
    """
    Base class for any object that has direct interaction with a QML object
    """

    _glo: Globals
    _engine: QQmlEngine = None
    _context: QQmlContext = None
    _window: QtQuick.QQuickView = None
    _component: QQmlComponent = None
    message_dialog: QtQuick.QQuickItem = None

    curr_callback = None

    def __init__(self, engine: QQmlApplicationEngine, qml_source: str, title: str = None):
        super().__init__()
        self._glo = Globals.get()
        self._engine = engine
        self._context = QQmlContext(engine)
        self._context.setContextProperty("handler", self)
        self.setup_contexts()
        self._component = QQmlComponent(engine)
        qml_source = os.path.join(wd, 'uis', qml_source)
        qml_relpath = os.path.relpath(qml_source, '.')
        print("QML Source:", qml_relpath, qml_source, QtCore.QUrl(qml_source).path())
        self._component.loadUrl(QtCore.QUrl(qml_relpath))
        # self._window = self._component.findChild(QtQuick.QQuickView, "window")
        self._window = self._component.create(self._context)
        if title is not None:
            self.set_title(title)
        self.message_dialog = self._window.findChild(QtCore.QObject, "messageDialog")
        if self.message_dialog is None:
            raise RuntimeError("Window QMLs must have a messageDialog object")
        else:
            self.message_dialog.accepted.connect(self.message_callback)

    @QtCore.pyqtSlot(name="messageCallback")
    def message_callback(self):
        if self.curr_callback is not None:
            self.curr_callback()

    def send_message(self, text: str, callback=None):
        """
        Deliver a message to the user
        :param text: The text to display
        :param callback: A function to call
        """
        if self.message_dialog is not None:
            self.message_dialog.close()
            self.message_dialog.setProperty("text", text)
            self.curr_callback = callback
            self.message_dialog.open()

    def add_context(self, context_name: str, context):
        """
        Adds a new context to the QML
        :param context_name: The name of the context for QML
        :param context: The context object
        """
        self._context.setContextProperty(context_name, context)

    def setup_contexts(self):
        """
        This should be overridden by the user to add their own contexts
        """
        # Example: self.add_context("my_context", my_context_object)
        pass

    @QtCore.pyqtSlot(bool, name="set_vis")
    def set_vis(self, state: bool):
        """
        Sets the visibility of the window
        :param state: True is shown and False is hidden
        """
        if state:
            self._window.show()
        else:
            self._window.hide()

    def hide(self):
        """Wrapper for set_vis"""
        self.set_vis(False)

    def show(self):
        """Wrapper for set_vis"""
        self.set_vis(True)

    @QtCore.pyqtSlot(str, name="set_title")
    def set_title(self, title: str):
        """
        Sets the window title
        :param title: The new title
        """
        self._window.setTitle(title)

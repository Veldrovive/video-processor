import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import vidViewer
import pandas as pd
import cv2


class Window(QtWidgets.QMainWindow):
	view = None
	scene = None

	def __init__(self):
		super(Window, self).__init__()
		self.setGeometry(50, 50, 800, 600)
		self.setWindowTitle("vidProc Test")
		self.home()

	def home(self):
		self.main_Widget = QtWidgets.QWidget(self)
		self.setCentralWidget(self.main_Widget)

		test_arr = pd.DataFrame({})
		vid_cap = cv2.VideoCapture("./movement_analysis/HadlockT.mp4")

		viewer = vidViewer.ImageViewer(vid_cap, test_arr, (1000, 800))

		viewer.frame_change_signal.connect(self.on_frame_change)
		viewer.playback_change_signal.connect(self.on_playback_change)

		# viewer.play()

		layout = QtWidgets.QVBoxLayout()
		layout.addWidget(viewer)
		self.main_Widget.setLayout(layout)

		self.resize(self.sizeHint())
		self.show()
		viewer.fitInView()

	def quit(self):
		choice = QtWidgets.QMessageBox.question(self, "Leave?", "Get out?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
		print(QtWidgets.QMessageBox.Yes)
		if choice == QtWidgets.QMessageBox.Yes:
			print("Ok")
			sys.exit()

	@QtCore.pyqtSlot(int)
	def on_frame_change(self, frame):
		print("Changed to frame:", frame)

	@QtCore.pyqtSlot(bool)
	def on_playback_change(self, playback_status):
		print("Playback status changed to:", playback_status)


def run():
	app = QtWidgets.QApplication(sys.argv)
	window = Window()
	sys.exit(app.exec_())


run()

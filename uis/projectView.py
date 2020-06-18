from typing import List, Union
import cv2
import os

from utils.qmlBase import WindowHandler
import utils.DataHolders as DataHolders
from utils.Globals import Globals
from PyQt5 import QtQuick, QtCore
import pandas as pd

class FileListModel(QtCore.QAbstractListModel):
    FileNameRole = QtCore.Qt.UserRole + 1
    FrameCountRole = QtCore.Qt.UserRole + 2
    NeedsLandmarksRole = QtCore.Qt.UserRole + 3

    files: List[dict]

    # fileAddedSignal = QtCore.pyqtSignal(str, arguments=["filePath"])
    # fileAddedSignal = QtCore.pyqtSignal(str, str, arguments=["filePath", "landmarkPath"])
    fileAddedSignal = QtCore.pyqtSignal([str], [str, str])
    fileRemovedSignal = QtCore.pyqtSignal(str, arguments=["filePath"])
    landmarkAddedSignal = QtCore.pyqtSignal(str, str, arguments=["videoPath, landmarkPath"])

    def __init__(self, parent=None):
        super().__init__(parent)
        self.files = list()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        row = index.row()
        if role == FileListModel.FileNameRole:
            return self.files[row]["name"]
        if role == FileListModel.FrameCountRole:
            return self.files[row]["frames"]
        if role == FileListModel.NeedsLandmarksRole:
            return self.files[row]["needs_landmarks"]

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.files)

    def roleNames(self):
        return {
            FileListModel.FileNameRole: b'fileName',
            FileListModel.FrameCountRole: b'frames',
            FileListModel.NeedsLandmarksRole: b'needsLandmarks'
        }

    @QtCore.pyqtSlot(str, name="addFile")
    def add_file(self, path: str):
        """
        Adds a new file to the project and finds the metadata
        :param path: The path to the new video
        """
        if path in [file["path"] for file in self.files]:
            return
        try:
            cap = cv2.VideoCapture(path)
        except:
            return
        if not cap or not cap.isOpened():
            return
        self.beginInsertRows(QtCore.QModelIndex(), self.rowCount(), self.rowCount())
        file_name = os.path.basename(path)
        vid_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        has_landmarks, landmarks_file = self.find_landmarks(path)
        self.files.append({"path": path, "name": file_name, "frames": vid_len, "landmark_file": landmarks_file, "needs_landmarks": not has_landmarks})
        cap.release()
        if has_landmarks:
            self.fileAddedSignal[str, str].emit(path, landmarks_file)
        else:
            self.fileAddedSignal[str].emit(path)
        self.endInsertRows()

    def find_landmarks(self, vid_path: str):
        """
        Tries to find the associated landmark file for a video
        :param vid_path: The path to the video file
        """
        csv_path = os.path.splitext(vid_path)[0] + ".csv"
        is_landmarks = os.path.isfile(csv_path)
        return is_landmarks, csv_path if is_landmarks else None

    @QtCore.pyqtSlot(int, str, name="setLandmarks")
    def set_landmarks(self, index, l_path):
        """
        Set the landmarks for the given file
        :param index: The index of the file
        :param l_path: The path to the landmarks
        """
        ix = self.index(index, 0)
        if os.path.splitext(l_path)[1] != ".csv":
            return
        if os.path.isfile(l_path):
            self.files[index]["landmark_file"] = l_path
            self.files[index]["needs_landmarks"] = False
            vid_path = self.files[index]["path"]
            self.landmarkAddedSignal.emit(vid_path, l_path)
        self.dataChanged.emit(ix, ix, self.roleNames())

    @QtCore.pyqtSlot(int, name="removeFile")
    def remove_file(self, row):
        """
        Removes the row from the list
        :param row: The index of the row to remove
        """
        self.beginRemoveColumns(QtCore.QModelIndex(), row, row)
        file_path = self.files[row]["path"]
        del self.files[row]
        self.endRemoveRows()
        self.fileRemovedSignal.emit(file_path)

    def clear(self):
        """Removes all files from the list"""
        while len(self.files) > 0:
            self.beginRemoveColumns(QtCore.QModelIndex(), 0, 0)
            del self.files[0]
            self.endRemoveRows()

class ProjectHandler(WindowHandler):
    _glo: Globals

    project: DataHolders.Project = None

    file_list_model: FileListModel

    projectNameChanged = QtCore.pyqtSignal(str, arguments=['projectName'])
    projectOpened = QtCore.pyqtSignal(str, arguments=["projectName"])

    fanAdded = QtCore.pyqtSignal()
    def has_fan(self):
        if self.project is None:
            return False
        return self.project.get_FAN_path() is not None
    hasFan = QtCore.pyqtProperty(bool, has_fan, notify=fanAdded)

    s3fdAdded = QtCore.pyqtSignal()
    def has_s3fd(self):
        if self.project is None:
            return False
        return self.project.get_s3fd_path() is not None
    hasS3fd = QtCore.pyqtProperty(bool, has_s3fd, notify=s3fdAdded)

    def __init__(self, engine: QtQuick.QQuickView):
        self.file_list_model = FileListModel()
        super().__init__(engine, "uis/projectView.qml", "Create Project")
        self.file_list_model.fileAddedSignal[str, str].connect(self.on_file_added)
        self.file_list_model.fileAddedSignal[str].connect(self.on_file_added)
        self.file_list_model.fileRemovedSignal.connect(self.on_file_removed)
        self.file_list_model.landmarkAddedSignal.connect(self.on_landmark_file_added)
        self._glo.onProjectChange.connect(self.init_project)
        self._save_loc_dialog = self._window.findChild(QtCore.QObject, "saveLocationDialog")
        # self.init_project()

    def init_project(self, project = None):
        """Updates the view for the new project"""
        if project is None:
            project = self._glo.project
        if project is None:
            project = DataHolders.Project("Placeholder")
        self.set_title(f"Editing Project: {project.name}")
        self.project = project
        self.setup_project()
        self.fanAdded.emit()
        self.s3fdAdded.emit()

    def show(self):
        """Overrides the show method to check if a save location is found"""
        super().show()
        if self._glo.project is not None and self.project is None:
            self.init_project()
        if self.project.save_loc is None:
            self._save_loc_dialog.open()

    def setup_contexts(self):
        """Overridden: Adds custom contexts"""
        self.add_context("fileListModel", self.file_list_model)

    def setup_project(self):
        """
        Uses the current project to fill in project details
        """
        self.file_list_model.clear()
        for file in self.project.files:
            self.file_list_model.add_file(file)
            if self.project.files_map[file] is not None:
                self.file_list_model.set_landmarks(len(self.file_list_model.files)-1, self.project.files_map[file])
        self.projectOpened.emit(self.project.name)

    @QtCore.pyqtSlot(name="hasSaveLoc", result=bool)
    def has_save_location(self):
        """
        :return: Gets whether the project has a save location
        """
        return self.project.save_loc is not None

    @QtCore.pyqtSlot(bool, name="setCopyFiles")
    def set_copy_files(self, val: bool):
        """
        Sets whether files should be copied into the project data directory
        :param val: True if files should be copied, false otherwise
        """
        self.project.copy_files = val

    @QtCore.pyqtSlot(str, name="onNameChange")
    def on_name_change(self, name: str):
        """
        Handles actions taken when the name input is changed
        :param name: The new name
        """
        self.project.name = name
        self.projectNameChanged.emit(name)

    def validate(self, save_dir: str) -> Union[None, str]:
        """
        Tests whether all required information is filled and the path is good
        :return: None if everything is in order, otherwise the error
        """
        if len(self.project.name) < 1:
            return "Project must have a name"
        if len(self.file_list_model.files) < 1:
            return "Project must have files in it"
        if self.project.get_FAN_path() is None:
            return "Project must have a FAN model"
        if self.project.get_s3fd_path() is None:
            return "Project must have a s3fd facial alignment model"
        if save_dir is None:
            return "Project must have a save location"
        if not os.path.isdir(save_dir):
            return "Save location must be a folder"
        failed_files = [path for path in os.listdir(save_dir) if path not in ["config", "data", "retraining_data", "models", "metric_output", "meta_data.csv", "video_data.csv"] and path[0] != "."]
        if len(failed_files) > 0:
            return "Save location must be empty: " + str(failed_files)
        if os.path.isfile(os.path.join(save_dir, "meta_data.csv")):
            meta_df = pd.read_csv(os.path.join(save_dir, "meta_data.csv"))
            curr_id = self.project.id
            old_id = meta_df['p_id'][0]
            if curr_id != old_id:
                return "Saved project id does not match current project id. \nThis means you probably are trying to save over an old project. \nDelete it first if this is the case."

    @QtCore.pyqtSlot(str, name="onFileAdded")
    @QtCore.pyqtSlot(str, str, name="onFileAdded")
    def on_file_added(self, file_path: str, landmark_path: str = None):
        """
        Adds a file to the project
        :param file_path: Video path
        :param landmark_path: Landmark Path
        """
        file_path = os.path.abspath(file_path)
        # We only copy files to the project if the landmarks are not found.
        # If we do not do this, the video path will have changed before we try
        # to add landmarks using it.
        self.project.add_file(file_path, copy_files=landmark_path is None)
        if landmark_path is not None:
            landmark_path = os.path.abspath(landmark_path)
            self.project.set_landmarks(file_path, landmark_path)

    @QtCore.pyqtSlot(str, str, name="onLandmarkFileAdded")
    def on_landmark_file_added(self, file_path: str, landmark_path: str):
        """
        Adds a map from the video path to its landmarks
        :param file_path: Video Path
        :param landmark_path: Landmark Path
        """
        file_path = os.path.abspath(file_path)
        landmark_path = os.path.abspath(landmark_path)
        self.project.set_landmarks(file_path, landmark_path)

    @QtCore.pyqtSlot(str, name="onFileRemoved")
    def on_file_removed(self, file_path: str):
        """
        Removes a file from the project
        :param file_path: Video path
        """
        file_path = os.path.abspath(file_path)
        self.project.remove_video(file_path, clean_disk=True)

    @QtCore.pyqtSlot(str, name="onSaveLocChange")
    def on_save_loc_change(self, save_dir: str):
        """
        Catches when a new save location has been set
        :param save_dir: The path to the new save location
        """
        if len(save_dir) == 0 and self.project.save_loc is None:
            # Then the project is invalid
            self.send_message("A new project must have a save location", self.hide)
        elif len(save_dir) > 0:
            # Then we have a new save location
            loaded_project = DataHolders.Project.load(save_dir, fail_to_none=True)
            if loaded_project is None:
                self.project.set_save_loc(save_dir)
            else:
                self.project = loaded_project
                self.init_project(self.project)

    @QtCore.pyqtSlot(str, name="addFanModel")
    def add_fan_model(self, path: str):
        self.project.add_FAN(path)
        self.fanAdded.emit()

    @QtCore.pyqtSlot(str, name="addS3fdModel")
    def add_s3fd_model(self, path: str):
        self.project.add_s3fd(path)
        self.s3fdAdded.emit()

    @QtCore.pyqtSlot(name="onFinish")
    @QtCore.pyqtSlot(str, name="onFinish")
    def on_finish(self, save_dir: str = None):
        if save_dir is None:
            save_dir = self.project.save_loc
        error = self.validate(save_dir)
        if error is None:
            self.project.save()
            self.set_title(f"Editing Project: {self.project.name}")
            self.send_message("Saved Project", lambda: self.hide())
            self._glo.select_project(self.project.save_loc)
        else:
            self.send_message(error)

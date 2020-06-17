import QtQuick 2.0
import QtQuick.Controls 2.13
import QtQuick.Dialogs 1.1

ApplicationWindow {
    id: main
    flags: Qt.WindowStaysOnTopHint
    objectName: "window"
    width: 640
    height: 480

    FileDialog {
        id: fileDialog
        title: "Select the videos you wish to add"
        selectExisting: true
        selectMultiple: true
        selectedNameFilter: "Video files (*.mp4 *.MOV)"
        sidebarVisible: true
        onAccepted: {
            for(const filePath of fileUrls){
                fileListModel.addFile(filePath.substring(7));
            }
        }
    }

    FileDialog {
        id: addFanDialog
        title: "Select the fan model you wish to add"
        selectExisting: true
        selectMultiple: false
//        selectedNameFilter: ""
        sidebarVisible: true
        onAccepted: {
            handler.addFanModel(fileUrls[0].substring(7))
        }
    }

    FileDialog {
        id: addS3fdDialog
        title: "Select the fan model you wish to add"
        selectExisting: true
        selectMultiple: false
//        selectedNameFilter: ""
        sidebarVisible: true
        onAccepted: {
            handler.addS3fdModel(fileUrls[0].substring(7))
        }
    }

    FileDialog {
        id: saveLocationDialog
        objectName: "saveLocationDialog"
        title: "Select the videos you wish to add"
        selectExisting: false
        selectFolder: true
        selectMultiple: false
        sidebarVisible: true
        onAccepted: {
            handler.onSaveLocChange(fileUrls[0].substring(7))
        }
        onRejected: {
            handler.onSaveLocChange(undefined)
        }
    }

    MessageDialog {
        id: messageDialog
        objectName: "messageDialog"
        text: "Default"
    }

    Rectangle {
        id: background
        color: "#f9fafa"
        anchors.fill: parent
    }

    Item {
        id: container
        anchors.rightMargin: 8
        anchors.leftMargin: 8
        anchors.bottomMargin: 8
        anchors.topMargin: 8
        anchors.fill: parent

        ScrollView {
            id: fileContainerScrollview
            anchors.bottom: configButtonContainer.top
            anchors.bottomMargin: 0
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: 0
            anchors.top: projectNameInput.bottom
            anchors.topMargin: 10

            ListView {
                id: fileListView
                anchors.fill: parent
                model: fileListModel
                delegate: Item{
                    anchors.left: parent.left
                    anchors.right: parent.right
                    height: 26

                    Text {
                        height: 24
                        text: "<b>File Name: </b>" + fileName
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: parent.left
                        anchors.right: parent.horizontalCenter
                        anchors.rightMargin: 100
                    }

                    Text {
                        height: 24
                        text: "<b>Length(Frames): </b>" + frames
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: parent.horizontalCenter
                        anchors.leftMargin: -100
                    }

                    Button {
                        id: selectLandmarksButton
                        height: 24
                        anchors.verticalCenter: parent.verticalCenter
                        text: "Select Landmarks"
                        anchors.right: deleteButton.left
                        anchors.rightMargin: 2
                        visible: true
                        onClicked: {
                            landmarkFileDialog.visible = true
                            landmarkFileDialog.open()
                        }
                    }

                    Button {
                        id: deleteButton
                        height: 24
                        anchors.verticalCenter: parent.verticalCenter
                        text: "X"
                        onClicked: fileListModel.removeFile(index)
                        anchors.right: parent.right
                    }

                    FileDialog {
                        id: landmarkFileDialog
                        title: "Select the cooresponding landmarks"
                        selectExisting: true
                        selectMultiple: false
                        selectedNameFilter: "CSV file (*.csv)"
                        sidebarVisible: true
                        onAccepted: {
                            fileListModel.setLandmarks(index, fileUrls[0].substring(7))
                        }
                    }
                }
            }
        }

        Rectangle {
            id: titleBar
            color: "#73808c"
            anchors.bottom: projectNameInput.bottom
            anchors.bottomMargin: -6
            anchors.right: parent.right
            anchors.rightMargin: -8
            anchors.left: parent.left
            anchors.leftMargin: -8
            anchors.top: parent.top
            anchors.topMargin: -8
        }

        Text {
            id: title
            objectName: "title"
            text: qsTr("Create Project")
            height: 24
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.top: parent.top
            anchors.topMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: 10
            font.pixelSize: 20
        }

        TextField {
            id: projectNameInput
            objectName: "project_name_input"
            height: 24
            anchors.top: title.bottom
            anchors.topMargin: 5
            padding: 0
            font.pointSize: 12
            anchors.right: parent.horizontalCenter
            anchors.rightMargin: 100
            anchors.left: parent.left
            anchors.leftMargin: 0
            placeholderText: qsTr("Project Name")
            onTextEdited: handler.onNameChange(projectNameInput.text)
        }

        Item {
            id: configButtonContainer
            height: 40
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 0
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: 0

            Button {
                id: addFileButton
                objectName: "add_file_button"
                text: qsTr("Add File")
                anchors.left: parent.left
                anchors.leftMargin: 0
                onClicked: {
                    fileDialog.visible = true
                    fileDialog.open()
                }
            }

            Button {
                id: addFanButton
                text: handler.hasFan ? qsTr("Add Another FAN Model") : qsTr("Add FAN Model")
                anchors.left: addFileButton.right
                anchors.leftMargin: 3
                anchors.top: addFileButton.top
                anchors.topMargin: 0
                anchors.bottom: addFanButton.bottom

                onClicked: {
                    addFanDialog.open()
                }
            }

            Button {
                id: button1
                text: handler.hasS3fd ? qsTr("Add Another s3fd Model") : qsTr("Add s3fd Model")
                anchors.left: addFanButton.right
                anchors.leftMargin: 3
                anchors.top: addFileButton.top
                anchors.topMargin: 0

                onClicked: {
                    addS3fdDialog.open()
                }
            }

            Button {
                id: finishProjectButton
                text: qsTr("Finish")
                objectName: "finish_project_button"
                anchors.right: parent.right
                anchors.rightMargin: 0
                onClicked: {
                    handler.hasSaveLoc() ? handler.onFinish() : saveLocationDialog.open()
                }
            }


        }

    }

    Connections {
        target: handler

        onProjectNameChanged: {
            if(projectName !== ""){
                title.text = "Project: " + projectName
            }else{
                title.text = "Create Project"
            }
        }

        onProjectOpened: {
            projectNameInput.text = projectName
        }
    }


}

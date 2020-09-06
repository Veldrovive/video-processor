import QtQuick 2.0
import QtQuick.Controls 2.13
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.15

ApplicationWindow {
    id: projectView
    objectName: "window"
    width: 720
    height: 480
    color: "#ecf0f1"

    property string currVideoPath: ""

    MessageDialog {
        id: messageDialog
        objectName: "messageDialog"
        text: "Default"
    }

    FileDialog {
        id: addLandmarksDialog
        title: "Select the landmarks you wish to add"
        selectExisting: true
        selectMultiple: false
        selectedNameFilter: "CSV files (*.csv)"
        sidebarVisible: true
        onAccepted: {
            handler.add_landmarks(projectView.currVideoPath, fileUrls[0].substring(7))
        }
    }

    FileDialog {
        id: addFilesDialog
        title: "Select the videos you wish to add"
        selectExisting: true
        selectMultiple: true
        selectedNameFilter: "Video files (*.mp4 *.MOV)"
        sidebarVisible: true
        onAccepted: {
            for(const filePath of fileUrls){
                handler.add_file(filePath);
            }
        }
    }

    FileDialog {
        id: setSaveLocDialog
        title: "Select a folder to save the project in"
        selectExisting: false
        selectFolder: true
        selectMultiple: false
        sidebarVisible: true
        onAccepted: {
            handler.save_loc = fileUrls[0];
        }
        onRejected: {
            handler.save_loc = "";
        }
    }

    Rectangle {
        id: header
        property var vertPadding: 5

        color: "#34495e"
        anchors.right: parent.right
        anchors.left: parent.left
        anchors.top: parent.top
        height: childrenRect.height + header.vertPadding * 2


        Text {
            id: headerText

            anchors.left: parent.left
            anchors.leftMargin: 10
            anchors.top: parent.top
            anchors.topMargin: header.vertPadding

            color: "#ecf0f1"
            text: {
                if (handler.name.length > 0) {
                    return qsTr(`Editing Project: ${handler.name}`)
                } else {
                    return qsTr("Create Project")
                }
            }
            font.pointSize: 18
        }
    }

    Item {
        id: mainContainer
        anchors.top: header.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 8

        Item {
            id: nameContainer
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            height: 36

            Text {
                id: nameFieldLabel
                anchors.left: parent.left
                anchors.verticalCenter: parent.verticalCenter
                text: qsTr("Name: ")
            }

            TextField {
                id: nameTextField
                anchors.left: nameFieldLabel.right
                anchors.leftMargin: 3
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                placeholderText: qsTr("Project Name")

                text: handler.name

                onTextEdited: {
                    handler.name = text
                }
            }
        }

        ScrollView {
            id: videosScroll
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: nameContainer.bottom
            anchors.topMargin: 8
            anchors.bottom: buttonsContainer.top
            anchors.bottomMargin: 8
            clip: true
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

            ListView {
                id: videosList
                model: handler.files
                anchors.fill: parent

                delegate: Rectangle {
                    id: video
                    anchors.left: parent.left
                    anchors.right: parent.right
                    height: 36
                    color: "#ecf0f1"
                    property string name: handler.files[index].name
                    property string path: handler.files[index].path

                    Text {
                        anchors.left: parent.left
                        anchors.verticalCenter: parent.verticalCenter

                        text: qsTr("<b>File Name: </b>" + video.name)
                    }

                    Button {
                        id: addLandmarksButton
                        anchors.right: deleteFileButton.left
                        anchors.rightMargin: 4
                        anchors.verticalCenter: parent.verticalCenter
                        height: 26

                        text: qsTr("Manually Add Landmarks")

                        onClicked: {
                            // Open a file chooser menu
                            projectView.currVideoPath = video.path
                        }
                    }

                    Button {
                        id: deleteFileButton
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        height: 26

                        text: qsTr("Delete")

                        onClicked: {
                            handler.remove_file(video.path)
                        }
                    }
                }
            }
        }

        Item {
            id: buttonsContainer
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: childrenRect.height

            Button {
                id: addFileButton
                anchors.left: parent.left
                anchors.bottom: parent.bottom

                text: qsTr("Add Files")

                onClicked: {
                    addFilesDialog.open();
                }
            }

            Button {
                id: saveButton
                anchors.right: parent.right
                anchors.bottom: parent.bottom

                text: qsTr("Save")

                onClicked: {
                    handler.save();
                }
            }
        }
    }

    Connections {
        target: handler

        function onOpenSaveLocDialog() {
            setSaveLocDialog.open()
        }
    }
}

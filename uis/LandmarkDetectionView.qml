import QtQuick 2.0
import QtQuick.Controls 2.13
import QtQuick.Dialogs 1.1

ApplicationWindow {
    id: main
    flags: Qt.WindowStaysOnTopHint
    objectName: "landmarkDetectionView"
    width: 640
    height: 480

    MessageDialog {
        id: messageDialog
        objectName: "messageDialog"
        text: "Default"
    }

    Rectangle {
        id: background
        color: "#ecf0f1"
        anchors.fill: parent
    }

    Item {
        id: container
        anchors.rightMargin: 8
        anchors.leftMargin: 8
        anchors.bottomMargin: 8
        anchors.topMargin: 8
        anchors.fill: parent

        Rectangle {
            id: titleBar
            z: 3
            color: "#34495e"
            anchors.bottom: title.bottom
            anchors.bottomMargin: -8
            anchors.right: parent.right
            anchors.rightMargin: -8
            anchors.left: parent.left
            anchors.leftMargin: -8
            anchors.top: parent.top
            anchors.topMargin: -8
        }

        Text {
            id: title
            z: 4
            objectName: "title"
            text: qsTr("Detect Landmarks For:")
            color: "#95a5a6"
            height: 24
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.top: parent.top
            anchors.topMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: 10
            font.pixelSize: 20
        }

        ListView {
            id: listView
            objectName: "listView"
            anchors.bottom: summaryText.top
            anchors.bottomMargin: 0
            anchors.top: titleBar.bottom
            anchors.topMargin: 10
            anchors.right: parent.horizontalCenter
            anchors.rightMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: 0

            model: fileListModel
            currentIndex: -1
            delegate: Rectangle {
                id: itemRect
                radius: 5
                anchors.left: parent.left
                anchors.right: parent.right
                color: {
                    if(ListView.isCurrentItem){
                        return "#95a5a6"
                    }
                    return state == "HOVER" ? "#bdc3c7" : "#ecf0f1"
                }
                height: 26

                Text {
                    text: qsTr(fileName)
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.left: parent.left
                    anchors.leftMargin: 3
                }

                states: [
                    State {
                        name: "NORMAL"
                    },
                    State {
                        name: "HOVER"
                    }
                ]

                MouseArea {
                    id: mouse_area1
                    z: 1
                    hoverEnabled: true
                    anchors.fill: parent

                    onClicked: {
                        itemRect.ListView.view.currentIndex = index;
                        handler.getVideoData(index)
                    }

                    onEntered: {
                        parent.state = "HOVER"
                    }

                    onExited: {
                        parent.state = "NORMAL"
                    }
                }
            }
        }

        CheckBox {
            id: fullVideoCheck
            text: qsTr("Detect Full Video")
            font.pointSize: 14
            anchors.top: titleBar.bottom
            anchors.topMargin: 10
            anchors.left: parent.horizontalCenter
            anchors.leftMargin: 0

            onToggled: {
                handler.setFullVideo(checked)
            }
        }

        ListView {
            id: keyPointList
            anchors.top: keyframesLabel.bottom
            anchors.topMargin: 3
            anchors.bottom: summaryText.top
            anchors.bottomMargin: 0
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.left: parent.horizontalCenter
            anchors.leftMargin: 0

            model: keypointListModel
            delegate: Item {
                anchors.left: parent.left
                anchors.right: parent.right

                height: 32


                CheckBox {
                    id: keypointCheckbox
                    anchors.left: parent.left
                    height: 32
                    checked: {
                        return keypointListModel.isChecked(index);
                    }

                    onToggled: {
                        handler.setFileKeypoint(index, keypointCheckbox.checked)
                    }
                }

                Text {
                    text: keypoint
                    anchors.left: keypointCheckbox.right
                    anchors.leftMargin: 3
                    anchors.verticalCenter: parent.verticalCenter
                }
            }
        }

        Text {
            id: summaryText

            anchors.left: parent.left
            anchors.rightMargin: 3
            anchors.right: detectButton.left
            anchors.bottom: progressBar.top
            anchors.top: detectButton.top

            text: qsTr("Num Frames:")
            anchors.bottomMargin: 3
            verticalAlignment: Text.AlignVCenter
            font.pixelSize: 12
        }

        Button {
            id: detectButton

            anchors.right: parent.right
            anchors.bottom: progressBar.top

            text: qsTr("Detect")
            anchors.bottomMargin: 3

            onClicked: {
                handler.detect()
            }
        }

        Text {
            id: rangeChooseText
            text: qsTr("Choose a range:")
            verticalAlignment: Text.AlignVCenter
            anchors.bottom: rangeChooser.bottom
            anchors.bottomMargin: 0
            anchors.left: parent.horizontalCenter
            anchors.leftMargin: 0
            anchors.top: rangeChooser.top
            anchors.topMargin: 0
            font.pixelSize: 14
        }

        TextField {
            id: rangeChooser
            anchors.top: fullVideoCheck.bottom
            anchors.topMargin: 5
            anchors.left: rangeChooseText.right
            anchors.leftMargin: 10
            placeholderText: qsTr("1-10, 20-30")

            onEditingFinished: {
                handler.setFrameRange(text)
            }
        }


        ProgressBar {
            id: progressBar
            objectName: "progressBar"
            anchors.left: parent.left
            anchors.leftMargin: 0
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 0
            value: 0
        }


        Text {
            id: keyframesLabel
            text: qsTr("Key Frames:")
            z: 3
            anchors.left: parent.horizontalCenter
            anchors.leftMargin: 0
            anchors.top: rangeChooseText.bottom
            anchors.topMargin: 3
            font.pixelSize: 14

            Rectangle {
                z: -1
                id: rectangle
                color: "#ecf0f1"
                anchors.bottomMargin: -3
                anchors.topMargin: -14
                anchors.fill: parent
            }
        }
    }

    Connections {
        target: handler

        onProjectChanged: {
            console.log("Project Changed", handler.project_name)
            title.text = "Detect Landmarks For: <b>" + handler.project_name + "</b>"
        }

        onVideoChanged: {
            fullVideoCheck.checked = handler.curr_full_video_state
            rangeChooser.text = handler.curr_video_range_state
        }

        onNumFramesChanged: {
            summaryText.text = "Num Frames: " + handler.total_frames
        }

        onProgressChanged: {
            progressBar.value = progress
        }
    }
}

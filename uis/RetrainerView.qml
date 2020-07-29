import QtQuick 2.0
import QtQuick.Controls 2.13
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.15

ApplicationWindow {
    id: main
    flags: Qt.WindowStaysOnTopHint
    objectName: "window"
    width: 640
    height: 720
    color: "#ecf0f1"

    MessageDialog {
        id: messageDialog
        objectName: "messageDialog"
        text: "Default"
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
                    return qsTr(`Retrain Network: ${handler.name}`)
                } else {
                    return qsTr("Retrain Facial Analysis Network (FAN)")
                }
            }
            font.pointSize: 18
        }
    }

    TabBar {
        id: tabBar
        width: parent.width
        anchors.top: header.bottom

        TabButton {
            text: qsTr("Basic Settings")
        }

        TabButton {
            text: qsTr("Advanced Settings")
        }
    }

    StackLayout {
        id: mainContent
        currentIndex: tabBar.currentIndex
        anchors.top: tabBar.bottom
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.right: parent.right

        Item {
            id: basicContent
            anchors.top: parent.top
            anchors.topMargin: 8
            anchors.left: parent.left
            anchors.leftMargin: 8
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8

            Item {
                id: creationBox
                width: parent.width
                anchors.top: parent.top
                height: loadCheckpointButton.height

                Text {
                    id: nameFieldLabel
                    anchors.left: parent.left
                    anchors.leftMargin: 8
                    anchors.verticalCenter: parent.verticalCenter
                    text: qsTr("Name:")
                }

                TextField {
                    id: nameTextField
                    anchors.left: nameFieldLabel.right
                    anchors.leftMargin: 3
                    anchors.right: loadCheckpointButton.left
                    anchors.rightMargin: 8
                    placeholderText: qsTr("Model Name")

                    text: handler.name

                    onTextEdited: {
                        handler.name = text
                    }
                }

                Button {
                    id: loadCheckpointButton
                    anchors.right: parent.right
                    text: qsTr("Load Checkpoint")

                    FileDialog {
                        id: loadCheckpointDialog
                        title: "Load a checkpoint"
                        selectMultiple: false
                        selectFolder: false
                        folder: handler.checkpointPath
                        //            fileMode: Platform.FileDialog.SaveFile

                        onAccepted: {
                            handler.loadCheckpoint(fileUrls[0].substring(7))
                        }
                    }

                    onClicked: {
                        loadCheckpointDialog.open();
                    }
                }
            }

            Item {
                id: selector
                width: parent.width
                anchors.top: creationBox.bottom
                anchors.topMargin: 8
                anchors.bottom: progressBars.top
                anchors.bottomMargin: 8

                Text {
                    id: videosLabel
                    anchors.left: parent.left
                    anchors.leftMargin: 4

                    text: qsTr("Videos:")
                }

                Text {
                    id: framesLabel
                    anchors.left: parent.horizontalCenter
                    anchors.leftMargin: 4

                    text: qsTr("Frames:")
                }

                Frame {
                    id: videosFrame
                    anchors.left: parent.left
                    anchors.right: parent.horizontalCenter
                    anchors.top: videosLabel.bottom
                    anchors.bottom: parent.bottom
                    padding: 1

                    ScrollView {
                        id: videoSelector
                        anchors.fill: parent
                        clip: true
                        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                        ListView {
                            id: videosList
                            anchors.fill: parent

                            model: handler.videos_list

                            delegate: MouseArea {
                                height: 36
                                width: videosList.width
                                id: videoMouseArea

                                Rectangle {
                                    id: videoBox
                                    anchors.fill: parent
                                    property string videoName: handler.videos_list[index].name

                                    color: {
                                        let color = "#ecf0f1"
                                        if (index % 2 === 0) color = "#95a5a6";
                                        if (handler.currentVideoIndex === index) color = "#7d8b8c";
                                        return color;
                                    }

                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        anchors.left: parent.left
                                        anchors.leftMargin: 3
                                        text: qsTr(videoBox.videoName)
                                    }
                                }

                                onClicked: {
                                    handler.updateCurrentVideo(index);
                                }
                            }
                        }
                    }
                }


                Frame {
                    id: framesFrame
                    anchors.left: parent.horizontalCenter
                    width: parent.width / 2
                    anchors.top: framesLabel.bottom
                    anchors.bottom: parent.bottom
                    padding: 1

                    property bool deleteEnabled: true

                    ScrollView {
                        id: frameSelector
                        anchors.fill: parent
                        clip: true
                        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                        ListView {
                            id: framesList
                            anchors.fill: parent

                            model: handler.frames_list

                            delegate: MouseArea {
                                id: frameMouseArea
                                width: framesList.width
                                height: 36
                                property int frameNumber: handler.frames_list[index].frame_number

                                Rectangle {
                                    id: frameBox
                                    anchors.fill: parent

                                    color: index % 2 == 0 ? "#95a5a6" : "#ecf0f1"

                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        anchors.left: parent.left
                                        anchors.leftMargin: 10
                                        text: {
                                            return qsTr((frameMouseArea.frameNumber + 1).toString());
                                        }
                                    }

                                    Button {
                                        id: deleteFrameButton
                                        anchors.right: parent.right
                                        anchors.rightMargin: 10
                                        anchors.verticalCenter: parent.verticalCenter
                                        height: parent.height * 0.7
                                        enabled: framesFrame.deleteEnabled
                                        text: qsTr("Delete");

                                        onClicked: {
                                            handler.deleteFrame(index);
                                        }
                                    }
                                }

                                onClicked: {
                                    handler.setCurrentFrame(frameNumber);
                                }
                            }
                        }

                        Connections {
                            target: handler

                            function onTrainingStarted() {
                                framesFrame.deleteEnabled = false;
                            }

                            function onTrainingFinished() {
                                framesFrame.deleteEnabled = true;
                            }
                        }
                    }
                }
            }

            Item {
                id: progressBars
                height: 50
                anchors.bottom: retrainButton.top
                anchors.left: parent.left
                anchors.right: parent.right
                Item {
                    id: epochBarContainer
                    anchors.top: parent.top
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.bottom: parent.verticalCenter

                    property int curr_epoch: -1
                    property int max_epoch: -1

                    Text {
                        id: epochText
                        anchors.left: parent.left
                        anchors.verticalCenter: parent.verticalCenter
                        text: {
                            if (epochBarContainer.curr_epoch < 0) {
                                return qsTr("Epoch:");
                            } else {
                                return qsTr(`Epoch(${epochBarContainer.curr_epoch}/${epochBarContainer.max_epoch}):`)
                            }
                        }
                    }

                    ProgressBar {
                        id: epochBar
                        anchors.left: epochText.right
                        anchors.leftMargin: 4
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter

                        value: 0
                    }

                    Connections {
                        target: handler

                        function onEpochCompleted(current, max, progress) {
                            epochBarContainer.curr_epoch = current;
                            epochBarContainer.max_epoch = max;
                            epochBar.value = progress;
                        }
                    }
                }

                Item {
                    id: batchBarContainer
                    anchors.top: parent.verticalCenter
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom

                    property int curr_batch: -1
                    property int max_batch: -1

                    Text {
                        id: batchText
                        anchors.left: parent.left
                        anchors.verticalCenter: parent.verticalCenter
                        text: {
                            if (batchBarContainer.curr_batch < 0) {
                                return qsTr("Batch: ");
                            } else {
                                return qsTr(`Batch(${batchBarContainer.curr_batch}/${batchBarContainer.max_batch}): `)
                            }
                        }
                    }

                    ProgressBar {
                        id: batchBar
                        anchors.left: batchText.right
                        anchors.leftMargin: 4
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter

                        value: 0

                    }

                    Connections {
                        target: handler

                        function onBatchCompleted(current, max, progress) {
                            batchBarContainer.curr_batch = current;
                            batchBarContainer.max_batch = max;
                            batchBar.value = progress;
                        }
                    }
                }
            }

            Button {
                id: retrainButton
                text: qsTr("Retrain")
                anchors.bottom: parent.bottom
                anchors.horizontalCenter: parent.horizontalCenter
                enabled: framesFrame.deleteEnabled

                onClicked: {
                    handler.retrain()
                }
            }
        }

        Item {
            id: advancedContent
            anchors.top: parent.top
            anchors.topMargin: 8
            anchors.left: parent.left
            anchors.leftMargin: 8
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8

            Text {
                text: qsTr("Test")
            }
        }
    }
}

import QtQuick 2.0
import QtQuick.Controls 2.13
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.15

ApplicationWindow {
    id: main
    objectName: "landmarkDetectionView"
    width: 640
    height: 480

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
            text: qsTr("Estimate Landmarks")
            font.pointSize: 18
        }
    }

    TabBar {
        id: tabBar
        width: parent.width
        currentIndex: 0
        anchors.top: header.bottom

        TabButton {
            text: qsTr("Detector")
        }

        TabButton {
            text: qsTr("Settings")
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
            id: detectorContent
            anchors.fill: parent
            anchors.margins: 8

            Text {
                id: videosLabel
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.leftMargin: 4

                text: qsTr("Videos:")
            }

            Frame {
                anchors.left: parent.left
                anchors.top: videosLabel.bottom
                anchors.bottom: progressBar.top
                anchors.bottomMargin: 3
                anchors.right: parent.horizontalCenter
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
                                    if (handler.curr_video_index === index) color = "#7d8b8c";
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
                                handler.curr_video_index = index;
                            }
                        }
                    }
                }
            }

            Frame {
                anchors.left: parent.horizontalCenter
                anchors.top: videosLabel.bottom
                anchors.bottom: progressBar.top
                anchors.bottomMargin: 3
                anchors.right: parent.right
                padding: 1

                CheckBox {
                    id: fullVideoCheckbox
                    anchors.top: parent.top
                    anchors.left: parent.left

                    enabled: {
                        return handler.curr_video_index >= 0;
                    }

                    checked: {
                        return handler.curr_all_frames;
                    }

                    font.pointSize: 14
                    text: qsTr("Full Video")

                    onToggled: {
                        handler.curr_all_frames = checked;
                    }
                }

                Item {
                    id: chooseRangeContainer
                    anchors.top: fullVideoCheckbox.bottom
                    anchors.left: parent.left
                    height: 36

                    Text {
                        id: chooseRangeLabel
                        anchors.left: parent.left
                        anchors.top: parent.top
                        anchors.verticalCenter: parent.verticalCenter
                        verticalAlignment: Text.AlignVCenter
                        text: qsTr("Choose a range:")
                    }

                    TextField {
                        id: chooseRangeTextbox
                        anchors.left: chooseRangeLabel.right
                        anchors.top: parent.top
                        anchors.verticalCenter: parent.verticalCenter
                        placeholderText: qsTr("1-10, 20-30")

                        enabled: {
                            return handler.curr_video_index >= 0 && !handler.curr_all_frames;
                        }

                        text: {
                            if (enabled) {
                                return handler.curr_some_frames;
                            }
                            return '';
                        }

                        onTextEdited: {
                            handler.curr_some_frames = text;
                        }

                        onEditingFinished: {
                            focus = false;
                            handler.finish_some_frames();
                        }
                    }
                }

                Rectangle {
                    id: seperator
                    color: "#c5d2e3"
                    anchors.left: parent.left
                    anchors.leftMargin: 5
                    anchors.topMargin: 8
                    anchors.right: parent.right
                    anchors.rightMargin: 5
                    height: 1
                    anchors.top: chooseRangeContainer.bottom
                    anchors.bottomMargin: 8
                }

                Text {
                    id: keypointSelectorLabel
                    anchors.top: seperator.bottom
                    anchors.left: parent.left
                    anchors.topMargin: 8
                    anchors.leftMargin: 3

                    text: qsTr("Keypoints:")
                }

                Button {
                    id: toggleKeypointsButton
                    anchors.top: keypointSelectorLabel.top
                    anchors.left: keypointSelectorLabel.right
                    anchors.leftMargin: 3
                    height: 16

                    text: qsTr("Toggle All Keypoints")

                    enabled: {
                        return handler.curr_video_index >= 0 && !handler.curr_all_frames;
                    }

                    onClicked: {
                        handler.toggle_all_keypoints()
                    }
                }

                ScrollView {
                    id: keypointsSelector
                    clip: true
                    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    anchors.top: keypointSelectorLabel.bottom
                    anchors.topMargin: 3

                    ListView {
                        id: keypointsList
                        anchors.fill: parent

                        model: handler.curr_keypoints

                        delegate: MouseArea {
                            height: 36
                            width: videosList.width
                            id: keypointMouseArea

                            Rectangle {
                                id: keypointBox
                                anchors.fill: parent
                                property int frame: handler.curr_keypoints[index]["frame"]
                                property bool active: handler.curr_keypoints[index]["active"]

                                color: {
                                    let color = "#ecf0f1"
                                    if (index % 2 === 0) color = "#95a5a6";
                                    return color;
                                }

                                CheckBox {
                                    id: keypointCheckbox
                                    anchors.left: parent.left
                                    anchors.verticalCenter: parent.verticalCenter
                                    anchors.leftMargin: 3

                                    enabled: {
                                        return handler.curr_video_index >= 0 && !handler.curr_all_frames;
                                    }

                                    checked: {
                                        return keypointBox.active && enabled
                                    }

                                    onToggled: {
                                        handler.set_keypoint_active(keypointBox.frame, keypointCheckbox.checked);
                                    }
                                }

                                Text {
                                    anchors.verticalCenter: parent.verticalCenter
                                    anchors.left: keypointCheckbox.right
                                    anchors.leftMargin: 3
                                    text: qsTr((keypointBox.frame + 1).toString())
                                }
                            }

                            onClicked: {
                                handler.go_to_frame(keypointBox.frame)
                            }
                        }
                    }
                }
            }

            ProgressBar {
                id: progressBar
                value: 0
                anchors.bottom: detectButton.top
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottomMargin: 4

                Connections {
                    target: handler

                    function onProgressChanged(progress) {
                        progressBar.value = progress
                    }
                }
            }

            Button {
                id: detectButton
                anchors.bottom: parent.bottom
                anchors.horizontalCenter: parent.horizontalCenter
                text: qsTr("Run Detection")

                enabled: {
                    console.log(handler.detecting)
                    !handler.detecting
                }

                onClicked: {
                    handler.detect();
                }
            }

            Text {
                id: totalFramesLabel
                anchors.verticalCenter: detectButton.verticalCenter
                anchors.left: parent.left

                text: {
                    return `Total Frames: ${handler.total_frames}`
                }
            }
        }

        Item {
            id: settingsContent
            anchors.fill: parent
            anchors.margins: 8

            Text {
                id: modelSectionLabel
                anchors.left: parent.left
                anchors.top: parent.top

                text: qsTr("Select Model: ")
                font.pointSize: 14
                font.bold: true
            }

            Text {
                id: modelNameLabel
                anchors.left: parent.left
                anchors.verticalCenter: modelNameChooser.verticalCenter

                text: qsTr("Model Name:")
            }

            ComboBox {
                id: modelNameChooser
                anchors.left: modelNameLabel.right
                anchors.leftMargin: 3
                anchors.top: modelSectionLabel.bottom
                anchors.topMargin: 4

                currentIndex: {
                    return find(handler.model_name)
                }

                model: handler.model_names

                onActivated: {
                    handler.set_model_name(currentText)
                }
            }

            Text {
                id: modelVersionLabel
                anchors.left: modelNameChooser.right
                anchors.leftMargin: 8
                anchors.verticalCenter: modelVersionChooser.verticalCenter

                text: qsTr("Model Version:")
            }

            ComboBox {
                id: modelVersionChooser
                anchors.left: modelVersionLabel.right
                anchors.leftMargin: 3
                anchors.top: modelSectionLabel.bottom
                anchors.topMargin: 4

                currentIndex: {
                    return find(handler.model_version)
                }

                model: handler.model_versions

                onActivated: {
                    handler.set_model_version(currentText)
                }
            }
        }
    }
}

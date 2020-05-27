import QtQuick 2.0
import QtQuick.Controls 2.13
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.14

import Backend 1.0

ApplicationWindow {
    id: applicationWindow
    flags: Qt.WindowStaysOnTopHint
    width: 640
    height: 920
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
            text: qsTr("Metric Graphs")
            font.pointSize: 18
        }
    }

    TabBar {
        id: fileBar
        property int memberWidth: 100
        width: parent.width

        anchors.top: header.bottom
        anchors.left: parent.left
        anchors.right: parent.right

        Repeater {
            id: fileBarNameRepeater
            model: handler.files_list

            TabButton {
                text: modelData.name
                width: implicitWidth + 20

                onClicked: {
                    handler.change_file(modelData.path)
                }
            }


        }
    }

    Text {
        id: fileNameText

        anchors.top: fileBar.bottom
        anchors.topMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8

        text: "Current File: " + fileBarNameRepeater.model[0].name
        font.pointSize: 18

        Connections {
            target: handler
            onFileChange: {
                fileNameText.text = "Current File: " + fileName
            }
        }
    }

    Item {
        id: normalizationOptions
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.top: fileNameText.bottom
        anchors.topMargin: 8
        height: childrenRect.height

        Text {
            id: normalizeOnText

            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter

            text: qsTr("Normalize On Metric: ")
        }

        ComboBox {
            id: normalizeOnCombo
            property int lastSelected: 0
            property var choices: handler.metrics_names

            anchors.left: normalizeOnText.right
            anchors.verticalCenter: parent.verticalCenter

            model: ["None"].concat(handler.metrics_names)

            function reset() {
                lastSelected = 0
                currentIndex = 0
            }

            onActivated: {
                if (currentIndex !== lastSelected){
                    lastSelected = currentIndex

                    handler.on_normalization_combo_set(currentIndex, currentValue)
                }
            }
        }

        SpinBox {
            id: normalizeOnSpin
            editable: true
            from: 0
            value: 100
            to: 100 * 10000000
            stepSize: 1

            anchors.verticalCenter: parent.verticalCenter
            anchors.left: parent.horizontalCenter

            property int decimals: 2
            property real realValue: value / 100

            validator: DoubleValidator {
                bottom: Math.min(normalizeOnSpin.from, normalizeOnSpin.to)
                top:  Math.max(normalizeOnSpin.from, normalizeOnSpin.to)
            }

            textFromValue: function(value, locale) {
                return Number(value / 100).toLocaleString(locale, 'f', normalizeOnSpin.decimals)
            }

            valueFromText: function(text, locale) {
                return Number.fromLocaleString(locale, text) * 100
            }

            onValueModified: {
                normalizeOnCombo.reset()

                handler.on_normalization_value_set(realValue)
            }

            function setValue(value){
                normalizeOnSpin.value = value * 100
            }

            Connections {
                target: handler
                onNormalizationValueChanged: {
                    normalizeOnSpin.setValue(value)
                }
            }
        }

        Text {
            id: normalizeSpinText

            anchors.left: normalizeOnSpin.right
            anchors.leftMargin: 3
            anchors.verticalCenter: parent.verticalCenter

            text: qsTr("cm/pixel")
        }
    }

    ProgressBar {
        id: progressBar
        value: 0

        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.top: metricDisplayChooser.bottom
        anchors.topMargin: 8

        Connections {
            target: handler
            onProgressChanged: {
                progressBar.value = progress
            }
        }

    }

    Frame {
        id: metricDisplayChooser

        anchors.top: normalizationOptions.bottom
        anchors.topMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        height: applicationWindow.height / 4
        padding: 1
        rightPadding: 1
        bottomPadding: 1
        leftPadding: 1
        topPadding: 1

        ScrollView {
            id: metricDisplayChooserScroll
            anchors.fill: parent
            clip: true

            ListView {
                id: metricDisplayChooserList
                property int openedIndex: -1

                anchors.fill: parent

                model: handler.metrics_list

                delegate: MouseArea {
                    id: delegate
                    height: 36
                    anchors.left: parent.left
                    anchors.right: parent.right

                    Rectangle {
                        id: metricBox
                        property string name: handler.metrics_list[index].name
                        property string type: handler.metrics_list[index].type
                        anchors.fill: parent
                        color: index % 2 == 0 ? "#ecf0f1" : "#95a5a6"

                        CheckBox {
                            id: metricShowCheckbox
                            anchors.leftMargin: 8
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.left: parent.left

                            checked: handler.check_shown(metricBox.name)

                            onClicked: {
                                handler.set_metric_shown(index, metricBox.name, metricShowCheckbox.checked)
                            }
                        }

                        Item {
                            anchors.left: metricShowCheckbox.right
                            anchors.leftMargin: 3
                            anchors.top: parent.top
                            anchors.bottom: parent.bottom
                            anchors.right: parent.right

                            visible: index !== metricDisplayChooserList.openedIndex

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                anchors.left: parent.left
                                text: qsTr(metricBox.name)
                            }
                        }

                        Item {
                            anchors.left: metricShowCheckbox.right
                            anchors.leftMargin: 3
                            anchors.top: parent.top
                            anchors.bottom: parent.bottom
                            anchors.right: parent.right

                            visible: index === metricDisplayChooserList.openedIndex

                            TextField {
                                anchors.verticalCenter: parent.verticalCenter
                                anchors.left: parent.left

                                height: 26
                                text: qsTr(metricBox.name)

                                onEditingFinished: {
                                    metricDisplayChooserList.openedIndex = -1

                                    handler.change_metric_name(index, metricBox.name, text)
                                }
                            }

                            Button {
                                id: deleteMetricButton
                                anchors.verticalCenter: parent.verticalCenter
                                anchors.right: parent.right
                                anchors.rightMargin: 8

                                height: 28

                                text: qsTr("Delete Metric")

                                onClicked: {
                                    handler.delete_metric(index, metricBox.name)
                                }
                            }

                            ComboBox {
                                id: changeMetricTypeCombo
                                anchors.verticalCenter: parent.verticalCenter
                                anchors.right: deleteMetricButton.left
                                anchors.rightMargin: 3

                                height: 28

                                model: ["Distance", "Area"]

                                currentIndex: metricBox.type == "LENGTH" ? 0 : 1

                                onActivated: {
                                    handler.change_metric_type(index, metricBox.name, currentText)
                                }
                            }
                        }
                    }

                    onDoubleClicked: {
                        if (index === metricDisplayChooserList.openedIndex) {
                            metricDisplayChooserList.openedIndex = -1
                        } else {
                            metricDisplayChooserList.openedIndex = index
                        }
                    }
                }
            }
        }
    }

    Item {
        id: figureArea

        anchors.top: progressBar.bottom
        anchors.topMargin: 3
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.bottom: saveMetricsButton.top
        anchors.bottomMargin: 8

        FigureCanvas {
            id: graphArea
            objectName: "figure"
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: toolbar.top


        }

        ToolBar {
            id: toolbar
            property int imgPad: 6
            anchors.bottom: parent.bottom
            anchors.right: parent.right
            anchors.left: parent.left
            RowLayout {
                ToolButton {
                    onClicked: {
                        displayBridge.home();
                    }

                    Image {
                        height: parent.height - toolbar.imgPad * 2
                        anchors.centerIn: parent
                        fillMode: Image.PreserveAspectFit
                        source: "../icons/home.svg"
                    }
                }

                Button {
                    onClicked: {
                        displayBridge.back();
                    }

                    Image {
                        height: parent.height - toolbar.imgPad * 2
                        anchors.centerIn: parent
                        fillMode: Image.PreserveAspectFit
                        source: "../icons/back.svg"
                    }
                }

                Button {
                    onClicked: {
                        displayBridge.forward();
                    }

                    Image {
                        height: parent.height - toolbar.imgPad * 2
                        anchors.centerIn: parent
                        fillMode: Image.PreserveAspectFit
                        source: "../icons/forward.svg"
                    }
                }

                ToolSeparator{}

                Button {
                    id: pan
                    checkable: true
                    onClicked: {
                        if (zoom.checked) {
                            zoom.checked = false;
                        }
                        displayBridge.pan();
                    }

                    Image {
                        height: parent.height - toolbar.imgPad * 2
                        anchors.centerIn: parent
                        fillMode: Image.PreserveAspectFit
                        source: "../icons/pan.svg"
                    }
                }

                Button {
                    id: zoom
                    checkable: true
                    onClicked: {
                        if (pan.checked) {
                            // toggle pan off
                            pan.checked = false;
                        }
                        displayBridge.zoom();
                    }

                    Image {
                        height: parent.height - toolbar.imgPad * 2
                        anchors.centerIn: parent
                        fillMode: Image.PreserveAspectFit
                        source: "../icons/zoom.svg"
                    }
                }
                ToolSeparator {}
                TextInput {
                    id: location
                    readOnly: true
                    text: displayBridge.coordinates
                }
            }
        }
    }



    Button {
        id: saveMetricsButton
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8

        text: qsTr("Save Metrics")

        FileDialog {
            id: saveMetricsDialog
            title: "Save metrics to a csv"
            selectMultiple: false
            selectFolder: false
            //            folder: shortcuts.home
            //            fileMode: Platform.FileDialog.SaveFile

            onAccepted: {
                handler.save_metrics(fileUrls[0].substring(7))
            }
        }

        onClicked: {
            // saveMetricsDialog.open()
            handler.save_metrics()
        }
    }

}


/*##^##
Designer {
    D{i:1;anchors_width:200;anchors_x:0;anchors_y:0}
}
##^##*/

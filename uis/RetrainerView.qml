import QtQuick 2.0
import QtQuick.Controls 2.13
import QtQuick.Dialogs 1.1

ApplicationWindow {
    id: main
    flags: Qt.WindowStaysOnTopHint
    objectName: "window"
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
            text: qsTr("Retrain Facial Analysis Network (FAN)")
            font.pointSize: 18
        }
    }

    Item {
        id: content
        anchors.top: header.bottom
        anchors.topMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8

        Button {
            text: qsTr("Retrain")
            anchors.bottom: parent.bottom
            anchors.horizontalCenter: parent.horizontalCenter

            onClicked: {
                handler.retrain()
            }
        }
    }
}

### Note: This currently only works on python version 3.7+

## Installation
1. `git clone https://github.com/Veldrovive/video-processor.git` or download the zip.
2. Navigate to the installation directory and run `python -m venv vProcessing` to create a virutal enviroment.
3. Run `vProcessing\Scripts\activate.bat` on Windows or `source vProcessing/bin/activate` on a Unix system.
4. Install depancies with `pip install -r requirements.txt`.


## Basic Usage:
1. Open the app by running `python vidProc.py`
2. Save your video file into some folder and put your landmark csv in the same folder. These must have the same base name for the program to work.
3. When the app is opened, it will display only tooling. To process a video, use `ctl+f` or open the `file` menu and select `Load Video File`. This will open the video file and landmarks.
4. Take any metrics you need or edit the landmarks then use `ctrl+s` to save a new csv file.

## Controls:
### Selecting:
To select a point, left click on it. In order to select a range of points, hold shift and drag over the points. When creating a metric, the order of selection decides the order to points in the metric. Points selected by dragging over them will be grouped and will act as one point at their centroid.
In order to deselct all points, left click away from all points.
### Editing:
To pick up and replace a point, right click on the point you want to move then right click on the location you wish to place the point. Use `ctrl+s` to save your edits to a csv.
### Metrics:
*Work in progress*

With the points you want in the metric selected, press `1` to create a length metric and `2` to create an area metric. Press `0` to remove metrics and `3` to evaluate metrics.
### Display:
*Work in progress*

By default, landmarks, bounding boxes, and metrics are show. In order to change this, use the hotkeys `L`, `B`, and `M` respectively. Pressing these keys will toggle whether each of these is shown.

In order to zoom in, use your scroll wheel.

Left click and drag to pan around the scene.

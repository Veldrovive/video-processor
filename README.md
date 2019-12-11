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
### Landmark Estimating:
To estimate landmarks from the opened video, to to the `Landmarks` menu and press `Process Frames` or press `Ctrl+A`. This will bring up the landmark estimating window. In order to estimate only for a few frames, input frames in the standard printer format. E.g. `1-10` will estimate for frames 1-10 while `1-10, 20, 30` will estimate frames for 1-10 as well as 20 and 30.
### Selecting:
To select a point, left click on it. In order to select a range of points, hold shift and drag over the points. When creating a metric, the order of selection decides the order to points in the metric. Points selected by dragging over them will be grouped and will act as one point at their centroid.
In order to deselct all points, left click away from all points or press escape.
### Editing:
To pick up and replace a point, right click on the point you want to move then right click on the location you wish to place the point. Use `ctrl+s` to save your edits to a csv.
### Metrics:
*Work in progress*

Select the points you wish to be in the metric then select the first landmark or group of landmarks again to complete the metric.
Press `Shift+Ctrl+S` to display metric analysis. This will bring up a window where metrics can be viewed.
By default, all metrics will be normalized by the distance between the eyes.

To rename or change the type of a metric, double click on the metric name in the selector.

If you change metrics with the window open and wish to update the changes, press `Recalculate Metrics`.

If you want to save a csv file with the currently viewed data (Including normalization) then press `Save Data`.

If you want to remove a metric, select it and press `Delete Selected Metrics`.
### Display:
By default, landmarks and metrics are shown. In order to change this, use the hotkeys `Ctrl+L`, `Ctrl+B`, and `Ctrl+M` respectively. Pressing these keys will toggle whether each of these is shown.

In order to zoom in, use your scroll wheel.

Left click and drag without holding shift to pan around the scene.

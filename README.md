### Note: This currently only works on python version 3.7+

## Installation
1. `git clone https://github.com/Veldrovive/video-processor.git` or download the zip.
2. Navigate to the installation directory and run `python -m venv vProcessing` to create a virutal enviroment.
3. Run `vProcessing\Scripts\activate.bat` on Windows or `source vProcessing/bin/activate` on a Unix system.
4. Install depancies with `pip install -r requirements.txt`.


## Basic Usage:
1. Save your video file into some folder and put your landmark csv in the same folder. These must have the same base name for the program to work.
2. When the app is opened, it will display only tooling. To process a video, use `ctl+f` or open the `file` menu and select `Load Video File`. This will open the video file and landmarks.
3. Once you do the editing you want, press `ctl+s` or open the file menu and select `Save Landmark Edits` to save a new csv.

Playback Speed: This can be changed by using the config menu that pops up with the app.
Zooming: Use your scroll wheel to zoom in and out from the video.
Editing Landmarks: Right click on a landmark to pick it up and left click to place it back.
Groups: Not implemented
Metrics: Not implemented

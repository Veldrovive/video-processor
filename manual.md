# F.A.M.E. (Facial Analysis and Metric Estimation)
FAME (Facial Analysis and Metrics Estimation) brings the power of modern machine learning models and algorithms to the clinic by providing an easy to use software tool for objective facial analysis. FAME allows you to find facial landmarks automatically in videos, manually adjust those landmarks to improve accuracy, and perform different analysis related to facial movement and symmetry automatically. FAME has the ability to learn from you, so the localization of facial landmarks gets better the more you use it.

# Installaton
The current version of this software is not ready for distribution. You will need to use the Python code directly (we are sorry about this). We will try to make this process as painless as possible.

You need to be connected to the internet to install the software. Here are detailed steps for Windows and Mac computers:
### Mac:
Requirements: **Python 3.7+** and **git**
1. Download this file: https://bit.ly/2ESEdID.
2. Create a new folder called FAME and put the file in that folder.
3. Open the terminal and run the file.

### Windows:
Work in progress

# Terminology:
**Project**: Within FAME, a project refers to a collection of facial videos, machine learning models, and different settings that you adjust within the software. All of these elements are stored in one folder 

**KeyFrame**: Videos are composed of frames, and in FAME, a KeyFrame is a frame chosen from the video. KeyFrame usually represent important moments in the video, for instance, the moment of maximum mouth opening, or the moment of maximum eye closure. KeyFrames can be used to navigate the video quickly, and are usually the frames most likely to receive manual correction of landmarks.

**Landmark**: In FAME, a landmark is represented with a red dot. Facial landmarks are automatically placed by the software, but they can be easily adjusted using your mouse pointer.

**Metric**: FAME includes two types of metrics, distance and area metrics.
* A distance metric represents the distance between two or more landmarks
* An area metric calculates the area enclosed by the sequence of landmarks.

Metrics are computed for each frame in the video for which landmarks are available. FAME provides an easy to navigate graphical view of the different metrics. 
FAME comes with some predefined metrics, but you are free to create your own. 

**Retrain**: Improve the automatic localization of facial landmarks by teaching FAME to work better with you data. 

# Hotkey Reference
On mac, &#8984; means command while, on windows, it means control.

| Mac Hotkey | Windows Hotkey | Command |
|--------|---------|---------|
|<b>Projects<b/>|||
|&#8984;+N|Ctrl+N|Create a new project|
|&#8984;+F|Ctrl+F|Open an existing project|
|&#8984;+E|Ctrl+E|Edit the current project|
|<b>Video<b/>|||
|&#8984;+K|Ctrl+K|Place/Remove Keyframe|
|&#8984;+P|Ctrl+P|Take a screenshot of the current video|
|&#8984;+B|Ctrl+B|Show/Hide Bounding Boxes|
|&#8984;+L|Ctrl+L|Show/Hide Landmarks|
|&#8984;+M|Ctrl+M|Show/Hide Metrics|
|<b>Playback<b/>|||
|Space|Space|Play/Pause the video|
|Right Arrow|Right Arrow|Step forward by one frame|
|Left Arrow|Left Arrow|Step back by one frame|
|Shift+Right Arrow|Shift+Right Arrow|Seek to next keyframe|
|Shift+Left Arrow|Shift+Left Arrow|Seek to previous keyframe|
|&#8984;+Shift+Right Arrow|Ctrl+Shift+Right Arrow|Move to next video|
|&#8984;+Shift+Left Arrow|Ctrl+Shift+Left Arrow|Move to previous video|
|<b>Analysis<b/>|||
|&#8984;+A|Ctrl+A|Analyze Metrics|
|&#8984;+D|Ctrl+D|Find Landmarks|
|&#8984;+R|Ctrl+R|Retrain Analysis Network|

# Uses
FAME has three main three main usages: 

**Video Player**: FAME can open most video formats and allows you to visualize your video frame by frame or with adjustable playback speed.

**Metric Estimation**: FAME can be used to localize facial landmarks in all or some video frames, and estimate facial metrics based on those landmarks.

**Model Retraining**: FAME can be used to retrain machine learning models for facial landmarks estimation based on manual adjustments. For this, FAME allows you to select a few KeyFrames from you video(s), localize landmarks on those frames, perform manual adjustments, and use the adjusted landmarks position to improve the model accuracy. Once the model is retrained, it can be used to find landmarks in the remaining video frames.

# Workflow
FAME can be used to process one or many videos, using more videos of multiple subjects performing different tasks for model retraining will likely generate better models. That said, you can start retraining your model with a single video and grow from there. FAME results will keep improving as you add more data. 

Graphically, FAME workflow can be summarized as shown in Figure 1. In this workflow, you can go back and forth between most steps to modify the final product - a retrained model for facial landmark localization-. That is, if you are not satisfied with the final model, you can always add more videos, KeyFrames, and manual adjustments to improve the model.

| ![](https://i.imgur.com/mr16Cb2.png) |
|:--:|
| *Figure 1: FAME Workflow* |

Next we will describe in detail how to use FAME and perform some basic and advance operations. 

### Project Creation:
A [**project**](#Terminology) represents a way to organize all videos, settings, and models into one central place. These projects exist as a folder on the file system. Once the project folder is created, the user should refrain from modifying it in any way, as any change in the project folder may produce errors in the operation. Note that projects are stored as folders so that users can easily access the different videos and models to share with others. 

To create a project, launch FAME and use the hotkey `Ctrl+N` or select `File > New Project`. A new window will open to add a name and files to the project, see Figure 2. A folder will be created inside the chosen directory with the project’s name.

Give the project an appropriate name and then click on `Add Files`. Search in your computer for video files that you want to add to the project. You can add as many videos as you want. If a video file already has landmarks and you want to include them in the project, please add them by clicking `Select Landmarks`. A pop-up window will allow you to select the appropriate landmark file. Click “Finish” to start analyzing the videos.

| ![](https://i.imgur.com/4v8fMwQ.png) |
|:-:|
| *Figure 2. Project Creation Window used to add video and landmarks files to the project.* |

One you click on “Finish”, the project creation window will be closed, and you will be able to visualize the video files and landmarks – if available – in the video viewer window. The video viewer video also allows you to manually adjust the landmarks in each frame.

### Playback Controls:
Playback controls can be used to navigate the videos. These controls will appear below the video viewer. 

![](https://i.imgur.com/z8KfROK.png)

1. **Move to Previous Video**: (Hotkey - &#8984;+Shift+Left Arrow Key): Move to the previous video in the project.
2. **Seek Backward** (Hotkey - Left Arrow): Moves backward through the video one frame at a time.
3. **Play/Pause** (Hotkey - Space): Pauses and resumes video playback.
4. **Seek Forward** (Hotkey - Right Arrow): Moves forward through the video one frame at a time.
5. **Move to Next Video**: (Hotkey - &#8984;+Shift+Right Arrow Key): Move to the next video in the project.

**KeyFrames**:
KeyFrames are special frames within the video that are used for model retraining. 
The user must indicate which frames are KeyFrames using special HotKeys:
* **Add Keyframe** (Hotkey - &#8984;+K): Select the current frame as a KeyFrame, a tick mark will appear on the scroll bar.
* **Remove Keyframe** (Hotkey - &#8984;+K): If the current frame is already a keyframe, it will be removed.
* **Jump Forward between KeyFrames** (Hotkey - Shift+ Right Arrow Key): Move to the next KeyFrame.
* **Jump Backwards between KeyFrames** (Hotkeys - Shift+Left Arrow Key): Move to the previous KeyFrame.

**Interacting with frames**:
You can use your mouse scroll wheel to ZOOM in or out. While zoomed, you can PAN around the frame by pressing and dragging with left click.

### Estimate Landmarks:
After the project is created and one or mode videos are selected, is possible to find landmarks automatically in all video frames or in the previously selected KeyFrames. To access the Find Landmarks Window – see Figure 3 - use the hotkey `Ctrl+D` or select `Analysis > Estimate Landmarks`.

| ![](https://i.imgur.com/0e1ZLUI.png) |
|:-:|
| *Figure 3. Find Landmarks Window used to define which video and frames should be processed by the machine learning algorithm that finds facial landmarks* |

In the Estimate Landmarks Window you can easily choose what videos and what frames you want to find landmark. You can choose multiple frames in as many videos as you want. The facial landmark detection algorithm will work on each frame sequentially.

To select a video, click on its name.

Once a video is selected, you can choose to detect landmark in all the video by checking the `Full Video` box. You can also type a specific frame, frames (separated by coma), or a range or frames (using dash) in the `Choose a range` text box. 
If the video has KeyFrames, they will be listed in here. Each KeyFrame can be individually selected, or you can select of de-select all KeyFrames using the `Toggle All KeyFrames` button. Clicking on the KeyFrame frame number will show that video frame in the video viewer window.


After selecting the desired frames to analyze in the videos click on `Run Detection` and FAME will automatically analyze the desired frames. This process can take a few minutes depending on the number of frames. 

So far you have you have only used the default machine learning model for landmarks localization that comes pre-loaded with FAME. You can easily use alternative models to find landmarks, including the models that you improved with your own data. FAME makes it easy to change the model, click on the `Settings` tab - see figure 4 - to select models already inside your project. You can choose different models – models trained with different data sets - inside the project or different versions or checkpoints – models trained with the same dataset but using different parameters - of the same model. 

| ![](https://i.imgur.com/dhDB5Ec.png) |
|:-:|
| *Figure 4: Settings tab in the Estimate Landmarks Window* |

### Interacting with Landmarks:
Once you estimate the landmarks for a frame, those landmarks will be visible in the video viewer window as red dots. You can easily manipulate those dots to adjust the landmarks to a better position. To change the position of a landmark, simply right click on it, the landmarks will disappear from the screen. Use your right click one more time to reposition the landmark. Note that 1) once a position is modified it will be saved and you won’t be able to take the landmark back to its previous position, and 2) if a landmark is modified, the frame will be added to the list of frames used for model retraining. 

To create a metric, you select a sequence of landmarks and then select the first one again. To select a landmark, left click on it. It will become yellow to alert you that it has been selected. Subsequent landmarks selected will create a landmark squence that can be used to define a metric.

Use Shift+Left click drag to select multiple landmarks at the same time. Doing this will select the point at their average position.

To deselect all landmarks, press escape or left click away from any landmark. To deselect a single landmark left click on that landmark a second time.

### Retraining:
After correcting the positions of landmarks, use the hotkey `&#8984;+R` or select `Analysis > Retrain`. This will pop up a window that allows you to name and train a new model.

| ![](https://i.imgur.com/0PT4B2e.png) |
|:-:|
| *Figure 5: Retraining Window* |

In the retraining window, you can choose which frames will be used for retraining. To select a video, click on its name in the left column. To view a frame, click on the frame number in the right column. By default, all frames that were edited are used for retraining. To tell FAME not to use a specific frame for retraining, click `Delete` next to the frame number (The changes you made to the landmarks will remain, it will just not be used for training).

A model must have a name to be saved under. If there is already a model with the same name, the new model will be saved as the next version of the existing model. 

If a previous training was interupted, click the `Load Checkpoint` button and select the latest checkpoint to resume the training.

Click `Retrain` to start retraining the model. This process can take a few minutes up to a few hours depending on the number of frames used for retraining. If your computer is `cuda` enabled, hardware acceleration will automatically be used.

### Metric Analysis:
Once landmarks have been found, open the metric analysis window by using the hotkey `&#8984;+A` or by selecting `Analysis > Analyze Metrics`. This popup allows you to view and export your data. You can export the landmarks as a csv or you can export the value of your metrics on each frame as a csv.

| ![](https://i.imgur.com/kqX1tss.png) |
|:-:|
| *Figure 6: Metric Analysis Window* |

The analysis window is made up of five main components:

**Top Bar**: This component is used to select which video is being analyzed. It scrolls horizontally through every video in the project. Click on a video name to view its metrics.

**Normalization**: If normalization is set, it divides the metric value by the normalization value for all frames. `Normalize On Metric` automatically sets the normalization value to the average value of the metric over the whole video.

**Metric Chooser**: You can check or uncheck individual metrics to show or hide them in the graph below. 

In order to set the name or type of metric, double click on the metric name. You can then set the name, metric type (Distance or Area), or delete it.

**Graph**: This component allows you to visualize and validate your data before exporting it. If you click on a frame without a tool selected, FAME will seek to that frame in the video.

Tools from left to right:
* Home: Return to the default view where the entire graph can be seen.
* Undo: Undos the last action.
* Redo: Redoes the last undid action.
* Pan: Left click and drag to change your view of the graph.
* Zoom: Drag a box over the area you want to zoom in on.

**Save Button**: Pops up a menu to choose where to save the exported metrics. Only metrics currently shown on the graph will be exported.
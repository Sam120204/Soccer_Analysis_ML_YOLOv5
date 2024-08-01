#### Project Overview

This project involves the development of a machine learning model and associated utilities for analyzing soccer videos. The primary tasks include detecting and tracking players, referees, and the ball, as well as assigning ball control to teams and individual players. The following key functionalities are implemented:

- Object detection using YOLO (You Only Look Once) model
- Object tracking with ByteTrack
- Ball and player position interpolation
- Annotation and drawing of bounding boxes and ellipses around detected objects
- Calculation of ball possession by teams



### Dataset

The dataset used for this project can be found on Kaggle at the following link: [DFL Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data?select=clips). 



#### Prerequisites

- Python 3.7+
- OpenCV
- YOLO (Ultralytics)
- Supervision (for ByteTrack)
- OpenCV
- Pandas
- Numpy

You can install the necessary Python packages using `pip`:

```sh
pip install -r requirements.txt
```

#### Usage

1. **Model Training**:
   - The `football_training_yolo_v5.ipynb` notebook contains the code for training the YOLO model on a custom dataset.

2. **Object Detection and Tracking**:
   - The `yolo_inference.py` script runs YOLO detection on video frames.
   - The `tracker.py` script handles object tracking and interpolation of ball positions.
   
3. **Ball and Player Assignment**:
   - The `player_ball_assigner.py` script assigns the ball to the closest player.
   - The `team_assigner.py` script assigns players to their respective teams based on their jersey colors.

4. **Utilities**:
   - The `bbox_utils.py` script contains utility functions for bounding box calculations.
   - The `video_utils.py` script contains utility functions for video processing.

5. **Main Execution**:
   - The `main.py` script is the entry point for the project. It reads video frames, runs detection and tracking, assigns ball control, and draws annotations on the video.

#### Running the Project

To run the project, follow these steps:

1. **Place Video File**:
   - Place your input video file in the `videos/` directory. For example, `sample_video.avi`.

2. **Execute the Main Script**:
   - Run the `main.py` script:

```sh
python main.py
```

3. **Output**:
   - The annotated video frames will be saved or displayed as specified in the script.

#### Detailed Function Descriptions

- `main.py`: Main script to orchestrate the detection, tracking, and annotation processes.
- `yolo_inference.py`: Contains functions to perform inference using the YOLO model.
- `player_ball_assigner.py`: Assigns the ball to the nearest player based on bounding box positions.
- `team_assigner.py`: Assigns players to teams using color-based clustering.
- `tracker.py`: Implements the tracking and interpolation logic.
- `bbox_utils.py`: Provides utility functions for handling bounding boxes.
- `video_utils.py`: Contains functions for reading and processing video frames.
- `football_training_yolo_v5.ipynb`: Jupyter notebook for training the YOLO model.


# Task 2 
# Football Player Tracking and Mapping (YOLOv5 + Deep SORT)

This project uses a custom-trained YOLOv5 model and Deep SORT to track players, ball, goalkeeper, and referees in football match videos. It consists of two main parts:

## Cell 1: Detection and Tracking on Broadcast Video

- The code reads the video `broadcast.mp4`.
- It loads a YOLOv5 model (`best.pt`) trained on four classes: player, ball, goalkeeper, and referee.
- Each frame is processed to:
  - Detect all objects of interest
  - Track players using Deep SORT to assign consistent IDs
  - Calculate the ball’s velocity using position difference between frames
    - If the ball is missing in a frame, its position is estimated using previous velocity
- The following data is stored for each frame:
  - Ball position and velocity
  - Positions of up to 3 referees
  - Position of the goalkeeper
  - Positions of 22 players
- Outputs:
  - `output.mp4`: video with detection and tracking annotations
  - `tracking_output.csv`: CSV file storing all positions and ball velocity per frame

## Cell 2: Player Mapping from Broadcast View to Tacticam View

- The code reads a second video `tacticam.mp4`, and also loads `tracking_output.csv` from Cell 1.
- It detects players and the ball in each frame of `tacticam.mp4`.
- The goal is to assign the same global player IDs (from broadcast view) to players detected in the tacticam view.

### How the Mapping Works:

- For each frame, the player positions from both views are extracted.
- The ball is used as a reference point to align the coordinate systems.
- Using matching players from both views, a **rotation matrix**, **translation vector**, and **scale factor** are calculated using the **Kabsch algorithm**:
  - First, a rigid alignment is computed between the two sets of points (broadcast and tacticam).
  - This gives the best-fit rotation (R), scale (s), and translation (t) to transform one set of points to the other.
- The transformation is then applied to map player positions in the tacticam view to the broadcast view.
- Player IDs are assigned based on nearest matches after alignment.

### Output:

- `tacticam_matched.mp4`: video with bounding boxes and matched player IDs from the broadcast view
- `matched_output.csv`: frame-by-frame player positions with global ID alignment

#### Scope for improvements :- 
- The model can further be trained to detect the flags on the sidelines, which are a stationay object, and will improve the calculation for the rotation matrix, incase the ball is not detected for a long time. 
- The code can be further improved to include the scaling factor. Using this and the task 1 code, we can effectively map all the players, whether inside or outside the field, and the data for velocity, distance covered can be collected. 

## Requirements

```bash
pip install ultralytics opencv-python deep_sort_realtime scipy numpy

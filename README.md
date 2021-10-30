# fishTrackPY

Python / OpenCV tool for detecting and tracking fish using video data.

## Runing the code
The code can be run as by executing
<code>
sh runTracking.sh
</code>

Alternatively, one can call 'runTracking.py' with input arguments to the path of the video (required) and camera calibration (optional).

Output are
* videoname-detect.npz -- locations (pixel x, pixel y) for each individual for all frames (int) of the video
* videoname-track.npz -- tracks (pixel x, pixel y) for individuals
* videoname-vtracks.npz -- trajectory data (position, velocity, acceleration) for individuals.  More details on vtracks file below.
* videoname-track.mp4 -- video vizualization of trajectory data, path of individuals is overlay on raw video

Control of output can be adjusted by commenting out lines in **runTracking.py** file.

<img src="https://github.com/puckettlab/fishTrackPY/blob/master/figures/sampleVideo-track.gif" width="250" />



## Description

Using the raw video data, the code will do background subtraction, locate, track and calculate the trajectories of individual fish.
If camera calibration data is provided, the code will convert the raw image data and project into the world frame.
Velocity and acceleration data is calculated and tracks are recorded with variables: ind, t, x, v, a.

**vtracks format:**
* ind = index of individual
* t = time = frame / frame_rate, where frame_rate is extracted from the video
* x = position [x,y,z]
* v = velocity [vx,vy,vz]
* a = acceleration [ax,ay,az]


Provided a frame as shown below on the left.  Traditional object detection will not be able to separate fish which swim close together.
These occulusions occur frequently, which makes for poor tracking of individuals.
This code uses a combination of traditional techniques along with a novel method to separate individuals to improve tracking.
fishTrackPY uses line segment detection, group line segments, and is able to improve detection significantly.
With typical lighting and a dense population of fish, the code is able to track 98% of individuals.

<p float="left">
  <img src="https://github.com/puckettlab/fishTrackPY/blob/master/figures/sup-figure10detect01.png" width="250" />
  <img src="https://github.com/puckettlab/fishTrackPY/blob/master/figures/sup-figure10detect02-2.png" width="250" />
  <img src="https://github.com/puckettlab/fishTrackPY/blob/master/figures/sup-figure10detect03-2.png" width="250" />
</p>

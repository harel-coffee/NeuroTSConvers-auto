<!--- ## Introduction --->
## Introduction
This repository represent a system of brain activity prediction using behavioral features.
The behavioral features represent verbal and non-verbal variables extracted during a fMRI experience of human-human and human-robot conversations conducted on several subjects.
The aim is to detect the behavioral features that are responsible for the activation of each brain area, by means of prediction.
A feature selection step is performed to select the input variables for the prediction of brain activity, then the most relevant input features are those how lead to the best prediction score.

## Extracted features
* Speech and text features:  Signal envelop, Speech activity, Overlap, Filled-breaks, Feedbacks, Discourse markers, Particles items, Laughters, Lexical richness, Polarity, and Subjectivity.
* Video and eyetracking features: Facial Action Units, Landmarks, Head Pose coordinates, saccades, gaze and speed coordinates, and variables categorizing where the subject is looking in at each time step (face, eyes, mouth).


## Dependencies
  * python>=3.6
  * Openface is required (https://github.com/TadasBaltrusaitis/OpenFace) to compute facial features from videos. 
  * Install dependencies for python:
    ```bash
    pip install -r requirements.txt
    ```


## Demo
  * To run a demo, we need a video file (of the interlocutor), and the audios of both the participant and the interlocutor, and an eyetracking file of the participant.

  * A example is provided in the folder "demo". To run the example:
  ```bash
  python demo/run.py -g -t h -rg 5 6 7 8 9 -ofp "path_to_openface"

  -g : to generate behavioral features from raw signals
  -t : type on interaction. -h for human-human and -r for human-robot.
  -rg: codes of brain areas to predict (see brain_areas.csv).
  -ofp: path where OpenFace is installed.
  ```
  * To visualize the predictions of brain activity:

	```bash
	python demo/animation.py
	```



<!---
## Data
The data are recorded during an experience of human-human and human-robot conversations
conducted on more than twenty participants, and divided into a set of
four sessions, where each one contains six conversations of sixty seconds each.
The conversations are performed under the same experimental conditions, but
alternatively with a human and a robot. The conversations are in a form of face
to face talk about images, in the same time, brain activity is obtained by fMRI,
as well as eye movement of the participant, the audio files of both the agent
and the participants, and the videos of the agent.

## Time series extraction
### Processing neuro-physiological signals
The brain activity is obtained by fMRI, where the
BOLD signal is measured in different brain area during conversations. Each
conversation spans one minute, and the observations are spaced 1.205 seconds
apart. As a consequence, for each subject, the activities of 277 areas are recorded
by averaging voxels activity in each area.

### Processing audios
*  First, speech to text is performed then transcriptions are generated from audio files and the associated text.
*  We extract time series from the transcriptions,  represented by the following features:
Speech signal, Speech activity, Overlap, filled breaks, feedbacks, Discourse markers, particles items, laughters, lexical richness, polarity, and subjectivity.

### Processing videos
* We use OpenFace to extract the following features:
 * Facial Action Units
 * Landmarks
 * Head Pose Estimation
* Finally, we construct time series using these features  by analyzing each image of the videos.

### Processing eyetracking data
* An eyetracking system is used to record the gaze coordinates. We process the data and compute the gradient,
then we project the coordinates on visual stimulation to record where the subject is looking in at each time step (face, eyes, mouth, else).
--->

<!--- ## Introduction --->
## Introduction
This repository represent a system of brain activity prediction using behavioral features.
The behavioral features represent verbal and non-verbal variables extracted during a fMRI experience of human-human and human-robot conversations conducted on several subjects.
The aim is to detect the behavioral features that are responsible for the activation of each brain area, by means of prediction.
A feature selection step is performed to select the input variables for the prediction of brain activity, then the most relevant input features are those how lead to the best prediction score.

## Extracted features
* Speech and text features:  Signal envelop, Speech activity, Overlap, Filled-breaks, Feedbacks, Discourse markers, Particles items, Laughters, Lexical richness, Polarity, and Subjectivity.
* Video and eyetracking features: Facial Action Units, Landmarks, Head Pose coordinates, saccades, gaze and speed coordinates, and variables categorizing where the subject is looking in at each time step (face, eyes, mouth).


## Requirements
  * Python>=3.6
  * Openface  (https://github.com/TadasBaltrusaitis/OpenFace) is required to compute facial features from videos.
  * SPPAS (http://www.sppas.org/) is required for automatic annotation and segmentation of the speech (a copy is included in the code source of the prediction module).
  * Python packages:
    ```bash
      pip install -r requirements.txt
    ```
  * Spacy  models:
    ```bash
      python -m spacy download fr_core_news_sm
      python -m spacy download en_core_web_sm
    ```

## Demo
  * To run a demo, we need a video file (of the interlocutor), and the audios of both the participant and the interlocutor, and an eyetracking file of the participant.

  * A example is provided in the folder "demo". To run the example:

  ```bash
  # Generate time series from input data
  python Demo/generate_time_series.py -rg 1 2 3 4 5 6 -pmp .
  -in Demo -ofp "path to Openface"

  # Compute the predictions
  python Demo/predict.py -rg 1 2 3 4 5 6 -pmp . -in Demo -t r

  # Time series animation using the obtained predictions
  python Demo/animation.py -in Demo

  # Visualize the predictions in the brain
  python Demo/visualization.py -in Demo
  ```

  * The used arguments are as follows:
  ```bash
  -t : type on interaction. -h for human-human and -r for human-robot.
  -rg: codes of brain areas to predict (see brain_areas.csv).
  -ofp: path where OpenFace is installed.
  -in: Working directory (Demo in this case)
  -pmp: Prediction module path (the current directory)
  ```

<!--- ## Introduction --->
## NeuroTSConvers: neuro-physiological time series prediction in conversations
This repository represent a system for multimodal brain activity prediction using behavioral features. The behavioral features represent verbal and non-verbal variables extracted during a fMRI experience of human-human and human-robot conversations conducted on several subjects.
The aim is to detect the behavioral features that are responsible for the activation of each brain area, by means of prediction. The system can also be used to predict the brain activity of a given natural bidirectional conversation.  

__This project is still under development.__


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
  * To run a demo, we need data of a conversation between an interlocutor (human or robot), and a subject (human, on which we try to predict fMRI responses). The data consist of the video of the interlocutor, and the audios of both the participant and the interlocutor (with transcriptions), and an eyetracking file of the subject

  * A first step consists in generating features from raw data (see behavioral_features.tsv for the description). The obtained features are themselves an interesting output of this process, which can be used by the user for other purposes.
  Then, pre-trained classification models can be used to predict brain activity from the extracted features.

  * A example is provided in the folder "demo". To run the example:

  ```bash
  # Generate time series from input data
  python demo/generate_time_series.py -rg 3 4 7 8 9 10 21 -pmp .
  -in Demo -ofp "path to Openface"

  # Compute the predictions
  python demo/predict.py -rg 3 4 7 8 9 10 21 -pmp . -in demo -t r

  # Time series animation using the obtained predictions
  python demo/animation.py -in demo

  # Visualize the predictions in the brain
  python demo/visualization.py -in demo
  ```

  * Description of the used arguments:

    * -t : type on interaction. -h for human-human and -r for human-robot.
    * -rg: codes of brain areas to predict (see brain_areas.csv).
    * -ofp: path to OpenFace.
    * -in: working directory (demo in this case)
    * -pmp: Prediction module path (the current directory)

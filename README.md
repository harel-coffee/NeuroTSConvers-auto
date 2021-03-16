<!--- ## Introduction --->
## NeuroTSConvers: neuro-physiological time series prediction in conversations
This repository represent a system for multimodal brain activity prediction using behavioral features. The behavioral features represent verbal and non-verbal variables extracted during a fMRI experience of human-human and human-robot conversations conducted on several subjects.
The aim is to detect the behavioral features that are responsible for the activation of each brain area, by means of prediction. The system can also be used to predict the brain activity of a given natural bidirectional conversation.  

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
  * Boost: optional, to run local neural networks models.


## Finding best models / reproducing existing results
  * This is to find the best parameter of the used classifiers for each brain area, and the best set of predictive features via feature selection. The results of this step are stored in the folder results. A k-fold-cross validation is performed, then evaluation are made on a test set (25% of the data, or 6 participants from 24) to evaluate the models based on their Fscores.

  * Available feature selection methods: K_MEDOIDS, MI_RANK, Model_RANK.
  * Available classifiers: RF (Random Forrest), SVM, LREG (Logistic Regression), MLP (Model based on a Multi Layer Perceptron), LSTM (Model based on the Long Short Term Memory network). MLP and LSTM architectures are based on Keras library.

  * Another code for MLP  and LSTM is provided using a C++ library (src/prediction/network_export.so) exposed to python using Boost. See the script src/prediction/predict_local_ann.py for more details.

  * Example, let's make evaluations on two brain areas using K_MEDOIDS as feature selection and the Random Forrest as classifier:
    ```bash
     python src/find_models.py -rg 3 4 -p 6 -all -mthd K_MEDOIDS -m RF
    ```

## Training
  * After finding the appropriate model for each brain area, we can train the models on all available data:
    ```bash
     python src/train_models.py -rg 3 4
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

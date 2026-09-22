# Models

This directory contains documentation related to the machine learning
model used for facial emotion recognition.

## Emotion Recognition Model

The facial emotion recognition system uses a custom **TinyVGG-like CNN**
implemented in PyTorch.

The model takes grayscale facial images of size **48 × 48 pixels** as input
and predicts six emotion classes:

- Angry
- Fear
- Happy
- Neutral
- Sad
- Surprise

### Architecture

The model consists of:
````
- Two convolutional blocks
- Two convolutional layers per block
- ReLU activation functions
- Max pooling
- A fully connected classification layer
````
The model is defined and trained in:

```text
notebooks/Facial_recognition_Model_1_FER2013.ipynb
````
The trained model is used for real-time inference in:
````
src/main_six_classes.py

````
## Trained Weights

The trained model weights are not included in this repository.

The real-time inference script expects a local trained weight file:
````
fer_weights_v5.pth
````
The path to the weight file must be specified locally using:
````
MODEL_PATH = "path/to/fer_weights_v5.pth"

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)
````
Replace:
````
path/to/fer_weights_v5.pth
````
with the actual local path to the trained model weights.

For example:
````
MODEL_PATH = "models/fer_weights_v5.pth"
````
if the weight file is stored locally inside the models/ directory.

## Training

The model can be trained using:
````
notebooks/Facial_recognition_Model_1_FER2013.ipynb
````
The notebook contains the dataset preparation, preprocessing,
data augmentation, model definition, training, and evaluation steps.

## Model Reproducibility

The source code and training notebook are provided to reproduce the
model training process.

The FER2013 dataset and trained model weights are not included in
this repository.

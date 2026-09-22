# Dataset

This project uses the **FER2013 (Facial Expression Recognition 2013)** dataset
for training and evaluating the facial emotion recognition model.

## Dataset source

The dataset is available through Kaggle:

- **Dataset:** FER2013
- **Source:** [Kaggle — FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

The dataset is downloaded programmatically using `kagglehub` in the training
notebook.

## Dataset structure

The dataset is organized into training and testing subsets:

```text
FER2013/
├── train/
│   ├── angry/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
````
The project uses six emotion classes:
````
Angry
Fear
Happy
Neutral
Sad
Surprise
````
## Dataset availability

The original dataset is not included in this repository.

To obtain the dataset, please download it from the original Kaggle source
and follow the applicable terms of use and licensing conditions.

The training notebook automatically downloads the dataset using:
````
import kagglehub

path = kagglehub.dataset_download("msambare/fer2013")
````
## Privacy and data policy

No personal or user-generated facial images are included in this repository.
The repository only contains the code and documentation required to reproduce
the experiments.

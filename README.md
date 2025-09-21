# HC-CNN for Microplastic Classification
An implementation of the Holographic-Classifier Convolutional Neural Network (HC-CNN) for automatic classification of microplastic pollutants from raw holographic images. This project provides a fast, lightweight, and efficient tool for environmental monitoring, eliminating the need for complex image pre-processing.

This repository provides a complete pipeline to train the HC-CNN model, evaluate its performance, and run inference on new holograms.

✨ Features
- Lightweight Model: An efficient CNN architecture perfect for real-time analysis.

- End-to-End Pipeline: Scripts for data preparation, training, and evaluation.

- Direct Inference: Classify pollution levels directly from raw hologram files.

- High Performance: Aims to replicate the high accuracy (~97%) and efficiency reported in the original paper.

File Structure:
```
hc-cnn-microplastics/
├── data/
│   └── raw/
│       └── dataset_microplastics/  # <-- The downloaded dataset goes here
├── models/                         # <-- Saved model weights (e.g., .h5 files) will be stored here
├── notebooks/                      # <-- Jupyter notebooks for experimentation and analysis
│   └── 1_data_exploration.ipynb
├── results/                        # <-- Output from scripts (e.g., confusion matrix images, logs)
│   └── confusion_matrix.png
├── src/                            # <-- Main source code
│   ├── __init__.py
│   ├── model.py                    # <-- HC-CNN architecture is defined here
│   ├── dataset.py                  # <-- Data loading, augmentation, and preprocessing logic
│   ├── train.py                    # <-- Script to run model training
│   ├── evaluate.py                 # <-- Script to evaluate the model on the test set
│   └── predict.py                  # <-- Script for making predictions on a single image
├── .gitignore                      # <-- Specifies files/folders for Git to ignore
├── LICENSE
├── README.md                       # <-- The file you are reading
└── requirements.txt                # <-- Project dependencies
```

🚀 Getting Started
Follow these instructions to set up the project and run the model on your local machine.

1. Prerequisites
- Python 3.8+
- Pip & a virtual environment tool (venv)
2. Installation
First, clone the repository to your local machine:
```
git clone [https://github.com/your-username/hc-cnn-microplastics.git](https://github.com/your-username/hc-cnn-microplastics.git)
cd hc-cnn-microplastics
```
Next, create and activate a Python virtual environment. This keeps your project dependencies isolated.
```
# For Linux/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```
Finally, install all the required packages from the requirements.txt file:
```
pip install -r requirements.txt
```
💻 Usage
This repository includes scripts for training, evaluating, and running predictions.

Training the Model
To train the HC-CNN model from scratch, run the train.py script from the root directory.
```
python src/train.py --epochs 120 --learning-rate 0.0001 --batch-size 32
```

- epochs: Number of training epochs. The optimal range is 100-130.

- learning-rate: The learning rate for the Adam optimizer.

- batch-size: Number of samples per training batch.

The script will save the best performing model weights to the models/ directory.

Evaluating the Model
Once training is complete, you can evaluate the model's performance on the test set using evaluate.py.

```
python src/evaluate.py --model-path models/hc_cnn_best.h5
```

This script will print a classification report with accuracy, precision, recall, and F1-score. It will also save a normalized confusion matrix image to the results/ folder.

Making a Prediction
To classify a single raw hologram image, use the predict.py script.
```
python src/predict.py --model-path models/hc_cnn_best.h5 --image-path /path/to/your/hologram.png
```

📊 Performance
This implementation aims to replicate the state-of-the-art results from the original research. The HC-CNN is both highly accurate and computationally efficient compared to deeper architectures like ResNet or VGG.

| Methods        | Accuracy (A)         | Precision (P)        | Recall (R)           | F1-Score ($F_1$)     | Decision Time ($T_D$, hours) |
| :------------- | :------------------- | :------------------- | :------------------- | :------------------- | :--------------------------- |
| MLP [34]       | 0.6974               | 0.5376               | 0.6370               | 0.6198               | 0.2500                       |
| VGG-16 [35]    | 0.8524               | 0.7890               | 0.7873               | 0.7737               | 1.0833                       |
| CNN            | 0.9403               | 0.9349               | 0.8897               | 0.9014               | 0.6000                       |
| ResNet [36]    | 0.9459               | 0.9518               | 0.8863               | 0.9049               | 1.0183                       |
| **HC-CNN** | **0.9701** | **0.9761** | **0.9595** | **0.9520** | **0.3833** |


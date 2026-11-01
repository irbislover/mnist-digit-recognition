# MNIST Digit Recognition

A PyTorch project for recognizing handwritten digits from 0 to 9.

In my taxi fare project, I used a model with just a weight and a bias. Here I use a neural network with a hidden layer and train it on images.

## The model

Each image is 28 × 28 pixels, flattened into 784 values:

```text
Linear(784, 128) → ReLU → Linear(128, 10)
```

The network outputs a score for each digit. The highest score gives the prediction.

I train on 60,000 MNIST images for 5 epochs, using batches of 64, Adam with a learning rate of 0.001, and `CrossEntropyLoss`. The training data is shuffled each epoch.

The included weights achieved **97.39% accuracy** on 10,000 test images: 9,739 correct predictions and 261 mistakes. Results may vary after retraining.

## Running the project

I used Python 3.12. From the project folder, create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux and macOS:

```bash
source .venv/bin/activate
```

Install the dependencies and test the included weights:

```bash
python -m pip install -r requirements.txt
python test.py
```

MNIST downloads to `data/` on the first run, which needs an internet connection. Testing runs on the CPU.

To train your own model:

```bash
python train.py
```

Training uses CUDA when available, otherwise the CPU. It prints loss and accuracy after each epoch, saves the weights, then evaluates the model on the test set. Each run starts from scratch and overwrites `mnist_weights.pth`.

## Project files

- `model.py` defines the network.
- `train.py` handles training and final evaluation.
- `test.py` evaluates the saved weights.
- `mnist_weights.pth` contains the trained weights.
- `requirements.txt` lists the dependencies.

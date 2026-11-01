# MNIST Digit Recognition

A PyTorch project for recognizing handwritten digits from 0 to 9.

In my taxi fare project, I used a model with just a weight and a bias. Here I use a neural network with a hidden layer and train it on images.

## The model

Each MNIST image is 28 × 28 pixels. I flatten it into 784 values before passing it to the network:

```text
Linear(784, 128) → ReLU → Linear(128, 10)
```

The output contains 10 scores, one for each digit. The highest score gives the prediction.

## Training

I train on 60,000 images in batches of 64, with the data shuffled each epoch. The model uses Adam with a learning rate of 0.001 and `CrossEntropyLoss`.

Training takes 5 epochs. For each batch, the script makes predictions, calculates the loss, clears the old gradients, and updates the weights after `loss.backward()`.

`train.py` prints loss and accuracy after each epoch. It then saves the weights to `mnist_weights.pth` and checks the model on 10,000 test images. Training uses CUDA if it's available, or the CPU otherwise.

## Results

The saved model reached 97.39% test accuracy: 9,739 correct predictions and 261 mistakes.

I keep the test in a separate script too. `test.py` loads the weights and runs on the CPU, so I can check the model again without training it. A new training run may give a different result.

## Project files

- `model.py` defines the network.
- `train.py` trains it and saves the weights.
- `test.py` checks the saved model.
- `requirements.txt` contains the dependencies.

The MNIST data and weights are excluded from Git. Both are created locally when you run training.

## Running the project

I used Python 3.12. From the project folder, create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies and train:

```bash
python -m pip install -r requirements.txt
python train.py
```

MNIST downloads to `data/` on the first run. You'll need an internet connection for that.

To test the saved weights later:

```bash
python test.py
```

Run training first: `test.py` needs `mnist_weights.pth`. Running `train.py` again starts from scratch and overwrites that file.

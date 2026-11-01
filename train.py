from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNISTModel


def main():
    base_dir = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=str(base_dir / "data"),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(base_dir / "data"),
        train=False,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = MNISTModel().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device).flatten(start_dim=1)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (output.argmax(dim=1) == labels).sum().item()

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Loss: {total_loss / len(train_dataset):.4f}")
        print(f"Accuracy: {correct / len(train_dataset):.2%}")
        print("-" * 30)

    torch.save(model.state_dict(), base_dir / "mnist_weights.pth")
    model.eval()
    test_correct = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device).flatten(start_dim=1)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            test_correct += (predictions == labels).sum().item()

    total = len(test_dataset)
    print("\nTEST RESULTS")
    print(f"Test Accuracy: {test_correct / total:.2%}")
    print(f"Correct: {test_correct}")
    print(f"Wrong: {total - test_correct}")
    print(f"Total: {total}")


if __name__ == "__main__":
    main()

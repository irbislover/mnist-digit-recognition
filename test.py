from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNISTModel


def main():
    base_dir = Path(__file__).resolve().parent

    model = MNISTModel()

    weights = torch.load(
        base_dir / "mnist_weights.pth",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(weights)
    model.eval()

    print("Веса загружены. Проверяем модель...", flush=True)

    test_dataset = datasets.MNIST(
        root=str(base_dir / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    test_loader = DataLoader(test_dataset, batch_size=64)

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.flatten(start_dim=1)
            predictions = model(images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"Точность: {correct / total:.2%}")
    print(f"Правильно: {correct}")
    print(f"Ошибок: {total - correct}")
    print(f"Всего: {total}")


if __name__ == "__main__":
    main()

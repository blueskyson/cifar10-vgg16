import os
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt

BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCH = 20
FILENAME = "vgg16.pth"
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

# load CIFAR10. Take reference from
# https://clay-atlas.com/blog/2019/10/20/pytorch-chinese-tutorial-classifier-cifar-10/
def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = datasets.CIFAR10(
        root=".", train=True, download=True, transform=transform
    )

    testset = datasets.CIFAR10(
        root=".", train=False, download=True, transform=transform
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainset, testset, classes


# Create a vgg16 model
def get_model():
    model = models.vgg16()
    # only use 10 features
    model.classifier._modules["6"] = nn.Linear(
        in_features=4096, out_features=10, bias=True
    )
    return model

def train(trainloader, model, criterion, optimizer):
    model.train()
    losses = 0.0
    corrects = 0
    total = 0
    
    for inputs, targets in tqdm(trainloader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()
        _, predicted = outputs.max(1)
        corrects += predicted.eq(targets).sum().item()
        total += targets.size(0)
    
    train_acc = corrects * 100.0 / total  # percentage
    train_loss = losses / len(trainloader)
    return train_acc, train_loss

def validate(validateloader, model, criterion):
    model.eval()
    losses = 0.0
    corrects = 0
    total = 0

    for inputs, targets in tqdm(validateloader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        losses += loss.item()
        _, predicted = outputs.max(1)
        corrects += predicted.eq(targets).sum().item()
        total += targets.size(0)
    
    validate_acc = corrects * 100.0 / total  # percentage
    validate_loss = losses / len(validateloader)
    return validate_acc, validate_loss

def test(index, dataset, classes):
    model = get_model()
    state_dict = torch.load(
        os.path.join(WORKING_DIR, FILENAME)
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Load selected image
    img = dataset[index][0]
    norm_img = img / 2 + 0.5
    norm_img = norm_img.numpy()
    plt.figure()
    plt.imshow(np.transpose(norm_img, (1, 2, 0)))
    plt.show()

    # Generate predict chart
    output = model(img.unsqueeze(0))
    softmax = torch.nn.Softmax()
    probabilities = softmax(output)

    plt.figure()
    plt.bar(np.asarray(classes), probabilities.detach().numpy()[0])
    plt.show()


def main():
    trainset, testset, classes = load_data()
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    model = get_model().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)\
    
    best_acc = 0.0
    train_accs = []
    validate_accs = []
    train_losses = []
    validate_losses = []
    
    for epoch in range(EPOCH):
        print(f"\nEPOCH {epoch + 1}:")
        print("Train:")
        tr_acc, tr_loss = train(trainloader, model, criterion, optimizer)
        print("Validate:")
        va_acc, va_loss = validate(testloader, model, criterion)
        print(f"train accuarcy: {tr_acc}, train loss: {tr_loss} | "
              f"validate accuracy: {va_acc}, validate loss: {va_loss}")
        
        train_accs.append(tr_acc)
        train_losses.append(tr_loss)
        validate_accs.append(va_acc)
        validate_losses.append(va_loss)

        if (va_acc > best_acc):
            torch.save(
                model.state_dict(),
                os.path.join(WORKING_DIR, FILENAME)
            )
    
    # Save accuracy as an image
    plt.clf()
    plt.plot(train_accs, "g", label="Training Acc")
    plt.plot(validate_accs, "b", label="Validation Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("accuracy.png")

    # Save loss as an image
    plt.clf()
    plt.plot(train_losses, "g", label="Training Loss")
    plt.plot(validate_losses, "b", label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss.png")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Cuda not available, abort.")
    else:
        main()


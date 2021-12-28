import os
import numpy as np
import torch
import torchvision
from torchvision import models
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt


# load CIFAR10. Take reference from
# https://clay-atlas.com/blog/2019/10/20/pytorch-chinese-tutorial-classifier-cifar-10/
def load_data():
    transform = transforms.Compose([transforms.ToTensor(), \
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = datasets.CIFAR10( \
    root='.', train=True, download=True, transform=transform)
    
    testset = datasets.CIFAR10( \
    root='.', train=False, download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', \
    'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, testset, classes


# Create a vgg16 model
def get_model():
    model = models.vgg16()
    # only use 10 features
    model.classifier._modules['6'] = nn.Linear(in_features=4096, out_features=10, bias=True)
    return model


# train model
def train(batch_size, learning_rate, epoch):
    trainset, testset, classes = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = 32,
        shuffle = True,
        num_workers = 4
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    acc_list = []
    loss_list = []

    for i in range(epoch):
        print(f"\nEpoch: {i}")
        train_loss = 0
        correct = 0
        total = 0
        
        # train using trainset 
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # Here inputs are images and targets are labels
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            tmp = tmp + 1
        acc = correct * 100.0 / total
        loss = train_loss / batch_idx
        print(f"Loss: {loss:.2f} | Acc: {acc:.2f}%")
        acc_list.append(acc)
        loss_list.append(loss)

    pwd = os.path.dirname(os.path.abspath(__file__))
    torch.save(model.state_dict(), os.path.join(pwd,'vgg16.pth'))
    print("Save model to ./vgg16.pth")

    # Save accuracy as an image
    x = list(range(epoch))
    plt.clf()
    plt.plot(x, acc_list)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy.png')

    # Save loss as an image
    plt.clf()
    plt.plot(x, loss_list)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')


def test(index):
    trainset, testset, classes = load_data()
    model = get_model()
    state_dict = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),f'vgg16.pth'))
    model.load_state_dict(state_dict)
    model.eval()

    # Load selected image
    img = testset[index][0]
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

if __name__ == "__main__":
    train(32, 0.01, 20)
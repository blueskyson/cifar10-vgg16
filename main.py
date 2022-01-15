import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)
import vgg16train
import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("cifar10-vgg16")

        button = [None] * 5
        button[0] = QPushButton("1. show random labeled images", self)
        button[0].clicked.connect(self.show_random_labeled_images)
        button[1] = QPushButton("2. show model shortcut", self)
        button[1].clicked.connect(self.show_model_shortcut)
        button[2] = QPushButton("3. show training accuracy", self)
        button[2].clicked.connect(self.show_training_accuracy)
        button[3] = QPushButton("4. show test", self)
        button[3].clicked.connect(self.show_test)
        self.index_txt = QLineEdit("0")

        vbox = QVBoxLayout()
        vbox.addWidget(button[0])
        vbox.addWidget(button[1])
        vbox.addWidget(button[2])
        vbox.addWidget(QLabel("select test image:"))
        vbox.addWidget(self.index_txt)
        vbox.addWidget(button[3])
        vbox.addStretch(1)
        self.setLayout(vbox)
        
        print("Loading data...")
        self.trainset, self.testset, self.classes = vgg16train.load_data()
    
    # Show images. Take reference from
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar
    def show_random_labeled_images(self):
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=9, shuffle=True, num_workers=2
        )

        # get some random training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        images = torchvision.utils.make_grid(images, nrow=3)
        images = images / 2 + 0.5  # unnormalize
        npimg = images.numpy()

        # generate label
        s = ""
        i = 1
        for l in labels:
            s += str(i) + "." + str(self.classes[l]) + " "
            i += 1
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(s)
        plt.show()

    def show_model_shortcut(self):
        print(vgg16train.get_model())

    def show_training_accuracy(self):
        accimg = cv2.imread("accuracy.png")
        lossimg = cv2.imread("loss.png")
        cv2.imshow("Accuracy", accimg)
        cv2.imshow("Loss", lossimg)

    def show_test(self):
        id = int(self.index_txt.text())
        vgg16train.test(id, self.testset, self.classes)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier


def grayscaleHistogram(temp_img, bins):
    gray_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_img], [0], None, [bins], [0, 256])
    return hist.flatten()


def colorHistogram(temp_img, bins):
    hist = cv2.calcHist([temp_img], [0, 1, 2], None, (bins, bins, bins), [0, 256, 0, 256, 0, 256])
    return hist.flatten()


def gridBasedExtractiong(img):
    height = img.shape[0]
    width = img.shape[1]


    #grid level
    level = 1
    bin = pow(2, level - 1)

    m = int(height / bin)
    n = int(width / bin)

    histogram_bins = 3

    extr = []

    for r in range(0, height, m):
        for c in range(0, width, n):
            temp_img = img[r:r + m, c:c + n, :]
            hist = colorHistogram(temp_img, histogram_bins)
            extr.append(hist)

    return extr


def trainscikit(trainData, trainLabel, validationData, validationLabel, testData, testLabel):
    neigh = KNeighborsClassifier(n_neighbors=9)
    neigh.fit(trainData, trainLabel)
    print("----TOTAL ACCURACY----")
    print(neigh.score(validationData, validationLabel))
    #print("----PREDICTONS FOR {}----".format(testLabel))
    #print(neigh.predict(testData))
    #print("\n")


def main():
    trainData = []
    trainLabel = []
    validationData = []
    validationLabel = []
    testData = []
    testLabel = ""

    for root, dirs, files in os.walk('TrainingSet/'):
        for name in files:
            file = os.path.split(root)
            name = root + '/' + name
            img = cv2.imread(name)
            hist = gridBasedExtractiong(img)
            trainData.extend(hist)
            for i in hist:
                trainLabel.append(file[1])


    for root, dirs, files in os.walk('ValidationSet/'):
        for name in files:
            file = os.path.split(root)
            name = root + '/' + name
            img = cv2.imread(name)
            hist = gridBasedExtractiong(img)
            validationData.extend(hist)
            for i in hist:
                validationLabel.append(file[1])


    for root, dirs, files in os.walk('TestSet/'):
        for name in files:
            file = os.path.split(root)
            name = root + '/' + name
            img = cv2.imread(name)
            hist = gridBasedExtractiong(img)
            testData.extend(hist)
            testLabel = file[1]
        if testData:
            trainscikit(trainData, trainLabel, validationData, validationLabel, testData, testLabel)

if __name__ == "__main__":
    main()

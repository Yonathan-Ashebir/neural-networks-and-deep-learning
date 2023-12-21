import numpy
import mnist_loader
import network


def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))


def testNumpyMatrices():
    g1 = numpy.random.randn(3, 3)
    print(g1)
    g1[0][0] = 5
    print(g1 + g1)

    print(sigmoid(g1))
    print(g1 ** 2)


def load_data():
    global training_data, validation_data, test_data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper();


def classify_digits1():
    global net
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

from torchvision import datasets

print("Downloading MNIST...")
datasets.MNIST(root="data/mnist/", train=True, download=True)
datasets.MNIST(root="data/mnist/", train=False, download=True)

print("Downloading CIFAR-10...")
datasets.CIFAR10(root="data/cifar10/", train=True, download=True)
datasets.CIFAR10(root="data/cifar10/", train=False, download=True)

print("All datasets downloaded and ready at ./data/")

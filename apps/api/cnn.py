import torch.nn as nn

class CatCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Features is a sequential neural network model comprised of four
        # convolutional layers
        # nn.Sequential() returns an object whose type is:
        # torch.nn.modules.container.Sequential
        self.features = nn.Sequential(
            # First convolution layer - Learn simple features
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: 32 x 128 x 128
            nn.MaxPool2d(2),
            # Output: 32 x 64 x 64

            # Second convolution layer - Learn intermediate features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: 64 x 64 x 64
            nn.MaxPool2d(2),
            # Output: 64 x 32 x 32

            # Third convolution layer - Learn more intermediate features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: 128 x 32 x 32
            nn.MaxPool2d(2),
            # Output: 128 x 16 x 16

            # Fourth convolution layer - Learn task-specific features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: 256 x 16 x 16
            nn.MaxPool2d(2)
            # Output: 256 x 8 x 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Map the flattened feature maps of size 256 x 8 x 8 to 256 hidden neurons
            nn.Linear(256 * 8 * 8, 256),
            # ReLU for non-linear activation
            nn.ReLU(),
            # Randomly dropout the output from the hidden neurons to reduce risk of
            #overfitting
            nn.Dropout(0.4),
            # Final output layer with 2 output neurons
            nn.Linear(256, num_classes)
            # Note: The final output layer doesn't need softmax activation because 
            # PyTorch's cross entropy loss calculates the log-softmax from the
            # raw logits of the final linear output layer
        )

    # In Python, instance methods normally write self explicitly
    # Thus, we pass in the self inside the method parameters when we define them
    # In PyTorch, children of the nn.Module class needs to define and override
    # a forward method
    # In PyTorch, the forward() method is automatically called when the children of
    # the nn.Module class gets passed in a data from a loader, with the data
    # being the argument into the forward() method call
    def forward(self, x):
        # In Python, some objects can be called like functions
        # Objects are callable if their class defines a special method called __call__
        # That is the case for nn.Module, of which nn.Sequential is a child class
        # features is a type of nn.Sequential, so it can also be called like a
        # function
        x = self.features(x)
        x = self.classifier(x)
        return x
    
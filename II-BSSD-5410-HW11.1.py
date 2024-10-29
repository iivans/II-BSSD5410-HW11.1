import numpy as np
import pandas as pd # FOR CSV FILE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)

                # Perceptron update rule
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted

    def _activation_function(self, x):
        return np.where(x >= 0, 1, 0)


# Load the Palmer Penguins dataset from GitHub URL
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)

# Drop rows with missing values
penguins.dropna(inplace=True)

penguins = penguins[penguins['species'].isin(['Adelie', 'Chinstrap'])]

# Define features and labels
X = penguins[['bill_length_mm', 'bill_depth_mm']].values  
y = penguins['species'].apply(lambda x: 1 if x == 'Adelie' else 0).values  

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initialize
perceptron = Perceptron(learning_rate=0.1, n_iters=1000)
perceptron.fit(X_train, y_train)

# Predictions
predictions = perceptron.predict(X_test)

# Accuracy
accuracy = np.mean(predictions == y_test)
print(f"Perceptron classification accuracy: {accuracy:.2f}")

# Visualization
def plot_decision_boundary(X, y, model):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                           np.arange(x2_min, x2_max, 0.01))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    predictions = model.predict(grid).reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, predictions, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolor='k')
    plt.title('Perceptron Decision Boundary - Penguins Dataset')
    plt.xlabel('Bill Length (mm)')
    plt.ylabel('Bill Depth (mm)')
    plt.show()

# Plot the decision boundary
plot_decision_boundary(X_train, y_train, perceptron)

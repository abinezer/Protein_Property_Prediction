# **Predicting Protein Properties using a SchNet-like Model**

This is some basic code for performing regression on protein structure data using a SchNet-like model architecure. SchNet is a graph neural network model that can be used for learning properties of molecules or protein structures. The goal of this code is to predict a continuous value (property) associated with a given protein structure.

 This experiment is inspired by the SchNet Paper - [Here](https://arxiv.org/pdf/1706.08566v5.pdf)

## **Code Overview**
The code is written in Python and uses the following libraries:

- `torch`: PyTorch deep learning library.
- `prody`: Protein Dynamics and Structural Bioinformatics library for handling protein structure data.
- `numpy`: Numerical computing library for array operations.
- `matplotlib`: Library for creating plots and visualizations.
- `torch_geometric`: PyTorch library for handling graph data.
- `scipy`: Scientific computing library for distance calculations.
- `sklearn`: Machine learning library for evaluation metrics and data splitting.

### The code consists of the following main sections:

- **Data Loading**: The protein structure data is loaded from PDB files and organized into a dictionary. Each entry in the dictionary contains the protein structure's key (filename) and the associated label (property value).

- **Data Preparation**: The protein structure data is preprocessed to create a PyTorch Geometric Data object. This includes computing the distance matrix, creating an adjacency matrix based on a distance threshold (in this case 20 A), and converting the data to PyTorch tensors.

- **Data Loading and Batching**: The data is split into train and test sets, and then converted into PyTorch Geometric DataLoader objects for efficient batching during training.

- **SchNet Model Definition**: The SchNet model is defined as a graph neural network consisting of SchNet convolutional layers and fully connected layers.

- **Model Training**: The SchNet model is trained on the train data using the mean squared error loss function and the Adam optimizer. The training loop runs for a specified number of epochs, and the model parameters are updated using backpropagation.

- **Model Evaluation**: The trained model is evaluated on the test data by making predictions on the protein structures. Evaluation metrics such as mean squared error, root mean squared error, mean absolute error, and R-squared are calculated.

- **Result Visualization**: The loss values during training are plotted using matplotlib.

## **SchNet**

The SchNet-like model incorporates the following components:

- **SchNetConv Layer**: This layer performs message passing and updates node representations based on the neighboring nodes' features. The SchNetConv layer consists of a multilayer perceptron (MLP) that processes node features and outputs new node representations. The MLP has two linear layers and a non-linear activation function (ReLU). The SchNetConv layer is useful because it allows the model to capture local interactions and learnn rich node representations based on the neighborhood.

- **Fully Connected Layers**: After the SchNetConv layers, the node representations are passed through fully connected layers. These layers further process the learned node representations and capture global information about the graph structure. The fully connected layers are useful for capturing higher-order dependencies and integrating information from the entire graph.

- **Output Layer**: The final fully connected layer in the SchNet model is responsible for producing the output predictions. It maps the learned node representations to the desired output dimension (in this case, a single continuous value). The output layer is useful for transforming the learned features into a format suitable for the regression task.

- **Graph Pooling**: In the SchNet model, global mean pooling is applied to aggregate the node representations into a fixed-size representation. This pooling operation computes the mean of the node features across the graph, resulting in a single feature vector that summarizes the entire graph. Global mean pooling is useful for obtaining a graph-level representation that can be used for making predictions at the whole graph level.

## **Reason for using MSE Loss**

- The objective of this code is to perform regression, which involves predicting a continuous value (property) associated with a protein structure. MSE is a commonly used loss function for regression tasks, as it is differentiable, smooth and measures the average squared difference between the predicted values and the true values. It is well-suited for continuous prediction problems where the magnitude of the difference matters.

## **Why this architecture should work as intended**

 Here's how the model architecture operates and why it should work as intended:

- **Input Features**: The input features in this architecture are the locations of the atoms in the protein structure. These 3D coordinates provide spatial information about the protein's constituent atoms, which is crucial for understanding its structure and properties.

- **SchNetConv Layers**: These layers enable message passing and update the node representations (atom features) based on the features of their neighboring atoms. By applying a multilayer perceptron (MLP) to the atom features, the SchNetConv layers capture local interactions and extract important information from the neighborhood structure.

- **New Representations**: The SchNetConv layers create new representations for the atoms in the protein structure. These representations encode learned information about the atoms and their respective neighborhoods, capturing important structural and chemical features. By iteratively updating the representations through message passing, the model can effectively incorporate and propagate information from neighboring atoms, which is crucial for understanding the protein's properties.

- **Fully Connected Layers**: The fully connected layers follow the SchNetConv layers and further process the learned node representations. These layers capture global information and higher-order dependencies across the graph. By performing additional transformations and non-linear operations, the fully connected layers enable the model to extract complex patterns and relationships from the learned representations.

- **Output Layer**: The output layer is responsible for producing the final predictions. It maps the learned representations to a single continuous value, which represents the predicted property associated with the protein structure. The output layer ensures that the learned features are transformed into a suitable format for the regression task.

- **Integration of Local and Global Information**: By combining the local information captured by the SchNetConv layers and the global information captured by the fully connected layers, the model integrates both local and global structural features. This integration allows the model to learn meaningful and informative representations that reflect the overall protein structure and its associated properties.

# QML
Wrappers for applying kernel ridge regression models using the QMLcode library : https://github.com/qmlcode/qml

# Requirements
* Numpy
* scipy
* scikit-learn
* qml-develop branch : https://github.com/qmlcode/qml/tree/develop

# Usage
The `KRR_example.py` file contains examples on how to perform cross-validate grid search for finding optimal hyperparameters and how to perform kernel ridge regression for both global and local representations using this wrapper. It also contains an example of using a pre-computed kernel matrix to perform kernel ridge regression which is useful with expensive kernels such as FCHL19.

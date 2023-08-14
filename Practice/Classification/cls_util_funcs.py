#!/usr/bin/env python
# coding: utf-8

# # Visualize performance

# ## Confusion matrix

# In[2]:


import numpy as np
import itertools
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix_with_std(model, X_test, y_test, n_samples, classes, model_name='', normalize=False, cmap=plt.cm.Blues, fmt='.3f', figsize=(10, 10), scale=1.0):
    """
    Plots a confusion matrix along with standard deviations for a given model's predictions. Works for multiclass models

    Parameters:
        model (object): The trained classification model.
        X_test (array-like): Test data features.
        y_test (array-like): True labels corresponding to the test data.
        n_samples (int): Number of bootstrap samples for standard deviation calculation.
        classes (list): List of class labels.
        model_name (str, optional): Name of the model for plot title. Default is an empty string.
        normalize (bool, optional): If True, normalize the confusion matrix and standard deviations. Default is False.
        cmap (colormap, optional): Colormap for the plot. Default is plt.cm.Blues.
        fmt (str, optional): Format string for formatting values. Default is '.4f'.
        figsize (tuple, optional): Figure size in inches (width, height). Default is (10, 10).
        scale (float, optional): Scaling factor for font sizes. Default is 1.0.

    Returns:
        None (displays the plot).

    Example:
        # Assuming you have a trained model, X_test, y_test, and class labels
        n_samples = 100  # Number of bootstrap samples
        model_name = "Your Model Name"
        plot_confusion_matrix_with_std(model, X_test, y_test, n_samples, classes, model_name, normalize=True, scale=1.2)
    """
    # Valid for sklearn and tensorflow models
    try:
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    except:
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cm_std = []

    np.random.seed(42)

    random_states = [np.random.randint(0, 100) for _ in range(n_samples)]

    for i, random_state in enumerate(random_states):
        X_test_sample, y_test_sample = resample(X_test, y_test, random_state=random_state)
        y_pred = model.predict(X_test_sample)
        cm_sample = confusion_matrix(y_test_sample, y_pred, labels=range(len(classes)))
        cm_std.append(cm_sample)

    cm_std = np.std(cm_std, axis=0)
    
    if normalize:
        cm_std = cm_std.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    plt.title(f"Confusion matrix {'(normalized)' if normalize else ''} for {model_name} model", fontsize=14 * scale)
    plt.colorbar()

    # Set colorbar limits to [0, 1] for normalized confusion matrix
    if normalize:
        plt.clim(0, 1)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10 * scale)
    plt.yticks(tick_marks, classes, fontsize=10 * scale)

    fmt = fmt if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{format(cm[i, j], fmt)} ± {format(cm_std[i, j], fmt)}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=8 * scale)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=12 * scale)
    plt.xlabel('Predicted label', fontsize=12 * scale)
    plt.show()


# DEMOSTRATION

# In[3]:


# import numpy as np
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# # Create synthetic data
# X, y = make_classification(n_samples=1000, n_features=5, n_classes=4, n_clusters_per_class=1, random_state=42)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a RandomForestClassifier (you can replace this with your own model)
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Define the class labels
# classes = [f'Class {i}' for i in range(4)]

# # Use the plot_confusion_matrix_with_std function
# plot_confusion_matrix_with_std(model, X_test, y_test, n_samples=50, classes=classes, model_name='Random Forest', normalize=True, scale=1.3)


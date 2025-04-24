import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
from itertools import cycle

# Set global default style for matplotlib
def set_matplotlib_dark_theme():
    """Configure matplotlib for dark mode plotting"""
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = '#1e1e1e'
    plt.rcParams['figure.facecolor'] = '#121212'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['grid.color'] = '#444444'


def plot_neural_network(input_size, hidden_size1, hidden_size2, output_size):
    """
    Create a visualization of a neural network architecture.
    
    Parameters:
    -----------
    input_size : int
        Number of input neurons
    hidden_size1 : int
        Number of neurons in first hidden layer
    hidden_size2 : int
        Number of neurons in second hidden layer
    output_size : int
        Number of output neurons
        
    Returns:
    --------
    fig : matplotlib.Figure
        Figure containing the network visualization
    """
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()
    ax.axis('off')
    
    # Define layer positions
    layer_positions = [0.1, 0.4, 0.7, 0.9]
    layer_sizes = [input_size, hidden_size1, hidden_size2, output_size]
    
    # Colors
    node_color = '#4a86e8'  # Blue
    edge_color = '#999999'  # Gray
    
    # Draw nodes and edges
    for l, layer_pos in enumerate(layer_positions):
        current_layer_size = layer_sizes[l]
        
        # Calculate positions for neurons in current layer
        layer_height = 0.8
        neuron_height = layer_height / max(current_layer_size, 1)
        start_height = (1 - (current_layer_size * neuron_height)) / 2
        
        # Draw neurons in current layer
        for i in range(current_layer_size):
            y_pos = start_height + i * neuron_height
            
            # Draw neuron
            circle = plt.Circle((layer_pos, y_pos), 0.02, color=node_color, fill=True)
            ax.add_patch(circle)
            
            # Draw connections to next layer if this is not the output layer
            if l < len(layer_positions) - 1:
                next_layer_size = layer_sizes[l + 1]
                next_layer_pos = layer_positions[l + 1]
                next_neuron_height = 0.8 / max(next_layer_size, 1)
                next_start_height = (1 - (next_layer_size * next_neuron_height)) / 2
                
                # Connect to all neurons in next layer
                for j in range(next_layer_size):
                    next_y_pos = next_start_height + j * next_neuron_height
                    plt.plot([layer_pos, next_layer_pos], [y_pos, next_y_pos], color=edge_color, alpha=0.6, linewidth=0.5)
    
    # Add layer labels
    plt.text(layer_positions[0], 0.02, "Input Layer", horizontalalignment='center')
    plt.text(layer_positions[1], 0.02, "Hidden Layer 1", horizontalalignment='center')
    plt.text(layer_positions[2], 0.02, "Hidden Layer 2", horizontalalignment='center')
    plt.text(layer_positions[3], 0.02, "Output Layer", horizontalalignment='center')
    
    return fig


def plot_training_history(history):
    """
    Plot training and validation metrics from a Keras history object.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        History object from model training
        
    Returns:
    --------
    fig : matplotlib.Figure
        Figure containing the plotted history
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Get available metrics
    metrics = [m for m in history.history.keys() if not m.startswith('val_')]
    
    # Plot each metric
    for i, metric in enumerate(metrics[:2]):  # Limit to first two metrics (typically loss and accuracy)
        ax = axes[i]
        
        # Plot training metric
        ax.plot(history.history[metric], label=f'Training {metric}')
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            ax.plot(history.history[val_metric], label=f'Validation {metric}')
        
        # Add labels and legend
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} vs. Epoch')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig


def plot_activation_functions():
    """
    Plot common activation functions used in neural networks.
    
    Returns:
    --------
    fig : matplotlib.Figure
        Figure containing the activation function plots
    """
    # Create data
    x = np.linspace(-5, 5, 1000)
    
    # Define activation functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(x):
        return np.tanh(x)
    
    def relu(x):
        return np.maximum(0, x)
    
    def leaky_relu(x, alpha=0.1):
        return np.maximum(alpha * x, x)
    
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Plot activation functions
    axes[0].plot(x, sigmoid(x), color='#4a86e8')
    axes[0].set_title('Sigmoid')
    
    axes[1].plot(x, tanh(x), color='#ff9900')
    axes[1].set_title('Tanh')
    
    axes[2].plot(x, relu(x), color='#6aa84f')
    axes[2].set_title('ReLU')
    
    axes[3].plot(x, leaky_relu(x), color='#cc0000')
    axes[3].set_title('Leaky ReLU')
    
    axes[4].plot(x, elu(x), color='#9900ff')
    axes[4].set_title('ELU')
    
    # Add softmax plot
    x_softmax = np.array([-2, -1, 0, 1, 2])
    y_softmax = np.exp(x_softmax) / np.sum(np.exp(x_softmax))
    axes[5].bar(x_softmax, y_softmax, color='#ff00ff')
    axes[5].set_title('Softmax')
    
    # Add grid, labels to all plots
    for ax in axes:
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
    
    plt.tight_layout()
    return fig


def plot_decision_boundary(model, X, y, scaler=None):
    """
    Plot decision boundary for a binary classification model.
    
    Parameters:
    -----------
    model : sklearn classifier
        Trained model with predict method
    X : numpy.ndarray
        Input features (2D)
    y : numpy.ndarray
        Target labels
    scaler : sklearn.preprocessing._data.StandardScaler, optional
        Scaler used to normalize the data
        
    Returns:
    --------
    fig : matplotlib.Figure
        Figure containing the decision boundary plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define bounds
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Create grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Prepare data for prediction
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Apply same normalization as training data if provided
    if scaler is not None:
        grid_points = scaler.transform(grid_points)
    
    # Get predictions
    Z = model.predict(grid_points)
    
    # Reshape back to grid
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    cmap = ListedColormap(['#0000FF', '#FF0000'])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
    
    # Add legend
    handles, labels = scatter.legend_elements()
    ax.legend(handles, ['Class 0', 'Class 1'], loc="upper right")
    
    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary')
    
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix for classification results.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    class_names : list, optional
        Names of classes
        
    Returns:
    --------
    fig : matplotlib.Figure
        Figure containing the confusion matrix plot
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix as image
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    # Set class names
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    # Set tick marks and labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Add values to cells
    for i in range(n_classes):
        for j in range(n_classes):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", color=text_color)
    
    # Add labels
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names=None, top_n=10):
    """
    Plot feature importance for a model.
    
    Parameters:
    -----------
    model : sklearn model with feature_importances_ or coef_ attribute
        Trained model
    feature_names : list, optional
        Names of features
    top_n : int, optional
        Number of top features to show
        
    Returns:
    --------
    fig : matplotlib.Figure
        Figure containing the feature importance plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check model type for feature importance
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_).flatten()
    else:
        # For neural network models like MLPClassifier
        if hasattr(model, 'coefs_') and len(model.coefs_) > 0:
            # Use first layer weights as approximation
            importances = np.mean(np.abs(model.coefs_[0]), axis=1)
        else:
            raise ValueError("Model type not supported for feature importance")
    
    # Set feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    # Sort features by importance
    indices = np.argsort(importances)[-top_n:]
    
    # Plot
    ax.barh(range(len(indices)), importances[indices], color='#4a86e8')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top Feature Importance')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_pred_prob, n_classes=None):
    """
    Plot ROC curve for classification results.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels (one-hot encoded for multi-class)
    y_pred_prob : numpy.ndarray
        Predicted probabilities
    n_classes : int, optional
        Number of classes
        
    Returns:
    --------
    fig : matplotlib.Figure
        Figure containing the ROC curve plot
    """
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine if binary or multi-class
    if n_classes is None:
        if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > 1:
            # Multi-class case
            n_classes = y_pred_prob.shape[1]
        else:
            # Binary case
            n_classes = 1
    
    # Convert to one-hot if needed for multi-class
    if n_classes > 1 and len(y_true.shape) == 1:
        from sklearn.preprocessing import label_binarize
        y_true = label_binarize(y_true, classes=range(n_classes))
    
    # Binary classification case
    if n_classes == 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='#4a86e8', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
    
    # Multi-class case
    else:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        colors = cycle(['#4a86e8', '#ff9900', '#6aa84f', '#cc0000', '#9900ff'])
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
    
    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Plot learning curve for a model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to evaluate
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Targets
    cv : int, optional
        Number of cross-validation folds
    train_sizes : numpy.ndarray, optional
        Training sizes to evaluate
        
    Returns:
    --------
    fig : matplotlib.Figure
        Figure containing the learning curve plot
    """
    from sklearn.model_selection import learning_curve
    
    # Compute learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes,
        return_times=False)
    
    # Calculate mean and std
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot learning curve
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#4a86e8")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="#ff9900")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="#4a86e8",
             label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="#ff9900",
             label="Cross-validation score")
    
    # Set plot properties
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curve")
    ax.legend(loc="best")
    
    plt.tight_layout()
    return fig

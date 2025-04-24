import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_regression

def set_matplotlib_dark_theme():
    """
    Configure matplotlib for dark mode plotting.
    
    This function sets the global style properties for matplotlib to
    create visualizations that look good on a dark-themed website.
    """
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = '#1e1e1e'
    plt.rcParams['figure.facecolor'] = '#121212'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['grid.color'] = '#444444'
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # Color cycler for multiple lines
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        color=['#ff4b4b', '#4a86e8', '#ff9900', '#109618', '#9900ff', '#dd4477']
    )

def create_synthetic_data(n_samples=1000, noise=0.3, random_state=None):
    """
    Generate synthetic data for demonstration purposes.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Noise level to add to the data
    random_state : int or None
        Random state for reproducibility
        
    Returns:
    --------
    X : numpy.ndarray
        Features array
    y : numpy.ndarray
        Target array
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate 2D features
    X = np.random.randn(n_samples, 2)
    
    # Generate target based on non-linear function
    y = (X[:, 0]**2 - 0.5 * X[:, 1]**2 + X[:, 0] * X[:, 1] + noise * np.random.randn(n_samples)) > 0
    y = y.astype(int)
    
    return X, y

def create_time_series_data(n_steps=1000, noise=0.2, random_state=None):
    """
    Generate synthetic time series data for demonstration purposes.
    
    Parameters:
    -----------
    n_steps : int
        Number of time steps to generate
    noise : float
        Noise level to add to the time series
    random_state : int or None
        Random state for reproducibility
        
    Returns:
    --------
    time_series : numpy.ndarray
        Generated time series
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate time steps
    time = np.arange(0, n_steps)
    
    # Generate components
    trend = 0.001 * time  # Linear trend
    seasonal = 0.5 * np.sin(2 * np.pi * time / 50) + 0.25 * np.sin(2 * np.pi * time / 25)  # Seasonal component
    noise_component = noise * np.random.randn(n_steps)  # Noise component
    
    # Combine components
    time_series = trend + seasonal + noise_component
    
    return time_series

def prepare_time_series_data(series, lookback=10):
    """
    Prepare time series data for sequence prediction models.
    
    Parameters:
    -----------
    series : numpy.ndarray
        Original time series data
    lookback : int
        Number of previous time steps to use for prediction
        
    Returns:
    --------
    X : numpy.ndarray
        Input sequences with shape (n_samples, lookback, 1)
    y : numpy.ndarray
        Target values with shape (n_samples,)
    """
    X, y = [], []
    
    for i in range(len(series) - lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    
    # Reshape for LSTM [samples, time steps, features]
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    
    return X, y

def generate_image_batch(batch_size=32, img_size=28, random_state=None):
    """
    Generate a batch of random images for demonstration purposes.
    
    Parameters:
    -----------
    batch_size : int
        Number of images to generate
    img_size : int
        Size of square images to generate
    random_state : int or None
        Random state for reproducibility
        
    Returns:
    --------
    images : numpy.ndarray
        Batch of images with shape (batch_size, img_size, img_size, 1)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random images
    images = np.random.rand(batch_size, img_size, img_size, 1)
    
    # Add some simple shapes to make images more interesting
    for i in range(batch_size):
        # Add a random shape
        shape_type = np.random.choice(['circle', 'square', 'line'])
        
        if shape_type == 'circle':
            center_x, center_y = np.random.randint(5, img_size - 5, 2)
            radius = np.random.randint(3, min(center_x, center_y, img_size - center_x, img_size - center_y) - 2)
            
            for x in range(img_size):
                for y in range(img_size):
                    if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                        images[i, x, y, 0] = 1.0
        
        elif shape_type == 'square':
            top, left = np.random.randint(5, img_size - 10, 2)
            size = np.random.randint(5, min(img_size - top, img_size - left) - 2)
            
            images[i, top:top+size, left:left+size, 0] = 1.0
        
        else:  # line
            start_x, start_y = np.random.randint(2, img_size - 2, 2)
            end_x, end_y = np.random.randint(2, img_size - 2, 2)
            
            # Draw simple line
            steps = max(abs(end_x - start_x), abs(end_y - start_y)) + 1
            for step in range(steps):
                t = step / steps
                x = int(start_x + t * (end_x - start_x))
                y = int(start_y + t * (end_y - start_y))
                
                if 0 <= x < img_size and 0 <= y < img_size:
                    images[i, x, y, 0] = 1.0
    
    return images

def setup_sklearn_logging():
    """
    Configure logging for scikit-learn models to show training progress.
    
    Returns:
    --------
    bool:
        True if logging was set up successfully, False otherwise
    """
    try:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        return True
    except:
        print("Could not set up logging")
        return False

def create_sklearn_pipeline(preprocessors=None, model=None):
    """
    Create a scikit-learn pipeline with preprocessing steps and a model.
    
    Parameters:
    -----------
    preprocessors : list, optional
        List of (name, transformer) tuples for preprocessing steps
    model : estimator, optional
        The final estimator in the pipeline
        
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
        Scikit-learn pipeline
    """
    from sklearn.pipeline import Pipeline
    
    steps = []
    
    # Add preprocessing steps if provided
    if preprocessors:
        steps.extend(preprocessors)
    
    # Add the model if provided
    if model:
        steps.append(('model', model))
    
    return Pipeline(steps)

def plot_image_grid(images, labels=None, pred_labels=None, n_rows=4, n_cols=4):
    """
    Plot a grid of images with optional labels.
    
    Parameters:
    -----------
    images : numpy.ndarray
        Batch of images with shape (batch_size, height, width, channels)
    labels : numpy.ndarray, optional
        Ground truth labels
    pred_labels : numpy.ndarray, optional
        Predicted labels
    n_rows, n_cols : int
        Grid dimensions
        
    Returns:
    --------
    fig : matplotlib.Figure
        Figure containing the image grid
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(images):
            # Get image and remove extra dimension if needed
            img = images[i]
            if img.shape[-1] == 1:
                img = img.squeeze()
            
            # Display image
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            
            # Add label if provided
            title = ""
            if labels is not None and i < len(labels):
                title += f"True: {labels[i]}"
            if pred_labels is not None and i < len(pred_labels):
                title += f"\nPred: {pred_labels[i]}"
            
            if title:
                ax.set_title(title)
            
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    return fig

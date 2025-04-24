import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Note: TensorFlow import commented for now due to compatibility issues
# Will use scikit-learn for simpler models
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression
import sklearn.datasets as datasets

# Custom visualization modules
from visualizations import (
    plot_neural_network, 
    plot_training_history, 
    plot_activation_functions,
    plot_decision_boundary
)
from code_examples import get_code_examples
from utils import set_matplotlib_dark_theme

# Set page config
st.set_page_config(
    page_title="Deep Learning Portfolio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply dark theme to matplotlib
set_matplotlib_dark_theme()

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Neural Network Types", "Frameworks", "Models", "Live Demos", "Code Examples"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    # About
    This portfolio showcases various neural network implementations and visualizations.
    
    Built with Streamlit, scikit-learn, and Python.
    """)
    
    # Content based on selected page
    if page == "Home":
        home_page()
    elif page == "Neural Network Types":
        nn_types_page()
    elif page == "Frameworks":
        frameworks_page()
    elif page == "Models":
        models_page()
    elif page == "Live Demos":
        live_demos_page()
    elif page == "Code Examples":
        code_examples_page()

def home_page():
    st.title("Deep Learning Portfolio")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ## Welcome to my Neural Network Portfolio
        
        This interactive website showcases my expertise in deep learning and neural networks.
        Explore live demonstrations, visualizations, and implementations of various neural network
        architectures.
        
        ### Featured Content:
        
        - **Neural Network Visualizations**: Interactive diagrams of different neural network architectures
        - **Live Training Demos**: Watch neural networks learn in real-time
        - **Implementation Examples**: Code snippets for key neural network implementations
        - **Interactive Experiments**: Adjust parameters and see how networks respond
        
        Use the sidebar navigation to explore different sections of this portfolio.
        """)
    
    with col2:
        st.markdown("### Neural Network Architecture")
        # Display a simple neural network architecture visualization
        fig = plot_neural_network(3, 4, 4, 1)
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.markdown("""
    ## What are Neural Networks?
    
    Neural networks are computational models inspired by the human brain. They consist of layers of nodes (neurons)
    that process information and learn patterns from data. Key concepts include:
    
    - **Neurons**: Basic computational units that process inputs
    - **Layers**: Groups of neurons that transform data
    - **Weights & Biases**: Parameters adjusted during learning
    - **Activation Functions**: Non-linear functions that introduce complexity
    - **Training**: Process of adjusting weights to minimize errors
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Activation Functions")
        fig = plot_activation_functions()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Common Neural Network Applications")
        st.markdown("""
        - Image Classification
        - Natural Language Processing
        - Time Series Forecasting
        - Recommendation Systems
        - Anomaly Detection
        - Reinforcement Learning
        """)
    
    st.markdown("---")
    
    # New section on Backpropagation
    st.header("Backpropagation")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Backpropagation is the fundamental algorithm used to train neural networks efficiently. It consists of two main phases:
        
        1. **Forward Pass**: Input data flows through the network, generating predictions
        2. **Backward Pass**: Error gradients are computed and propagated backward through the network
        
        The algorithm uses the chain rule from calculus to find the gradient of the loss function with respect to each weight in the network. These gradients indicate how to adjust weights to minimize error.
        
        ### Key Steps:
        - Calculate error at the output layer
        - Propagate error backwards through the network
        - Update weights using gradient descent
        - Repeat for multiple epochs until convergence
        """)
    
    with col2:
        st.image("https://miro.medium.com/max/1400/1*3fA77_mLNiJTSgZFhYnU0Q.png", 
                 caption="Backpropagation Process")
    
    # Section on Activation Functions
    st.header("Activation Functions")
    st.markdown("""
    Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Without them, neural networks would be limited to representing linear relationships.
    """)
    
    act_tabs = st.tabs(["ReLU", "Sigmoid", "Tanh", "Others"])
    
    with act_tabs[0]:
        st.subheader("ReLU (Rectified Linear Unit)")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            **Formula**: f(x) = max(0, x)
            
            **Advantages**:
            - Computationally efficient
            - Reduces vanishing gradient problem
            - Induces sparsity in neural networks
            
            **Disadvantages**:
            - "Dying ReLU" problem (neurons can become inactive)
            - Not zero-centered
            """)
        with col2:
            st.latex(r"f(x) = \max(0, x)")
    
    with act_tabs[1]:
        st.subheader("Sigmoid")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            **Formula**: f(x) = 1 / (1 + e^(-x))
            
            **Advantages**:
            - Output bounded between 0 and 1
            - Smooth gradient
            - Clear probabilistic interpretation
            
            **Disadvantages**:
            - Suffers from vanishing gradient problem
            - Not zero-centered
            - Computationally expensive
            """)
        with col2:
            st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
    
    with act_tabs[2]:
        st.subheader("Tanh (Hyperbolic Tangent)")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            **Formula**: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
            
            **Advantages**:
            - Zero-centered output (-1 to 1)
            - Stronger gradients than sigmoid
            
            **Disadvantages**:
            - Still suffers from vanishing gradient problem
            - Computationally expensive
            """)
        with col2:
            st.latex(r"f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}")
    
    with act_tabs[3]:
        st.subheader("Other Activation Functions")
        st.markdown("""
        - **Leaky ReLU**: f(x) = max(αx, x) where α is a small constant (e.g., 0.01)
        - **ELU** (Exponential Linear Unit): f(x) = x if x > 0, else α(e^x - 1)
        - **GELU** (Gaussian Error Linear Unit): Used in transformers
        - **Swish**: f(x) = x * sigmoid(x), introduced by Google
        """)
    
    # Section on Overfitting and Regularization
    st.header("Overfitting and Regularization")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        **Overfitting** occurs when a model learns the training data too well, capturing noise rather than the underlying pattern. Signs of overfitting include:
        
        - High accuracy on training data but poor performance on test data
        - Complex model that fits training data perfectly
        - Large weights with high variance
        
        **Regularization** techniques prevent overfitting by constraining model complexity:
        
        - **L1 Regularization (Lasso)**: Adds absolute value of weights to loss function, induces sparsity
        - **L2 Regularization (Ridge)**: Adds squared weights to loss function, prevents large weight values
        - **Dropout**: Randomly disables neurons during training
        - **Early Stopping**: Halt training when validation performance stops improving
        - **Data Augmentation**: Artificially expand training dataset with transformed examples
        """)
    
    with col2:
        st.image("https://miro.medium.com/max/1144/1*_7OPgojau8hkiPUiHoGK_w.png", 
                 caption="Overfitting vs. Underfitting")
        
        st.markdown("""
        **Balancing Model Complexity**
        - Too simple → Underfitting
        - Too complex → Overfitting
        - Just right → Good generalization
        """)

def nn_types_page():
    st.title("Neural Network Types")
    
    tab1, tab2, tab3 = st.tabs(["Feedforward Networks", "Convolutional Networks", "Recurrent Networks"])
    
    with tab1:
        st.header("Feedforward Neural Networks")
        st.markdown("""
        Feedforward Neural Networks (FNNs) are the simplest type of artificial neural network. 
        Information moves in only one direction—forward—from the input nodes, through the hidden layers, 
        and to the output nodes. There are no cycles or loops in the network.
        
        ### Key Features:
        
        - Simple architecture with fully connected layers
        - Each neuron in one layer connects to every neuron in the next layer
        - Good for tabular data and simple classification/regression tasks
        - Prone to overfitting with high-dimensional data
        """)
        
        # Display simple feedforward neural network
        st.subheader("Architecture Visualization")
        ffnn_fig = plot_neural_network(4, 6, 6, 2)
        st.pyplot(ffnn_fig)
        
        # Show code example
        st.subheader("Implementation Example")
        st.code(get_code_examples()["feedforward"], language="python")
    
    with tab2:
        st.header("Convolutional Neural Networks")
        st.markdown("""
        Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images. 
        They use convolution operations to extract features from the input data.
        
        ### Key Features:
        
        - Specialized for grid-like data (images, spectrograms)
        - Use convolutional layers to extract spatial features
        - Include pooling layers to reduce dimensionality
        - Shared weights reduce parameter count compared to fully connected networks
        - Translation invariance makes them robust to object position shifts
        """)
        
        # Show CNN architecture with example
        st.subheader("CNN Architecture")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.image("https://miro.medium.com/max/2000/1*vkQ0hXDaQv57sALXAJquxA.jpeg", 
                     caption="CNN Architecture")
        with col2:
            st.markdown("""
            CNNs typically contain:
            
            1. Convolutional layers
            2. Activation functions (ReLU)
            3. Pooling layers
            4. Fully connected layers
            
            This design allows them to automatically learn spatial hierarchies of features.
            """)
        
        # Show code example
        st.subheader("Implementation Example")
        st.code(get_code_examples()["cnn"], language="python")
    
    with tab3:
        st.header("Recurrent Neural Networks")
        st.markdown("""
        Recurrent Neural Networks (RNNs) are designed to work with sequential data. They have connections 
        that form directed cycles, allowing the network to maintain an internal state.
        
        ### Key Features:
        
        - Designed for sequential/temporal data
        - Maintain internal memory of previous inputs
        - Can process inputs of variable length
        - Variants like LSTM and GRU address vanishing gradient problems
        - Widely used in NLP, time series forecasting, and speech recognition
        """)
        
        # Show RNN architecture
        st.subheader("RNN Architecture")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png", 
                     caption="RNN Unrolled")
        with col2:
            st.markdown("""
            RNNs incorporate feedback loops, allowing information persistence:
            
            - Simple RNN: Basic recurrent architecture
            - LSTM: Long Short-Term Memory units
            - GRU: Gated Recurrent Units
            
            These architectures help the network remember important information over longer sequences.
            """)
        
        # Show code example
        st.subheader("Implementation Example")
        st.code(get_code_examples()["rnn"], language="python")

def live_demos_page():
    st.title("Live Neural Network Demos")
    
    demo_choice = st.selectbox(
        "Select a demo:", 
        ["Binary Classification", "Image Classification", "Time Series Prediction"]
    )
    
    if demo_choice == "Binary Classification":
        binary_classification_demo()
    elif demo_choice == "Image Classification":
        image_classification_demo()
    elif demo_choice == "Time Series Prediction":
        time_series_demo()

def binary_classification_demo():
    st.header("Binary Classification Demo")
    st.markdown("""
    This demo shows a neural network learning to classify points in a 2D space.
    You can adjust the complexity of the data and the network architecture, then watch it learn.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Controls for data and model
        st.subheader("Parameters")
        
        # Data parameters
        st.markdown("#### Data Parameters")
        n_samples = st.slider("Number of samples", 50, 500, 200)
        noise = st.slider("Noise level", 0.0, 1.0, 0.2)
        pattern = st.selectbox("Data pattern", ["Circles", "Moons", "Blobs"])
        
        # Model parameters
        st.markdown("#### Model Parameters")
        n_hidden_layers = st.slider("Hidden layers", 1, 5, 2)
        n_neurons = st.slider("Neurons per layer", 2, 32, 8)
        activation = st.selectbox("Activation function", ["relu", "tanh", "sigmoid"])
        learning_rate = st.select_slider(
            "Learning rate", 
            options=[0.001, 0.01, 0.05, 0.1, 0.5], 
            value=0.01
        )
        
        # Training parameters
        st.markdown("#### Training Parameters")
        epochs = st.slider("Epochs", 10, 200, 50)
        
        # Training button
        train_button = st.button("Train Model")
    
    with col2:
        # Generate synthetic data based on selected pattern
        if pattern == "Circles":
            from sklearn.datasets import make_circles
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        elif pattern == "Moons":
            from sklearn.datasets import make_moons
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        else:  # Blobs
            from sklearn.datasets import make_blobs
            X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, random_state=42, cluster_std=noise*3)
        
        # Split data into train and test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Display the data
        st.subheader("Data Visualization")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Binary Classification Data')
        st.pyplot(fig)
        
        # Create and train model when button is clicked
        if train_button:
            with st.spinner("Training the model..."):
                # Create model with scikit-learn's MLPClassifier
                hidden_layer_sizes = tuple([n_neurons] * n_hidden_layers)
                
                # Create model
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    solver='adam',
                    alpha=0.0001,
                    learning_rate_init=learning_rate,
                    max_iter=epochs,
                    random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Store training history in a format similar to keras
                class HistoryMock:
                    def __init__(self, model):
                        self.history = {
                            'loss': model.loss_curve_,
                            'accuracy': [(1-l) for l in model.loss_curve_]  # Approximating accuracy from loss
                        }
                
                history = HistoryMock(model)
                
                # Plot training history
                st.subheader("Training History")
                history_fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(history.history['loss'], label='Training Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss vs. Epoch')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(history_fig)
                
                # Plot decision boundary
                st.subheader("Decision Boundary")
                
                # Manually create decision boundary plot
                x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
                y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                    np.arange(y_min, y_max, 0.02))
                
                # Scale the grid points
                grid_points = np.c_[xx.ravel(), yy.ravel()]
                grid_points_scaled = scaler.transform(grid_points)
                
                # Predict
                Z = model.predict(grid_points_scaled)
                Z = Z.reshape(xx.shape)
                
                # Plot
                decision_fig, ax = plt.subplots(figsize=(10, 8))
                ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
                
                # Plot original data points
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
                
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                ax.set_title('Decision Boundary')
                
                st.pyplot(decision_fig)
                
                # Evaluate model
                accuracy = model.score(X_test, y_test)
                st.success(f"Test Accuracy: {accuracy:.4f}")

def image_classification_demo():
    st.header("Image Classification Demo")
    st.markdown("""
    This demo shows how a neural network can be used for image classification.
    We'll use scikit-learn to recognize handwritten digits from the MNIST dataset.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("MNIST Dataset")
        st.markdown("""
        The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9),
        each 28x28 pixels. It's a common benchmark in machine learning.
        
        In this demo, we'll:
        
        1. Load a subset of MNIST data
        2. Train a neural network to classify digits
        3. Visualize model performance
        """)
        
        # Demo controls
        st.subheader("Model Parameters")
        hidden_units = st.slider("Hidden units", 16, 128, 64)
        hidden_layers = st.slider("Hidden layers", 1, 3, 1)
        epochs = st.slider("Training Epochs", 10, 100, 50, help="Higher values give better accuracy but take longer")
        
        train_button = st.button("Train Image Model")
    
    with col2:
        # Load MNIST data
        if 'mnist_data' not in st.session_state:
            with st.spinner("Loading MNIST dataset..."):
                from sklearn.datasets import fetch_openml
                
                # Load a smaller subset of MNIST
                mnist = fetch_openml('mnist_784', version=1, parser='auto')
                X = mnist.data.astype('float64') / 255.0  # Scale to [0, 1]
                y = mnist.target.astype('int')
                
                # Split into train and test
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Store only a subset for faster training
                st.session_state.mnist_data = {
                    'X_train': X_train[:5000],
                    'y_train': y_train[:5000],
                    'X_test': X_test[:1000],
                    'y_test': y_test[:1000]
                }
        
        # Display sample images
        st.subheader("Sample MNIST Digits")
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i, ax in enumerate(axes.flat):
            if i < 10:  # Make sure we don't go out of bounds
                img = st.session_state.mnist_data['X_train'][i].reshape(28, 28)
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Digit: {st.session_state.mnist_data['y_train'][i]}")
                ax.axis('off')
        st.pyplot(fig)
        
        if train_button:
            with st.spinner("Training neural network model..."):
                # Create hidden layer sizes
                hidden_layer_sizes = tuple([hidden_units] * hidden_layers)
                
                # Build multi-layer perceptron classifier
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    batch_size=64,
                    max_iter=epochs,
                    verbose=False,
                    random_state=42
                )
                
                # Train model
                model.fit(
                    st.session_state.mnist_data['X_train'],
                    st.session_state.mnist_data['y_train']
                )
                
                # Plot learning curve
                st.subheader("Training Results")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(model.loss_curve_, label='Training Loss')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss vs. Iterations')
                ax.legend()
                st.pyplot(fig)
                
                # Evaluate model
                accuracy = model.score(
                    st.session_state.mnist_data['X_test'],
                    st.session_state.mnist_data['y_test']
                )
                st.success(f"Test Accuracy: {accuracy:.4f}")
                
                # Show sample predictions
                st.subheader("Sample Predictions")
                
                # Get random test samples
                import random
                test_indices = random.sample(range(len(st.session_state.mnist_data['X_test'])), 8)
                test_samples = st.session_state.mnist_data['X_test'][test_indices]
                true_labels = st.session_state.mnist_data['y_test'][test_indices]
                
                # Make predictions
                predictions = model.predict(test_samples)
                
                # Display images with predictions
                fig, axes = plt.subplots(2, 4, figsize=(12, 6))
                for i, ax in enumerate(axes.flat):
                    if i < len(test_samples):
                        ax.imshow(test_samples[i].reshape(28, 28), cmap='gray')
                        color = 'green' if predictions[i] == true_labels[i] else 'red'
                        ax.set_title(f"Pred: {predictions[i]}\nTrue: {true_labels[i]}", color=color)
                        ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig)

def time_series_demo():
    st.header("Time Series Prediction Demo")
    st.markdown("""
    This demo shows how machine learning can be used to forecast time series data.
    We'll generate a synthetic time series and train a model to predict future values.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        
        # Time series parameters
        st.markdown("#### Data Parameters")
        seq_length = st.slider("Sequence Length", 50, 300, 150)
        seasonality = st.slider("Seasonality Factor", 0.0, 5.0, 1.0)
        noise = st.slider("Noise Level", 0.0, 2.0, 0.5)
        
        # Model parameters
        st.markdown("#### Model Parameters")
        hidden_units = st.slider("Hidden Units", 10, 100, 50)
        lookback = st.slider("Lookback Window", 3, 20, 10, 
                            help="Number of previous time steps to use for prediction")
        alpha = st.slider("Regularization (Alpha)", 0.0001, 0.1, 0.001, step=0.001)
        
        # Training button
        train_button = st.button("Train Time Series Model")
    
    with col2:
        # Generate synthetic time series data
        def generate_time_series(length, seasonality, noise):
            time = np.arange(0, length)
            # Trend component
            trend = 0.01 * time
            # Seasonal component
            season = seasonality * np.sin(2 * np.pi * time / 25)
            # Noise component
            random_noise = noise * np.random.randn(length)
            # Combine components
            series = trend + season + random_noise
            return series
        
        # Generate data
        series = generate_time_series(seq_length, seasonality, noise)
        
        # Plot the time series
        st.subheader("Generated Time Series")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(series)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Synthetic Time Series')
        st.pyplot(fig)
        
        if train_button:
            with st.spinner("Training time series model..."):
                # Prepare data for time series prediction
                def create_dataset(data, lookback):
                    X, y = [], []
                    for i in range(len(data) - lookback):
                        X.append(data[i:i+lookback])
                        y.append(data[i+lookback])
                    return np.array(X), np.array(y)
                
                # Create sequences
                X, y = create_dataset(series, lookback)
                
                # Reshape for sklearn models [samples, features]
                X = X.reshape(X.shape[0], X.shape[1])
                
                # Split into train and test sets
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                # Use MLPRegressor for time series prediction
                model = MLPRegressor(
                    hidden_layer_sizes=(hidden_units, hidden_units//2),
                    activation='relu',
                    solver='adam',
                    alpha=alpha,
                    batch_size=min(32, len(X_train)),
                    max_iter=500,
                    verbose=False,
                    random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Plot learning curve
                st.subheader("Training Process")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(model.loss_curve_, label='Training Loss')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Loss (MSE)')
                ax.set_title('Training Loss Over Iterations')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Make predictions
                train_predict = model.predict(X_train)
                test_predict = model.predict(X_test)
                
                # Prepare data for plotting
                train_plot = np.empty_like(series)
                train_plot[:] = np.nan
                train_plot[lookback:lookback+len(train_predict)] = train_predict
                
                test_plot = np.empty_like(series)
                test_plot[:] = np.nan
                test_plot[lookback+len(train_predict):lookback+len(train_predict)+len(test_predict)] = test_predict
                
                # Plot results
                st.subheader("Prediction Results")
                pred_fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(series, label='Actual Data')
                ax.plot(train_plot, label='Training Predictions', color='green')
                ax.plot(test_plot, label='Testing Predictions', color='red')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.set_title('Time Series Prediction')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(pred_fig)
                
                # Calculate and display error metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                
                mse = mean_squared_error(y_test, test_predict)
                mae = mean_absolute_error(y_test, test_predict)
                
                st.subheader("Error Metrics")
                col1, col2 = st.columns(2)
                col1.metric("Mean Squared Error", f"{mse:.4f}")
                col2.metric("Mean Absolute Error", f"{mae:.4f}")
                
                # Forecast future values
                st.subheader("Future Forecast")
                forecast_steps = 20
                
                # Use the last lookback points as the starting point
                last_sequence = series[-lookback:].reshape(1, -1)
                forecast = []
                
                # Generate predictions one by one
                current_sequence = last_sequence.copy()
                for _ in range(forecast_steps):
                    # Get prediction for next step
                    next_pred = model.predict(current_sequence)[0]
                    forecast.append(next_pred)
                    
                    # Update sequence (remove oldest value, add prediction)
                    current_sequence = np.roll(current_sequence, -1)
                    current_sequence[0, -1] = next_pred
                
                # Plot forecast
                forecast_fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(range(len(series)), series, label='Historical Data')
                ax.plot(range(len(series), len(series) + forecast_steps), forecast, label='Forecast', color='red')
                ax.axvline(x=len(series)-1, color='k', linestyle='--')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.set_title('Future Forecast')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(forecast_fig)

def code_examples_page():
    st.title("Neural Network Code Examples")
    
    st.markdown("""
    This section provides complete code examples for implementing different types of neural networks
    using TensorFlow and Keras. These examples are designed to be educational and can serve as starting
    points for your own projects.
    """)
    
    code_category = st.selectbox(
        "Select a category:", 
        ["Basic Neural Networks", "Computer Vision", "Natural Language Processing", "Utilities"]
    )
    
    if code_category == "Basic Neural Networks":
        basic_nn_examples()
    elif code_category == "Computer Vision":
        computer_vision_examples()
    elif code_category == "Natural Language Processing":
        nlp_examples()
    elif code_category == "Utilities":
        utility_examples()

def basic_nn_examples():
    st.header("Basic Neural Network Examples")
    
    example = st.selectbox(
        "Select an example:",
        ["Feedforward Neural Network", "Deep Neural Network", "Custom Training Loop"]
    )
    
    if example == "Feedforward Neural Network":
        st.subheader("Simple Feedforward Neural Network")
        st.markdown("""
        This example demonstrates a basic feedforward neural network for binary classification.
        """)
        st.code(get_code_examples()["feedforward"], language="python")
        
    elif example == "Deep Neural Network":
        st.subheader("Deep Neural Network with Multiple Layers")
        st.markdown("""
        This example shows how to create a deeper neural network with multiple hidden layers.
        """)
        st.code(get_code_examples()["deep_nn"], language="python")
        
    elif example == "Custom Training Loop":
        st.subheader("Neural Network with Custom Training Loop")
        st.markdown("""
        This example demonstrates how to implement a custom training loop for fine-grained control.
        """)
        st.code(get_code_examples()["custom_training"], language="python")

def computer_vision_examples():
    st.header("Computer Vision Examples")
    
    example = st.selectbox(
        "Select an example:",
        ["CNN for Image Classification", "Transfer Learning", "Object Detection"]
    )
    
    if example == "CNN for Image Classification":
        st.subheader("CNN for Image Classification")
        st.markdown("""
        This example shows how to build a Convolutional Neural Network for image classification.
        """)
        st.code(get_code_examples()["cnn"], language="python")
        
    elif example == "Transfer Learning":
        st.subheader("Transfer Learning with Pre-trained Model")
        st.markdown("""
        This example demonstrates how to use transfer learning with a pre-trained model.
        """)
        st.code(get_code_examples()["transfer_learning"], language="python")
        
    elif example == "Object Detection":
        st.subheader("Object Detection with TensorFlow")
        st.markdown("""
        This example shows the setup for object detection using TensorFlow models.
        """)
        st.code(get_code_examples()["object_detection"], language="python")

def nlp_examples():
    st.header("Natural Language Processing Examples")
    
    example = st.selectbox(
        "Select an example:",
        ["Text Classification", "RNN/LSTM for Sequence Data", "Word Embeddings"]
    )
    
    if example == "Text Classification":
        st.subheader("Text Classification with Neural Networks")
        st.markdown("""
        This example demonstrates text classification using neural networks.
        """)
        st.code(get_code_examples()["text_classification"], language="python")
        
    elif example == "RNN/LSTM for Sequence Data":
        st.subheader("RNN/LSTM for Sequence Data")
        st.markdown("""
        This example shows how to use RNNs and LSTMs for sequence data.
        """)
        st.code(get_code_examples()["rnn"], language="python")
        
    elif example == "Word Embeddings":
        st.subheader("Working with Word Embeddings")
        st.markdown("""
        This example demonstrates how to use word embeddings in neural networks.
        """)
        st.code(get_code_examples()["word_embeddings"], language="python")

def utility_examples():
    st.header("Utility Examples")
    
    example = st.selectbox(
        "Select an example:",
        ["Data Preprocessing", "Model Evaluation", "Hyperparameter Tuning"]
    )
    
    if example == "Data Preprocessing":
        st.subheader("Data Preprocessing for Neural Networks")
        st.markdown("""
        This example shows common data preprocessing techniques for neural networks.
        """)
        st.code(get_code_examples()["data_preprocessing"], language="python")
        
    elif example == "Model Evaluation":
        st.subheader("Model Evaluation Techniques")
        st.markdown("""
        This example demonstrates how to evaluate neural network models.
        """)
        st.code(get_code_examples()["model_evaluation"], language="python")
        
    elif example == "Hyperparameter Tuning":
        st.subheader("Hyperparameter Tuning for Neural Networks")
        st.markdown("""
        This example shows how to perform hyperparameter tuning for neural networks.
        """)
        st.code(get_code_examples()["hyperparameter_tuning"], language="python")
        
def frameworks_page():
    st.title("Deep Learning Frameworks")
    
    st.markdown("""
    Deep learning frameworks provide the building blocks and tools needed to design, train, and validate
    deep neural networks. Each framework has its own strengths, philosophy, and ecosystem.
    """)
    
    # Create tabs for different frameworks
    framework_tabs = st.tabs(["TensorFlow", "PyTorch", "Keras"])
    
    # TensorFlow tab
    with framework_tabs[0]:
        st.header("TensorFlow")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            TensorFlow is an end-to-end open-source platform for machine learning developed by Google.
            It has a comprehensive, flexible ecosystem of tools, libraries, and community resources.
            
            TensorFlow allows developers to build and deploy machine learning models easily, with strong
            support for production deployment on various platforms.
            """)
        
        with col2:
            st.image("https://www.tensorflow.org/images/tf_logo_social.png", width=200)
        
        # Main functionalities
        st.subheader("Main Functionalities")
        
        func_tabs = st.tabs(["Model Building", "Data Pipelines", "Deployment"])
        
        with func_tabs[0]:
            st.markdown("""
            ### Model Building
            
            TensorFlow provides multiple levels of abstraction for building models:
            
            - **Keras API**: High-level, user-friendly API for building and training models
            - **Functional API**: More flexibility for complex model architectures
            - **Subclassing API**: Complete control with custom forward pass logic
            - **Low-level API**: Fine-grained control over operations and gradients
            
            #### Key Features:
            - Eager execution (immediate evaluation of operations)
            - AutoGraph (converting Python control flow to TensorFlow graphs)
            - Custom training loops
            - Model checkpointing
            """)
            
            st.code("""
# TensorFlow model building example
import tensorflow as tf

# Using Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
            """, language="python")
        
        with func_tabs[1]:
            st.markdown("""
            ### Data Pipelines
            
            TensorFlow provides efficient data loading and preprocessing with `tf.data`:
            
            #### Key Features:
            - Efficient loading of large datasets
            - GPU/CPU device placement optimization
            - Parallel data extraction and transformation
            - Prefetching to overlap computation and I/O
            - Batching and shuffling
            """)
            
            st.code("""
# TensorFlow data pipeline example
import tensorflow as tf

# Create a dataset from tensor slices
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# Shuffle and batch the dataset
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Apply transformations
dataset = dataset.map(
    lambda x, y: (data_augmentation(x), y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

# Use dataset in model training
model.fit(dataset, epochs=10)
            """, language="python")
        
        with func_tabs[2]:
            st.markdown("""
            ### Deployment
            
            TensorFlow offers multiple options for deploying models to production:
            
            #### Key Features:
            - TensorFlow Serving for production deployment
            - TensorFlow Lite for mobile and edge devices
            - TensorFlow.js for browser and Node.js
            - TensorFlow Extended (TFX) for production ML pipelines
            - SavedModel format for portable serialization
            """)
            
            st.code("""
# TensorFlow model saving and deployment example
import tensorflow as tf

# Save the model
model.save('my_model')

# Convert to TensorFlow Lite for mobile
converter = tf.lite.TFLiteConverter.from_saved_model('my_model')
tflite_model = converter.convert()

# Save the TF Lite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# For TF Serving (REST API)
# Run in terminal:
# docker run -p 8501:8501 \
#     --mount type=bind,source=/path/to/my_model/,target=/models/my_model \
#     -e MODEL_NAME=my_model -t tensorflow/serving
            """, language="python")
        
        # Live demo (simplified representation)
        st.subheader("Live Demo: TensorFlow Visualization")
        
        st.markdown("### TensorBoard-like visualization")
        
        # Create a simple visualization to represent TensorBoard
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        
        # Training metrics visualization
        epochs = np.arange(1, 21)
        training_loss = 1.0 / (1.0 + 0.1 * epochs) + 0.1 * np.random.randn(20)
        validation_loss = 1.2 / (1.0 + 0.1 * epochs) + 0.1 * np.random.randn(20)
        
        ax[0].plot(epochs, training_loss, 'b-', label='Training loss')
        ax[0].plot(epochs, validation_loss, 'r-', label='Validation loss')
        ax[0].set_title('Model Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        
        # Computational graph visualization (simplified)
        ax[1].set_title('Computation Graph')
        ax[1].axis('off')
        
        # Draw a simple graph structure
        positions = {
            'input': (0.2, 0.5),
            'hidden1': (0.4, 0.7),
            'hidden2': (0.4, 0.3),
            'output': (0.6, 0.5),
            'loss': (0.8, 0.5)
        }
        
        for node, pos in positions.items():
            circle = plt.Circle(pos, 0.05, fill=True, color='skyblue')
            ax[1].add_patch(circle)
            ax[1].text(pos[0], pos[1]-0.1, node, ha='center')
        
        # Add edges
        ax[1].plot([positions['input'][0], positions['hidden1'][0]], 
                [positions['input'][1], positions['hidden1'][1]], 'k-')
        ax[1].plot([positions['input'][0], positions['hidden2'][0]], 
                [positions['input'][1], positions['hidden2'][1]], 'k-')
        ax[1].plot([positions['hidden1'][0], positions['output'][0]], 
                [positions['hidden1'][1], positions['output'][1]], 'k-')
        ax[1].plot([positions['hidden2'][0], positions['output'][0]], 
                [positions['hidden2'][1], positions['output'][1]], 'k-')
        ax[1].plot([positions['output'][0], positions['loss'][0]], 
                [positions['output'][1], positions['loss'][1]], 'k-')
        
        st.pyplot(fig)
    
    # PyTorch tab
    with framework_tabs[1]:
        st.header("PyTorch")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            PyTorch is an open-source machine learning library developed by Facebook's AI Research lab.
            It's known for its flexibility, intuitive design, and dynamic computational graph which
            makes debugging and experimentation easier.
            
            PyTorch is particularly popular in research and academic settings due to its Pythonic nature
            and ease of use.
            """)
        
        with col2:
            st.image("https://pytorch.org/assets/images/pytorch-logo.png", width=200)
        
        # Main functionalities
        st.subheader("Main Functionalities")
        
        func_tabs = st.tabs(["Dynamic Computation", "Neural Network API", "Ecosystem"])
        
        with func_tabs[0]:
            st.markdown("""
            ### Dynamic Computation
            
            PyTorch uses a dynamic computational graph, which provides several advantages:
            
            #### Key Features:
            - Define-by-run approach (graph built at runtime)
            - Easy debugging with standard Python tools
            - Dynamic neural networks with variable length inputs
            - Intuitive control flow (if statements, loops within model)
            - Immediate execution mode
            """)
            
            st.code("""
# PyTorch dynamic computation example
import torch

# Create tensors
x = torch.randn(5, 3)
y = torch.randn(5, 3)

# Dynamic control flow
if x.sum() > 0:
    z = x * y
else:
    z = x + y

# PyTorch tracks operations automatically
z.backward(torch.ones_like(z))  # backpropagation

# Dynamic shapes and sizes
def dynamic_network(x, hidden_size):
    layer1 = torch.nn.Linear(x.shape[1], hidden_size)
    return layer1(x)
            """, language="python")
        
        with func_tabs[1]:
            st.markdown("""
            ### Neural Network API
            
            PyTorch provides the `torch.nn` module for building neural networks:
            
            #### Key Features:
            - Modular design with nn.Module base class
            - Easy customization of network architecture
            - Built-in optimizers and loss functions
            - Automatic differentiation with autograd
            - Custom layer implementations
            """)
            
            st.code("""
# PyTorch neural network example
import torch
import torch.nn as nn
import torch.optim as optim

# Define a neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# Create model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for inputs, targets in data_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            """, language="python")
        
        with func_tabs[2]:
            st.markdown("""
            ### Ecosystem
            
            PyTorch has a rich ecosystem of libraries and tools:
            
            #### Key Features:
            - TorchVision for computer vision
            - TorchText for natural language processing
            - TorchAudio for audio processing
            - PyTorch Lightning for streamlined training
            - TorchServe for model serving
            - Many domain-specific extensions
            """)
            
            st.code("""
# PyTorch ecosystem example
import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl

# TorchVision for datasets and models
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True
)

# Using a pre-trained model
resnet = torchvision.models.resnet18(pretrained=True)

# PyTorch Lightning for structured code
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(28*28, 10)
        
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Train with Lightning
trainer = pl.Trainer(max_epochs=10)
trainer.fit(LitModel(), trainloader)
            """, language="python")
        
        # Live demo (simplified representation)
        st.subheader("Live Demo: PyTorch Visualization")
        
        st.markdown("### Interactive Model Visualization")
        
        # Create a simple visualization to represent PyTorch model inspection
        layers = ["Conv2d", "ReLU", "MaxPool2d", "Conv2d", "ReLU", "MaxPool2d", "Linear", "ReLU", "Linear"]
        params = [1280, 0, 0, 51200, 0, 0, 40960, 0, 650]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Create a bar chart for model parameters
        bars = ax.bar(layers, params, color='#EE4C2C')
        ax.set_title('Model Parameters by Layer')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Parameters')
        ax.set_yscale('log')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Keras tab
    with framework_tabs[2]:
        st.header("Keras")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Keras is a high-level neural networks API, originally developed as an independent library
            but now integrated with TensorFlow. It's designed to enable fast experimentation with deep
            neural networks and focuses on being user-friendly, modular, and extensible.
            
            Keras allows for easy and fast prototyping with its simple, consistent interface
            and minimal code requirements for common use cases.
            """)
        
        with col2:
            st.image("https://keras.io/img/logo.png", width=200)
        
        # Main functionalities
        st.subheader("Main Functionalities")
        
        func_tabs = st.tabs(["Model Building", "Training API", "Pre-trained Models"])
        
        with func_tabs[0]:
            st.markdown("""
            ### Model Building
            
            Keras offers multiple APIs for building models:
            
            #### Key Features:
            - Sequential API for straightforward stacking of layers
            - Functional API for complex model architectures
            - Model subclassing for custom behaviors
            - Built-in layers for most use cases
            - Easy custom layer creation
            """)
            
            st.code("""
# Keras model building example
from tensorflow import keras

# Sequential API
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Functional API
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(512, activation='relu')(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
            """, language="python")
        
        with func_tabs[1]:
            st.markdown("""
            ### Training API
            
            Keras provides a simple API for training and evaluation:
            
            #### Key Features:
            - fit() method for training
            - evaluate() method for testing
            - predict() method for inference
            - Callbacks for monitoring and intervention
            - Learning rate scheduling
            - Early stopping
            """)
            
            st.code("""
# Keras training API example
from tensorflow import keras
import numpy as np

# Sample data
x_train = np.random.random((1000, 784))
y_train = np.random.randint(10, size=(1000,))
x_val = np.random.random((200, 784))
y_val = np.random.randint(10, size=(200,))

# Create callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='model_checkpoint.h5', save_best_only=True
    ),
    keras.callbacks.TensorBoard(log_dir='./logs')
]

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

# Evaluate the model
test_scores = model.evaluate(x_val, y_val)
print(f"Test loss: {test_scores[0]}")
print(f"Test accuracy: {test_scores[1]}")

# Make predictions
predictions = model.predict(x_val[:5])
            """, language="python")
        
        with func_tabs[2]:
            st.markdown("""
            ### Pre-trained Models
            
            Keras provides a wide range of pre-trained models:
            
            #### Key Features:
            - Models pre-trained on ImageNet
            - Easy fine-tuning for transfer learning
            - Applications module with state-of-the-art architectures
            - Consistent interface across different models
            - Weights-only or full model loading options
            """)
            
            st.code("""
# Keras pre-trained models example
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained model
model = ResNet50(weights='imagenet')

# Load and preprocess image
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make prediction
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

# Fine-tuning
base_model = VGG16(weights='imagenet', include_top=False, 
                  input_shape=(224, 224, 3))
                  
# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False
    
# Add new classification head
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(256, activation='relu')(x)
predictions = keras.layers.Dense(10, activation='softmax')(x)

# Create new model
model = keras.Model(inputs=base_model.input, outputs=predictions)

# Compile and train for new task
model.compile(optimizer='adam', loss='categorical_crossentropy')
            """, language="python")
        
        # Live demo (simplified representation)
        st.subheader("Live Demo: Keras Model Visualization")
        
        # Create a stylized model summary
        st.markdown("### Model Summary & Architecture")
        
        model_summary = """
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 512)               401,920   
_________________________________________________________________
dropout (Dropout)            (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               131,328   
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2,570     
=================================================================
Total params: 535,818
Trainable params: 535,818
Non-trainable params: 0
_________________________________________________________________
"""
        
        st.code(model_summary)
        
        # Create a visual representation of model training
        st.markdown("### Training Progress")
        
        # Generate sample training data
        epochs = np.arange(1, 11)
        accuracy = 0.5 + 0.4 * (1 - np.exp(-0.3 * epochs)) + 0.01 * np.random.randn(10)
        val_accuracy = 0.4 + 0.35 * (1 - np.exp(-0.25 * epochs)) + 0.01 * np.random.randn(10)
        
        # Create training progress plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, accuracy, 'bo-', label='Training accuracy')
        ax.plot(epochs, val_accuracy, 'ro-', label='Validation accuracy')
        ax.set_title('Model Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)

def models_page():
    st.title("Applied Neural Network Models")
    
    st.markdown("""
    This section showcases practical implementations of neural networks for specific tasks.
    Each model demonstrates how neural network architectures can be applied to solve real-world problems.
    """)
    
    # Create tabs for different model types
    model_tabs = st.tabs(["Image Classifier (MNIST)", "Sentiment Analysis (NLP)", "Simple Chatbot"])
    
    # Image Classifier tab
    with model_tabs[0]:
        st.header("Image Classifier (MNIST)")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            MNIST is a dataset of handwritten digits that is commonly used for training image processing systems.
            It consists of 70,000 28x28 pixel grayscale images of handwritten digits (0-9).
            
            This model demonstrates how convolutional neural networks can be used for image classification tasks.
            """)
            
            st.markdown("### Key Features:")
            st.markdown("""
            - Convolutional layers for feature extraction
            - Pooling layers for dimensionality reduction
            - Dropout for regularization
            - Dense layers for classification
            - Real-time digit recognition
            """)
        
        with col2:
            st.image("https://miro.medium.com/max/941/1*Ug8fnEAv3jQRKVNx6__RYA.png", 
                    caption="MNIST Dataset Sample")
        
        # Live demo section
        st.subheader("Live Demo: MNIST Classifier")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Draw a digit (0-9)")
            
            # In a real implementation, this would be a canvas for drawing
            # For this demo, we'll simulate with pre-made digits
            digit_options = st.selectbox(
                "Select a digit to classify:",
                ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            )
            
            classify_btn = st.button("Classify Digit")
        
        with col2:
            st.markdown("### Classification Results")
            
            # Generate a sample confusion matrix / prediction visualization
            if classify_btn:
                # Sample digit images (in a real app, this would be from the canvas)
                digit_images = {
                    "0": "https://www.researchgate.net/profile/Jose-Garcia-207/publication/321586653/figure/fig7/AS:600114292436997@1520113378445/Examples-of-digit-0-in-MNIST-dataset.png",
                    "1": "https://i.stack.imgur.com/vXYpA.png",
                    "2": "https://www.researchgate.net/profile/Addisson-Salazar/publication/334269094/figure/fig2/AS:837624301936640@1576764366844/Image-examples-of-digit-2-from-MNIST-dataset-2.jpg",
                    "3": "https://www.researchgate.net/profile/Addisson-Salazar/publication/334269094/figure/fig3/AS:837624301940736@1576764366886/Image-examples-of-digit-3-from-MNIST-dataset-2.jpg",
                    "4": "https://www.researchgate.net/publication/334269094/figure/fig4/AS:837624301944833@1576764366926/Image-examples-of-digit-4-from-MNIST-dataset-2.jpg",
                    "5": "https://www.researchgate.net/profile/Addisson-Salazar/publication/334269094/figure/fig5/AS:837624301948930@1576764366974/Image-examples-of-digit-5-from-MNIST-dataset-2.jpg",
                    "6": "https://www.researchgate.net/profile/Addisson-Salazar/publication/334269094/figure/fig6/AS:837624301953024@1576764367005/Image-examples-of-digit-6-from-MNIST-dataset-2.jpg",
                    "7": "https://www.researchgate.net/profile/Addisson-Salazar/publication/334269094/figure/fig7/AS:837624301957121@1576764367043/Image-examples-of-digit-7-from-MNIST-dataset-2.jpg",
                    "8": "https://www.researchgate.net/profile/Addisson-Salazar/publication/334269094/figure/fig8/AS:837624301961217@1576764367074/Image-examples-of-digit-8-from-MNIST-dataset-2.jpg",
                    "9": "https://www.researchgate.net/profile/Addisson-Salazar/publication/334269094/figure/fig9/AS:837624301965312@1576764367113/Image-examples-of-digit-9-from-MNIST-dataset-2.jpg"
                }
                
                # Show the selected digit image
                st.image(digit_images[digit_options], width=200)
                
                # Simulated prediction probabilities
                import numpy as np
                
                # Create sample predictions with correct digit having highest probability
                predictions = np.random.rand(10) * 0.2
                predictions[int(digit_options)] = 0.7 + np.random.rand() * 0.3
                predictions = predictions / np.sum(predictions)  # Normalize to sum to 1
                
                # Display prediction probabilities
                fig, ax = plt.subplots(figsize=(8, 5))
                classes = list(range(10))
                ax.bar(classes, predictions)
                ax.set_xlabel('Digit Class')
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Probabilities')
                ax.set_xticks(classes)
                
                # Highlight the correct/predicted class
                bars = ax.patches
                predicted_class = np.argmax(predictions)
                bars[predicted_class].set_color('red')
                
                st.pyplot(fig)
                
                st.markdown(f"Predicted digit: **{predicted_class}**")
        
        # Code implementation
        st.subheader("Implementation")
        st.code("""
# MNIST Image Classification with CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Build the CNN model
model = models.Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_images, train_labels, 
    epochs=10, 
    validation_data=(test_images, test_labels)
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc:.3f}')

# Function to predict a single image
def predict_digit(img):
    # Preprocess the image
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255
    
    # Get prediction probabilities
    predictions = model.predict(img)
    
    # Get predicted class
    predicted_class = np.argmax(predictions[0])
    
    return predicted_class, predictions[0]
        """, language="python")
    
    # Sentiment Analysis tab
    with model_tabs[1]:
        st.header("Sentiment Analysis (NLP)")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            Sentiment analysis is the task of determining the emotional tone behind a piece of text.
            This model uses natural language processing techniques to classify text as positive, negative, or neutral.
            
            The model demonstrates how recurrent neural networks can be used for text classification tasks.
            """)
            
            st.markdown("### Key Features:")
            st.markdown("""
            - Word embeddings for text representation
            - Bidirectional LSTM layers for sequence processing
            - Attention mechanisms for focusing on important words
            - Dense layers for classification
            - Real-time sentiment prediction
            """)
        
        with col2:
            st.image("https://cdn-images-1.medium.com/max/1600/1*BU8LHmLuTDf-rX2rJngkLA.png", 
                    caption="Sentiment Analysis Architecture")
        
        # Live demo section
        st.subheader("Live Demo: Sentiment Analyzer")
        
        # Text input for sentiment analysis
        user_text = st.text_area(
            "Enter text to analyze sentiment:",
            "I really enjoyed using this neural network demo. It's a well-designed application with great examples!"
        )
        
        analyze_btn = st.button("Analyze Sentiment")
        
        if analyze_btn and user_text:
            # Simulate sentiment analysis with a simple rule-based approach
            import re
            
            # Simple positive and negative word lists
            positive_words = ['good', 'great', 'awesome', 'excellent', 'love', 'happy', 'enjoy', 'enjoyed',
                             'well', 'positive', 'nice', 'beautiful', 'best', 'perfect', 'wonderful']
            negative_words = ['bad', 'awful', 'terrible', 'hate', 'dislike', 'poor', 'horrible', 'worst',
                             'negative', 'ugly', 'difficult', 'annoying', 'disappointing', 'failure']
            
            # Tokenize text
            words = re.findall(r'\w+', user_text.lower())
            
            # Count sentiment words
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            # Determine sentiment score (-1 to 1)
            total_count = len(words)
            sentiment_score = (pos_count - neg_count) / max(1, total_count)
            
            # Normalize to 0-1 range
            normalized_score = (sentiment_score + 1) / 2
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(8, 3))
            sentiment_labels = ['Negative', 'Neutral', 'Positive']
            
            # Determine sentiment category
            if normalized_score < 0.4:
                sentiment_cat = 'Negative'
                color = 'red'
            elif normalized_score > 0.6:
                sentiment_cat = 'Positive'
                color = 'green'
            else:
                sentiment_cat = 'Neutral'
                color = 'blue'
            
            # Create visualization
            ax.barh(0, normalized_score, height=0.5, color=color)
            ax.barh(0, 1-normalized_score, height=0.5, left=normalized_score, color='lightgray')
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
            
            # Add markers for sentiment categories
            ax.text(0.1, -0.5, 'Negative', fontsize=10, ha='center')
            ax.text(0.5, -0.5, 'Neutral', fontsize=10, ha='center')
            ax.text(0.9, -0.5, 'Positive', fontsize=10, ha='center')
            
            # Add score marker
            ax.scatter(normalized_score, 0, color='black', s=100, zorder=5)
            
            st.pyplot(fig)
            
            # Display result
            st.markdown(f"### Sentiment: **{sentiment_cat}**")
            st.markdown(f"Confidence: {normalized_score:.2%}")
        
        # Code implementation
        st.subheader("Implementation")
        st.code("""
# Sentiment Analysis with LSTM
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

# Sample data for sentiment analysis
texts = [
    "I love this product, it's amazing!",
    "This is a great app with excellent features",
    "The service was terrible and the staff was rude",
    "I didn't like the quality, very disappointing",
    # ... more training examples ...
]

labels = [1, 1, 0, 0]  # 1 for positive, 0 for negative

# Tokenize text
max_features = 10000  # Maximum number of words to keep
max_len = 100  # Max sequence length

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x_data = pad_sequences(sequences, maxlen=max_len)
y_data = np.array(labels)

# Build the model
embedding_dim = 100

model = Sequential([
    # Embedding layer
    Embedding(max_features, embedding_dim, input_length=max_len),
    
    # Bidirectional LSTM
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    
    # Classification layers
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_data, y_data,
    epochs=10,
    validation_split=0.2,
    batch_size=32
)

# Function to predict sentiment
def predict_sentiment(text):
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    
    # Get prediction
    prediction = model.predict(padded)[0][0]
    
    # Classify sentiment
    if prediction < 0.4:
        return "Negative", prediction
    elif prediction > 0.6:
        return "Positive", prediction
    else:
        return "Neutral", prediction
        """, language="python")
    
    # Chatbot tab
    with model_tabs[2]:
        st.header("Simple Chatbot")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            This is a sequence-to-sequence neural network model that can understand and generate text 
            for simple conversations. It demonstrates how encoder-decoder architectures with attention 
            mechanisms can be used for language generation tasks.
            """)
            
            st.markdown("### Key Features:")
            st.markdown("""
            - Encoder-decoder architecture
            - Attention mechanism for context understanding
            - Word embeddings for text representation
            - Beam search for response generation
            - Context-aware responses
            """)
        
        with col2:
            st.image("https://miro.medium.com/max/1200/1*DQxj8K93jjN-4ltruSQ8iw.jpeg", 
                    caption="Seq2Seq Chatbot Architecture")
        
        # Live demo section
        st.subheader("Live Demo: Simple Chatbot")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [
                {"role": "bot", "content": "Hello! I'm a neural network chatbot. How can I help you today?"}
            ]
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div style='text-align: right; color: #4b8bbe; padding: 10px; border-radius: 5px;'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color: #2e2e2e; color: #ffffff; padding: 10px; border-radius: 5px;'><strong>Bot:</strong> {message['content']}</div>", unsafe_allow_html=True)
        
        # User input
        user_input = st.text_input("Type your message:", key="user_message")
        
        # Send button
        send_btn = st.button("Send")
        
        if send_btn and user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Simple response logic
            responses = {
                "hello": "Hello! Nice to meet you.",
                "how are you": "I'm just a neural network, but I'm functioning well. How are you?",
                "what can you do": "I can have simple conversations with you, answer questions about neural networks, and demonstrate how a chatbot works.",
                "goodbye": "Goodbye! It was nice chatting with you.",
                "thanks": "You're welcome! Is there anything else I can help with?",
                "neural network": "Neural networks are computing systems inspired by the human brain. They're the core technology behind many AI applications!",
                "machine learning": "Machine learning is a field where computers learn from data without explicit programming.",
                "deep learning": "Deep learning is a subset of machine learning that uses neural networks with many layers to learn from data."
            }
            
            # Find best matching response
            best_match = None
            best_score = 0
            
            for key in responses:
                if key in user_input.lower():
                    if len(key) > best_score:
                        best_score = len(key)
                        best_match = key
            
            # Generate response
            if best_match:
                bot_response = responses[best_match]
            else:
                bot_response = "That's interesting! Tell me more or ask me something about neural networks."
            
            # Add bot response to history
            st.session_state.chat_history.append({"role": "bot", "content": bot_response})
            
            # Rerun to update the UI
            st.rerun()
        
        # Code implementation
        st.subheader("Implementation")
        st.code("""
# Sequence-to-Sequence Chatbot
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample conversation pairs
conversations = [
    ("Hello", "Hi there!"),
    ("How are you?", "I'm good, thank you. How about you?"),
    ("What is your name?", "I'm a neural network chatbot."),
    ("Tell me about neural networks", "Neural networks are computing systems inspired by the human brain."),
    # ... more conversation pairs ...
]

# Prepare data
input_texts = [pair[0] for pair in conversations]
target_texts = ['\t' + pair[1] + '\n' for pair in conversations]  # Add start/end tokens

# Create tokenizers
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(target_texts)

# Convert text to sequences
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

# Pad sequences
max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in target_sequences)

encoder_inputs = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
decoder_inputs = pad_sequences(target_sequences, maxlen=max_target_len, padding='post')

# Create decoder target data (shifted by one)
decoder_targets = np.zeros_like(decoder_inputs)
decoder_targets[:, :-1] = decoder_inputs[:, 1:]

# Define vocabulary sizes
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

# Define embedding dimension
embedding_dim = 128

# Encoder
encoder_inputs_tensor = Input(shape=(max_input_len,))
encoder_embedding = Embedding(input_vocab_size, embedding_dim)(encoder_inputs_tensor)
encoder_lstm = LSTM(256, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs_tensor = Input(shape=(max_target_len,))
decoder_embedding = Embedding(target_vocab_size, embedding_dim)(decoder_inputs_tensor)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention mechanism
attention = Attention()
context_vector = attention([decoder_outputs, encoder_outputs])

# Combine context and decoder output
decoder_combined_context = tf.concat([context_vector, decoder_outputs], axis=-1)

# Output layer
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_combined_context)

# Create and compile the model
model = Model([encoder_inputs_tensor, decoder_inputs_tensor], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
# model.fit([encoder_inputs, decoder_inputs], decoder_targets, batch_size=32, epochs=100, validation_split=0.2)

# Inference model
# (Encoder part remains the same)
encoder_model = Model(encoder_inputs_tensor, [encoder_outputs] + encoder_states)

# Decoder for inference
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_enc_output = Input(shape=(max_input_len, 256))

decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]

context_vector2 = attention([decoder_outputs2, decoder_enc_output])
decoder_combined_context2 = tf.concat([context_vector2, decoder_outputs2], axis=-1)
decoder_outputs2 = decoder_dense(decoder_combined_context2)

decoder_model = Model(
    [decoder_inputs_tensor, decoder_enc_output] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

# Function to generate responses
def generate_response(input_text):
    # Tokenize and pad input
    input_seq = input_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
    
    # Encode input
    enc_outputs, h, c = encoder_model.predict(input_seq)
    
    # Start with start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['\t']
    
    # Generate response
    decoded_sentence = ''
    stop_condition = False
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, enc_outputs, h, c])
        
        # Sample token with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        
        # Convert token to word
        for word, index in target_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break
        
        # Append to response
        if sampled_word == '\n' or len(decoded_sentence) > max_target_len:
            stop_condition = True
        else:
            decoded_sentence += sampled_word + ' '
        
        # Update target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
    
    return decoded_sentence
        """, language="python")

if __name__ == "__main__":
    main()

# Deep Learning Portfolio

A Streamlit-based web application showcasing deep learning models and visualizations.

## Features

- Interactive deep learning model demonstrations
- Data visualizations
- Neural network architecture exploration
- Code examples and explanations

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── neural_networks.py  # Neural network implementations
├── visualizations.py   # Data visualization utilities
├── utils.py           # Helper functions
├── code_examples.py   # Example code snippets
├── assets/            # Static assets and resources
└── requirements.txt   # Python dependencies
```

## Setup Instructions

### Local Development

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DeepLearning-portfolio-main
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

### Docker Deployment

The application can be deployed using Docker:

1. Build the Docker image:
   ```bash
   docker build -t deep-learning-portfolio .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 deep-learning-portfolio
   ```

## Deployment

This application is configured for deployment on Render:

1. Push your code to a Git repository
2. Create a new Web Service on Render
3. Connect your repository
4. Configure the following settings:
   - Build Command: `docker build -t deep-learning-portfolio .`
   - Start Command: `docker run -p 8501:8501 deep-learning-portfolio`
   - Environment: `PYTHONUNBUFFERED=1`

## Dependencies

- streamlit
- numpy
- matplotlib
- scikit-learn
- tensorflow-cpu==2.10.0

## License

[Add your license information here]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 


# Windows command
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
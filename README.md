# DTC Driver Behavior Analysis

A real-time driver behavior analysis application for Delhi Transport Corporation (DTC) that uses machine learning to categorize driver behavior based on various parameters.

## Features

- Real-time driver behavior monitoring and classification
- Dynamic data generation for testing
- Interactive dashboard with live updates
- Machine learning model for behavior classification
- Performance metrics tracking

## Installation

1. Clone this repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser with the following features:

- Real-time speed and acceleration monitoring
- Behavior category distribution
- Model performance metrics
- Recent driver data table
- Controls for data generation and update intervals

## Components

- `app.py`: Main Streamlit application
- `data_generator.py`: Generates realistic driver behavior data
- `ml_model.py`: TensorFlow-based machine learning model
- `requirements.txt`: Project dependencies

## Data Parameters

- Speed (km/h)
- Acceleration (m/s²)
- Brake Intensity
- Route Location
- Behavior Score
- Category (Safe, Moderate, Risky)

## Model Performance

The machine learning model evaluates driver behavior based on:
- Accuracy
- Precision
- Recall

## License

MIT License

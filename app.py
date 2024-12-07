import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_generator import DriverDataGenerator
from ml_model import DriverBehaviorModel
import time

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = DriverDataGenerator()
if 'model' not in st.session_state:
    st.session_state.model = DriverBehaviorModel()
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

def update_real_time_data():
    """Update data with new records"""
    new_record = st.session_state.data_generator.generate_single_record()
    new_df = pd.DataFrame([new_record])
    
    if st.session_state.data.empty:
        st.session_state.data = new_df
    else:
        st.session_state.data = pd.concat([st.session_state.data, new_df]).tail(100)

def train_model():
    """Train the model with current data"""
    if len(st.session_state.data) > 50:
        with st.spinner('Training model...'):
            st.session_state.model.train(st.session_state.data)
            st.session_state.metrics = st.session_state.model.evaluate(st.session_state.data)
        st.success('Model trained successfully!')

# Streamlit UI
st.title('DTC Driver Behavior Analysis')

# Sidebar
st.sidebar.header('Controls')
if st.sidebar.button('Generate New Data'):
    new_data = st.session_state.data_generator.generate_batch(100)
    st.session_state.data = new_data
    train_model()

update_interval = st.sidebar.slider('Update Interval (seconds)', 1, 10, 2)

# Main dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader('Real-time Speed and Acceleration')
    if not st.session_state.data.empty:
        # Speed line chart
        fig_speed = px.line(st.session_state.data.tail(20), 
                           x='timestamp', 
                           y='speed',
                           title='Vehicle Speed (km/h)')
        st.plotly_chart(fig_speed, use_container_width=True)

with col2:
    st.subheader('Behavior Category Distribution')
    if not st.session_state.data.empty:
        # Category distribution
        category_counts = st.session_state.data['category'].value_counts()
        fig_cat = px.bar(x=category_counts.index, 
                        y=category_counts.values,
                        title='Behavior Categories',
                        color=category_counts.index)
        st.plotly_chart(fig_cat, use_container_width=True)

# Model metrics
if st.session_state.metrics:
    st.subheader('Model Performance Metrics')
    col1, col2, col3 = st.columns(3)
    col1.metric('Accuracy', f"{st.session_state.metrics['accuracy']:.2%}")
    col2.metric('Precision', f"{st.session_state.metrics['precision']:.2%}")
    col3.metric('Recall', f"{st.session_state.metrics['recall']:.2%}")

# Data table
st.subheader('Recent Driver Data')
if not st.session_state.data.empty:
    st.dataframe(st.session_state.data.tail(10).style.format({
        'speed': '{:.2f}',
        'acceleration': '{:.2f}',
        'brake_intensity': '{:.2f}',
        'behavior_score': '{:.2f}'
    }))

# Auto-update data
if st.sidebar.checkbox('Enable Real-time Updates', value=True):
    update_real_time_data()
    time.sleep(update_interval)
    st.rerun()

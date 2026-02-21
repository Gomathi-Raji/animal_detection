import streamlit as st
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import tempfile
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import shutil
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import io
from pydub import AudioSegment
from streamlit_option_menu import option_menu

class LayerScale(tf.keras.layers.Layer):
    def __init__(self, dim, init_values=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.scale = self.add_weight(
            name="scale",
            shape=(dim,),
            initializer=tf.keras.initializers.Constant(init_values),
            trainable=True
        )

    def call(self, inputs):
        return inputs * self.scale

st.set_page_config(
    page_title="Generic Audio Classifier",
    page_icon="🔊",
    layout="wide"
)

SAMPLE_RATE = 22050
DURATION = 5  # seconds
RECORDING_PATH = "recorded_audio.wav"
DATASET_PATH = "DATASET_FLAC"
MODEL_PATHS = {
    "NasNet Mobile": "NasNet_Mobile",
    "DualNet CX": "DualNet_CX",
    "DualNet Xpert": "DualNet_Xpert",
    "EfficientNet V2 B0": "EfficientNet_V2_B0"
}

def get_parameters():
    return {
        'data_dir': 'DATASET',
        'sample_rate': 22050,
        'duration': 5,  # seconds
        'n_mfcc': 40,
        'n_mels': 128,
        'n_fft': 2048,
        'hop_length': 512,
        'batch_size': 32,
        'epochs': 20,
        'validation_split': 0.2,
        'random_state': 42,
        'num_test_samples': 10  # Number of random samples to test
    }

def extract_features(file_path, params):
    try:
        audio, sr = librosa.load(file_path, sr=params['sample_rate'], duration=params['duration'])

        if len(audio) < params['sample_rate'] * params['duration']:
            padding = params['sample_rate'] * params['duration'] - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=params['sample_rate'],
            n_mfcc=params['n_mfcc'],
            n_fft=params['n_fft'],
            hop_length=params['hop_length']
        )

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=params['sample_rate'],
            n_mels=params['n_mels'],
            n_fft=params['n_fft'],
            hop_length=params['hop_length']
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)

        return {
            'mfccs': mfccs,
            'mel_spec': mel_spec_db,
            'audio': audio,
            'sr': sr
        }

    except Exception as e:
        st.error(f"Error extracting features from {file_path}: {e}")
        return None


def load_audio_classifier(model_path):

    try:
        model_file = os.path.join(model_path, 'audio_classifier.h5')
        metadata_file = os.path.join(model_path, 'metadata.npy')
        
        if not os.path.exists(model_file) or not os.path.exists(metadata_file):
            st.error(f"Model files not found in {model_path}. Please check the path.")
            return None, None
            
        model = load_model(model_file)
        metadata = np.load(metadata_file, allow_pickle=True).item()
        
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict_with_filters(features, model, metadata, selected_categories, selected_subcategories):
    if features is None:
        return "Error extracting features from the audio file."

    mfccs = np.array([features['mfccs']])
    mfccs = mfccs[..., np.newaxis]

    predictions = model.predict(mfccs)
    
    if isinstance(metadata['category_mapping'], dict):
        category_map = metadata['category_mapping']
    else:
        category_map = {i: cat for i, cat in enumerate(metadata['category_mapping'])}
        
    if isinstance(metadata['subcategory_mapping'], dict):
        subcategory_map = metadata['subcategory_mapping']
    else:
        subcategory_map = {i: subcat for i, subcat in enumerate(metadata['subcategory_mapping'])}
    
    category_probs = predictions[0][0].copy()
    subcategory_probs = predictions[1][0].copy()
    
    for i in range(len(category_probs)):
        if category_map[i] not in selected_categories:
            category_probs[i] = 0
    
    for i in range(len(subcategory_probs)):
        if subcategory_map[i] not in selected_subcategories:
            subcategory_probs[i] = 0
    
    if np.max(category_probs) == 0:
        category = "No selected category matches the audio"
        category_confidence = 0
    else:
        category_idx = np.argmax(category_probs)
        category = category_map[category_idx]
        category_confidence = float(category_probs[category_idx])
    
    if np.max(subcategory_probs) == 0:
        subcategory = "No selected subcategory matches the audio"
        subcategory_confidence = 0
    else:
        subcategory_idx = np.argmax(subcategory_probs)
        subcategory = subcategory_map[subcategory_idx]
        subcategory_confidence = float(subcategory_probs[subcategory_idx])

    return {
        'category': category,
        'subcategory': subcategory,
        'category_confidence': category_confidence,
        'subcategory_confidence': subcategory_confidence,
        'category_probs': {category_map[i]: float(predictions[0][0][i])
                           for i in range(len(category_probs))},
        'subcategory_probs': {subcategory_map[i]: float(predictions[1][0][i])
                             for i in range(len(subcategory_probs))}
    }
    
def record_audio(duration=5, fs=22050):
    st.write("🎙️ Recording...")
    progress_bar = st.progress(0)
    
    animation_placeholder = st.empty()
    
    animation_frames = ["🔴", "⭕", "⚪", "⭕"]
    
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    
    for i in range(duration * 10):  
        time.sleep(0.1)
        progress_bar.progress((i + 1) / (duration * 10))
        animation_placeholder.markdown(f"## {animation_frames[i % len(animation_frames)]} Recording...")
    
    sd.wait()  
    animation_placeholder.empty()
    st.success("✅ Recording completed!")
    
    sf.write(RECORDING_PATH, recording, fs)
    return RECORDING_PATH

def add_data_to_dataset(audio_file, category, subcategory, save_path):
    try:
        # Create directories if they don't exist
        category_dir = os.path.join(save_path, category)
        subcategory_dir = os.path.join(category_dir, subcategory)
        
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        
        if not os.path.exists(subcategory_dir):
            os.makedirs(subcategory_dir)
        
        # Generate timestamp and create filename with .flac extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sample_{timestamp}.flac"  # Changed extension to .flac
        destination = os.path.join(subcategory_dir, filename)
        
        # If the source file is not FLAC, convert it
        if not audio_file.lower().endswith('.flac'):
            # Convert to FLAC using AudioSegment
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_file)
            audio.export(destination, format="flac")
        else:
            # Simply copy the file if it's already FLAC
            shutil.copy(audio_file, destination)
        
        return True, destination
    except Exception as e:
        return False, str(e)

def visualize_audio_waveform(audio, sr):
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    return fig

def visualize_mfcc(mfccs):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    ax.set_title('MFCC')
    fig.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    return fig

def visualize_mel_spectrogram(mel_spec):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, ax=ax)
    ax.set_title('Mel Spectrogram')
    fig.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    return fig

def plot_probability_distribution(probabilities, title):
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    
    sorted_indices = np.argsort(values)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    fig = px.bar(
        x=sorted_labels,
        y=sorted_values,
        labels={'x': '', 'y': 'Probability'},
        title=title
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

def get_model_info(model, metadata):
    # Get model summary as string
    summary_str = []
    model.summary(print_fn=lambda x: summary_str.append(x))
    
    model_layers = []
    for layer in model.layers:
        try:
            output_shape = str(layer.output.shape)  # For newer TensorFlow versions
        except AttributeError:
            output_shape = "N/A"
            
        model_layers.append({
            "Layer Name": layer.name,
            "Layer Type": layer.__class__.__name__,
            "Output Shape": output_shape,
            "Param #": f"{layer.count_params():,}"
        })
        
    model_df = pd.DataFrame(model_layers)
    
    if isinstance(metadata['category_mapping'], dict):
        categories = [metadata['category_mapping'][i] for i in sorted(metadata['category_mapping'].keys())]
    else:
        categories = metadata['category_mapping']
        
    if isinstance(metadata['subcategory_mapping'], dict):
        subcategories = [metadata['subcategory_mapping'][i] for i in sorted(metadata['subcategory_mapping'].keys())]
    else:
        subcategories = metadata['subcategory_mapping']
    
    info = {
        "Model Architecture": model.__class__.__name__,
        "Total Parameters": f"{model.count_params():,}",
        "Input Shape": str(metadata['input_shape']),
        "Categories": ", ".join(categories),
        "Subcategories": ", ".join(subcategories),
        "Sample Rate": metadata['params']['sample_rate'],
        "Duration": f"{metadata['params']['duration']} seconds",
        "MFCC Features": metadata['params']['n_mfcc'],
        "Mel Bands": metadata['params']['n_mels']
    }
    
    return info, "\n".join(summary_str), model_df

def get_categories_and_subcategories():
    categories = []
    subcategories = {}
    dataset_path = DATASET_PATH
    
    if os.path.exists(dataset_path):
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if os.path.isdir(category_path):
                categories.append(category)
                subcategories[category] = []
                
                for subcategory in os.listdir(category_path):
                    subcategory_path = os.path.join(category_path, subcategory)
                    if os.path.isdir(subcategory_path):
                        subcategories[category].append(subcategory)
    
    return categories, subcategories

if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'current_metadata' not in st.session_state:
    st.session_state.current_metadata = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

st.title("🔊 GAC - Generic Audio Classifier")

st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem !important;
                color: #1E88E5;
                font-weight: 700;
            }
            .sub-header {
                font-size: 1.5rem !important;
                color: #424242;
                font-weight: 500;
            }
            .card {
                border-radius: 10px;
                padding: 2px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 2px;
            }
            .stat-card {
                background-color: #f1f7ff;
                border-left: 5px solid #1E88E5;
            }
            .model-card {
                background-color: #f5f5f5;
                transition: transform 0.3s;
            }
            .model-card:hover {
                transform: translateY(-5px);
            }
            .feature-icon {
                font-size: 2rem;
                margin-bottom: 2px;
                color: #1E88E5;
            }
        </style>
        """, unsafe_allow_html=True)

with st.sidebar:
    app_mode = option_menu(
        "Choose a mode",
        ["Home", "Classify Audio", "Add Training Data", "Model Information"],  
        icons=["house", "soundwave", "folder-plus", "info-circle"],  
        menu_icon="cast",  
        default_index=0,  
    )
    
selected_model_name = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))
selected_model_path = MODEL_PATHS[selected_model_name]

if st.session_state.current_model is None or selected_model_path != st.session_state.current_model_path:
    
    with st.spinner("Loading model..."):
        model, metadata = load_audio_classifier(selected_model_path)
        if model is not None and metadata is not None:
            st.session_state.current_model = model
            st.session_state.current_metadata = metadata
            st.session_state.current_model_path = selected_model_path
            st.sidebar.success(f"Model {selected_model_name} loaded successfully!")
        else:
            st.sidebar.error(f"Failed to load model {selected_model_name}")

if app_mode == "Home":
    
    st.markdown('<p>A Powerful audio classification using state-of-the-art deep learning models</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 What Can You Do?")
        st.markdown("""
        This application allows you to classify audio files into various categories and subcategories using advanced machine learning models.
        
        - **Upload** your audio files for instant classification
        - **Record** audio directly through your microphone
        - **Visualize** classification results with detailed analytics
        - **Contribute** to the dataset by adding new labeled audio files
        - **Explore** the existing dataset structure and examples
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 Powered by Advanced Models")
        
        model_cols = st.columns(4)
        
        models = [
            {"name": "NASNet Mobile", "desc": "Neural Architecture Search Network optimized for mobile", "acc": "95 %"},
            {"name": "EfficientNet V2 B0", "desc": "Optimized CNN with balanced performance", "acc": "87 %"},
            {"name": "DualNet CX", "desc": "Dual-pathway network for contextual features", "acc": "99 %"},
            {"name": "DualNet Xpert", "desc": "Expert system with dual feature extraction", "acc": "98 %"}
        ]
        
        for i, model in enumerate(models):
            with model_cols[i]:
                st.markdown(f'<div class="card model-card">', unsafe_allow_html=True)
                st.markdown(f"**{model['name']}**")
                st.markdown(f"{model['desc']}")
                st.markdown(f"**Accuracy:** {model['acc']}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        stats = {'exists': True, 'categories': 4, 'subcategories': 23, 'files': 23303}
        
        st.markdown('<div class="card stat-card">', unsafe_allow_html=True)
        st.subheader("📊 Dataset Overview")
        
        if stats["exists"]:
            st.metric("Audio Files", stats["files"])
            st.metric("Categories", stats["categories"])
            st.metric("Subcategories", stats["subcategories"])
        else:
            st.warning("No dataset found. Start by adding audio files.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Classification Visualization")
        
        sample_results = {
            "Category": ["Animals", "Birds", "Environment", "Vehicles"],
            "Confidence": [0.91, 0.94, 0.96, 0.96]  
        }
        
        sample_df = pd.DataFrame(sample_results)
        
        fig = px.bar(sample_df, x="Category", y="Confidence", color="Confidence",
                    color_continuous_scale=["#90CAF9", "#1E88E5", "#0D47A1"],
                    title="Classification Results")
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Key features
    st.markdown('<h2 class="sub-header">Key Features</h2>', unsafe_allow_html=True)
    
    feature_cols = st.columns(4)
    
    features = [
        {"icon": "🎙️", "title": "Audio Processing", "desc": "Process various audio formats with intelligent feature extraction"},
        {"icon": "🔄", "title": "Real-time Classification", "desc": "Get instant predictions with high accuracy and precision"},
        {"icon": "📊", "title": "Advanced Visualization", "desc": "See detailed analytics and confidence scores for each prediction"},
        {"icon": "🔍", "title": "Dynamic Dataset", "desc": "Flexible system that grows and improves with new data"}
    ]
    
    for i, feature in enumerate(features):
        with feature_cols[i]:
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="feature-icon">{feature["icon"]}</div>', unsafe_allow_html=True)
            st.markdown(f"**{feature['title']}**")
            st.markdown(f"{feature['desc']}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Call to action
    st.markdown('<div class="card" style="background-color: #e3f2fd; text-align: center; padding: 2px;">', unsafe_allow_html=True)
    st.markdown("### Ready to classify your audio... ?")
    
if app_mode == "Classify Audio":
    st.header("Audio Classification")
    
    st.subheader("1️⃣ Audio Input")
    input_method = st.radio("Choose input method:", ["Upload Audio File", "Record Audio"])
    
    if input_method == "Upload Audio File":
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])
        
        if uploaded_file is not None:

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.audio_file = tmp_file.name
            
            st.audio(uploaded_file, format='audio/wav')
            st.success("✅ Audio file uploaded!")
    
    else:
        if st.button("Start Recording (5 seconds)"):
            st.session_state.audio_file = record_audio(duration=5, fs=SAMPLE_RATE)
            st.audio(st.session_state.audio_file, format='audio/wav')
    
    if st.session_state.audio_file is not None and st.button("Extract Features"):
        with st.spinner("Extracting audio features..."):
            params = get_parameters()
            st.session_state.features = extract_features(st.session_state.audio_file, params)
            
            if st.session_state.features is not None:
                st.success("✅ Features extracted successfully!")
                
                st.subheader("2️⃣ Audio Visualizations")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.pyplot(visualize_audio_waveform(st.session_state.features['audio'], st.session_state.features['sr']))
                
                with col2:
                    st.write("MFCC and Mel Spectrogram")
                    st.pyplot(visualize_mfcc(st.session_state.features['mfccs']))
                    st.pyplot(visualize_mel_spectrogram(st.session_state.features['mel_spec']))
                   
    
    if st.session_state.features is not None and st.session_state.current_model is not None:
        st.subheader("3️⃣ Classification Filters")
                
        metadata = st.session_state.current_metadata

        if isinstance(metadata['category_mapping'], dict):
            all_categories = [metadata['category_mapping'][i] for i in sorted(metadata['category_mapping'].keys())]
        else:
            all_categories = metadata['category_mapping']

        if isinstance(metadata['subcategory_mapping'], dict):
            all_subcategories = [metadata['subcategory_mapping'][i] for i in sorted(metadata['subcategory_mapping'].keys())]
        else:
            all_subcategories = metadata['subcategory_mapping']

        if 'subcategory_by_category' not in st.session_state:

            subcategory_by_category = {
                'Animals': ['cat', 'dog', 'elephant', 'horse', 'lion'],
                'Birds': ['crow', 'parrot', 'peacock', 'sparrow'],
                'Environment': ['crowd', 'office', 'rainfall', 'wind', 'traffic', 'military'],
                'Vehicles': ['airplane', 'bicycle', 'bike', 'bus', 'car', 'helicopter', 
                             'train', 'truck']
            }
            
            st.session_state.subcategory_by_category = subcategory_by_category
        else:
            subcategory_by_category = st.session_state.subcategory_by_category

        if 'category_selected' not in st.session_state:
            st.session_state.category_selected = {cat: True for cat in all_categories}

        def on_category_change(cat):
            
            is_selected = st.session_state[f"cat_{cat}"]
            st.session_state.category_selected[cat] = is_selected
            
            if not is_selected and cat in subcategory_by_category:
                for subcat in subcategory_by_category[cat]:
                    st.session_state[f"subcat_{subcat}"] = st.session_state.get(f"subcat_{subcat}", False)

        st.write("Select categories to include in classification:")
        selected_categories = []

        category_cols = st.columns(min(4, len(all_categories)))
        for i, category in enumerate(all_categories):
            col_idx = i % len(category_cols)
            with category_cols[col_idx]:
                if st.checkbox(category, value=st.session_state.category_selected.get(category, True), 
                            key=f"cat_{category}", 
                            on_change=on_category_change, 
                            args=(category,)):
                    selected_categories.append(category)

        st.write("Select subcategories to include in classification:")
        selected_subcategories = []

        subcat_cols = st.columns(3)
                
        for cat, subcats in subcategory_by_category.items():
            for i, subcat in enumerate(subcats):
                col_idx = i % len(subcat_cols)
                with subcat_cols[col_idx]:
                    default_value = st.session_state["category_selected"].get(cat, True)
                    
                    if f"subcat_{subcat}" not in st.session_state:
                        st.session_state[f"subcat_{subcat}"] = default_value
                    
                    if not st.session_state["category_selected"].get(cat, True):
                        st.session_state[f"subcat_{subcat}"] = False
                    
                    if st.checkbox(subcat, value=st.session_state[f"subcat_{subcat}"], key=f"subcat_{subcat}"):
                        if st.session_state["category_selected"].get(cat, True):  
                            selected_subcategories.append(subcat)
        
        st.session_state.selected_categories = selected_categories
        st.session_state.selected_subcategories = selected_subcategories
        
        if st.button("Classify Audio"):
            print("selected_categories", selected_categories)
            print("selected_subcategories", selected_subcategories)
            with st.spinner("Classifying..."):
                prediction_results = predict_with_filters(
                    st.session_state.features,
                    st.session_state.current_model,
                    metadata,
                    selected_categories,
                    selected_subcategories
                )
                st.session_state.prediction_results = prediction_results
                
                st.subheader("4️⃣ Classification Results")
                
                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    st.markdown(f"""
                                <div style="
                                    padding: 15px; 
                                    border-radius: 10px; 
                                    background-color: #f0f2f6; 
                                    text-align: center; 
                                    font-size: 20px; 
                                    font-weight: bold;
                                    color: black;">
                                    Predicted Category : <span style="text-transform: uppercase; color: #211C84;">{prediction_results['category']}</span>
                                </div>
                            """, unsafe_allow_html=True)

                    st.progress(prediction_results['category_confidence'])
                    cat_confidence = prediction_results['category_confidence'] * 100

                    st.metric(
                        label="Confidence",
                        value=f"",
                        delta=f"{cat_confidence:.2f}%"  
                    )

                
                with result_col2:
                    
                    st.markdown(f"""
                            <div style="
                                padding: 15px; 
                                border-radius: 10px; 
                                background-color: #f0f2f6; 
                                text-align: center; 
                                font-size: 20px; 
                                font-weight: bold;
                                color: black">
                                Predicted Sub Category : <span style="text-transform: uppercase; color: #0D4715;">{prediction_results['subcategory']}</span>
                            </div>
                        """, unsafe_allow_html=True)
                
                    st.progress(prediction_results['subcategory_confidence'])

                    sub_cat_confidence = prediction_results['category_confidence'] * 100

                    st.metric(
                        label="Confidence",
                        value=f"",
                        delta=f"{sub_cat_confidence:.2f}%"  
                    )
                
                st.subheader("Probability Distributions")
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    st.plotly_chart(
                        plot_probability_distribution(
                            prediction_results['category_probs'], 
                            "Category Probabilities"
                        ),
                        use_container_width=True
                    )
                
                with prob_col2:
                    st.plotly_chart(
                        plot_probability_distribution(
                            prediction_results['subcategory_probs'], 
                            "Subcategory Probabilities"
                        ),
                        use_container_width=True
                    )

elif app_mode == "Add Training Data":
    
    def reset_form_state():
        """Reset session state variables for form"""
        st.session_state.category_selection_complete = False
        st.session_state.subcategory_selection_complete = False
        st.session_state.selected_category = None
        st.session_state.selected_subcategory = None
        st.session_state.audio_path = None
    
    st.header("Add New Training Data")
    
    existing_categories, existing_subcategories = get_categories_and_subcategories()
    print(existing_categories, existing_subcategories)
    
    tab1, tab2, tab3 = st.tabs(["📂 Category Selection", "🔍 Subcategory Selection", "🎙️ Audio Input"])
    
    if 'category_selection_complete' not in st.session_state:
        st.session_state.category_selection_complete = False
    if 'subcategory_selection_complete' not in st.session_state:
        st.session_state.subcategory_selection_complete = False
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = None
    if 'selected_subcategory' not in st.session_state:
        st.session_state.selected_subcategory = None
    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = None
    
    # Tab 1: Category Selection
    with tab1:
        st.subheader("Step 1: Select or Create Category")
        
        # Use columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            category_option = st.radio(
                "Category options:", 
                ["Use existing category", "Create new category"],
                key="category_option"
            )
        
        with col2:
            if category_option == "Use existing category":
                if existing_categories:
                    selected_category = st.selectbox(
                        "Select category:", 
                        options=existing_categories,
                        key="existing_category_select"
                    )
                    new_category = None
                else:
                    st.warning("No existing categories found. Please create a new category.")
                    category_option = "Create new category"
                    new_category = st.text_input("Enter new category name:", key="new_category_forced")
                    selected_category = new_category
            else:
                new_category = st.text_input("Enter new category name:", key="new_category")
                selected_category = new_category
        
        if st.button("REGISTER", key="to_subcategory"):
            if selected_category:
                st.session_state.selected_category = selected_category
                st.session_state.category_selection_complete = True
                st.success(f"Category '{selected_category}' selected!")
                # st.rerun()
            else:
                st.error("Please specify a category name")
    
    # Tab 2: Subcategory Selection - Enable only after category selection
    with tab2:
        if not st.session_state.category_selection_complete:
            st.info("Please complete category selection first")
        else:
            st.subheader(f"Step 2: Select or Create Subcategory for '{st.session_state.selected_category}'")
            
            # Use columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                subcategory_option = st.radio(
                    "Subcategory options:", 
                    ["Use existing subcategory", "Create new subcategory"],
                    key="subcategory_option"
                )
            
            with col2:
                if subcategory_option == "Use existing subcategory":
                    category = st.session_state.selected_category
                    if category in existing_subcategories and existing_subcategories[category]:
                        selected_subcategory = st.selectbox(
                            "Select subcategory:", 
                            options=existing_subcategories[category],
                            key="existing_subcategory_select"
                        )
                        new_subcategory = None
                    else:
                        st.warning("No existing subcategories found. Please create a new subcategory.")
                        subcategory_option = "Create new subcategory"
                        new_subcategory = st.text_input("Enter new subcategory name:", key="new_subcategory_forced")
                        selected_subcategory = new_subcategory
                else:
                    new_subcategory = st.text_input("Enter new subcategory name:", key="new_subcategory")
                    selected_subcategory = new_subcategory
            
            if st.button("REGISTER", key="to_audio"):
                if selected_subcategory:
                    st.session_state.selected_subcategory = selected_subcategory
                    st.session_state.subcategory_selection_complete = True
                    st.success(f"Subcategory '{selected_subcategory}' selected!")
                    # st.rerun()
                else:
                    st.error("Please specify a subcategory name")
    
    # Tab 3: Audio Input - Enable only after subcategory selection
    with tab3:
        if not st.session_state.category_selection_complete or not st.session_state.subcategory_selection_complete:
            st.info("Please complete category and subcategory selection first")
        else:
            st.subheader(f"Step 3: Add Audio to {st.session_state.selected_category}/{st.session_state.selected_subcategory}")
            
            # Audio input selection with better layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                data_input_method = st.radio(
                    "Choose input method:", 
                    ["Upload Audio File", "Record Audio"], 
                    key="data_input_method"
                )
            
            with col2:
                if data_input_method == "Upload Audio File":
                    data_file = st.file_uploader(
                        "Upload an audio file", 
                        type=["wav", "mp3", "ogg", "flac"], 
                        key="audio_upload"
                    )
                    
                    if data_file:
                        
                        # Save to temporary file with original extension
                        file_extension = os.path.splitext(data_file.name)[1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                            tmp_file.write(data_file.getvalue())
                            temp_input_path = tmp_file.name
                        
                        # Convert to FLAC regardless of input format
                        temp_flac_path = os.path.splitext(temp_input_path)[0] + ".flac"
                        audio = AudioSegment.from_file(temp_input_path)
                        audio.export(temp_flac_path, format="flac")
                        
                        # Store FLAC path in session state
                        st.session_state.audio_path = temp_flac_path
                        
                        # Display the converted FLAC file
                        st.audio(temp_flac_path, format='audio/flac')
                                
                else:  # Record Audio
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        if st.button("Start Recording (5s)", key="record_button"):
                            with st.spinner("Recording in progress..."):
                                # Record audio (likely as WAV based on your record_audio function)
                                temp_wav_path = record_audio(duration=5, fs=SAMPLE_RATE)
                                
                                # Convert recorded WAV to FLAC
                                temp_flac_path = os.path.splitext(temp_wav_path)[0] + ".flac"
                                audio = AudioSegment.from_file(temp_wav_path, format="wav")
                                audio.export(temp_flac_path, format="flac")
                                
                                # Remove the WAV file as we only need FLAC
                                if os.path.exists(temp_wav_path):
                                    os.remove(temp_wav_path)
                                
                                # Store FLAC path in session state
                                st.session_state.audio_path = temp_flac_path

                    with col_b:
                        if "audio_path" in st.session_state and st.session_state.audio_path and data_input_method == "Record Audio":
                            st.audio(st.session_state.audio_path, format='audio/flac')  # Play converted FLAC

            if st.button("ADD TO DATASET", key="final_submit", type="primary"):
                if "audio_path" in st.session_state and st.session_state.audio_path:
                    # Verify that we're working with a FLAC file
                    if not st.session_state.audio_path.lower().endswith('.flac'):
                        st.error("Only FLAC format is supported. Please try again.")
                    else:
                        with st.spinner("Processing audio..."):
                            success, result = add_data_to_dataset(
                                st.session_state.audio_path, 
                                st.session_state.selected_category, 
                                st.session_state.selected_subcategory, 
                                DATASET_PATH
                            )
                            
                            if success:
                                st.success(f"✅ Audio added to dataset at: {result}")
                                # Reset state for next entry
                                st.button("Add Another Audio Sample", key="reset_form", on_click=reset_form_state)
                            else:
                                st.error(f"Failed to add audio: {result}")
                else:
                    st.error("Please provide audio data first")

elif app_mode == "Model Information":
    st.header("Model Information")
    
    if st.session_state.current_model is not None and st.session_state.current_metadata is not None:
        model_info, summary_str, model_df = get_model_info(st.session_state.current_model, st.session_state.current_metadata)
        
        st.markdown("## 📊 Model Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Architecture**: {model_info['Model Architecture']}")
            st.info(f"**Total Parameters**: {model_info['Total Parameters']}")
            st.info(f"**Input Shape**: {model_info['Input Shape']}")
            
        with col2:
            st.success(f"**Sample Rate**: {model_info['Sample Rate']} Hz")
            st.success(f"**Duration**: {model_info['Duration']}")
            st.success(f"**MFCC Features**: {model_info['MFCC Features']}")
            st.success(f"**Mel Bands**: {model_info['Mel Bands']}")
        
        
        categories = model_info['Categories'].split(', ')
        subcategories = model_info['Subcategories'].split(', ')
        
        
        tab1, tab2, tab3 = st.tabs(["📋 Layer Summary", "📊 Parameter Distribution", "🔍 Full Architecture"])
        
        with tab1:
            st.code(summary_str, language="text")
        
        with tab2:
            fig = go.Figure()
            
            param_counts = [int(param.replace(',', '')) for param in model_df['Param #']]
            
            sorted_indices = np.argsort(param_counts)[::-1]
            top_indices = sorted_indices[:10]
            
            fig.add_trace(go.Bar(
                x=[model_df.iloc[i]['Layer Name'] for i in top_indices],
                y=[param_counts[i] for i in top_indices],
                marker_color='rgba(50, 171, 96, 0.7)',
                text=[f"{param_counts[i]:,}" for i in top_indices],
                textposition='auto',
            ))
            
            fig.update_layout(
                title='Top 10 Layers by Parameter Count',
                xaxis_title='Layer Name',
                yaxis_title='Number of Parameters',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            layer_types = model_df['Layer Type'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=layer_types.index,
                values=layer_types.values,
                hole=.3,
                marker_colors=plt.cm.tab10.colors[:len(layer_types)]
            )])
            
            fig.update_layout(
                title='Layer Type Distribution',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.dataframe(
                model_df,
                column_config={
                    "Layer Name": st.column_config.TextColumn("Layer Name", width="medium"),
                    "Layer Type": st.column_config.TextColumn("Layer Type", width="small"),
                    "Output Shape": st.column_config.TextColumn("Output Shape", width="medium"),
                    "Param #": st.column_config.TextColumn("Parameters", width="small"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Generate a visual representation of the model architecture
            st.markdown("### 🖼️ Visual Architecture")
            
            # Create a simple visual representation of the architecture
            if st.button("Generate Visual Architecture"):
                with st.spinner("Generating architecture visualization..."):
                    # Create a simplified visual representation
                    layer_heights = {
                        'Conv2D': 80,
                        'MaxPooling2D': 60, 
                        'Dense': 40,
                        'Dropout': 30,
                        'Flatten': 20,
                        'BatchNormalization': 25,
                        'Input': 100,
                        'LSTM': 90,
                        'GRU': 90,
                        'Bidirectional': 90,
                    }
                    
                    # Default height for unknown layer types
                    default_height = 50
                    
                    # Create an image 
                    img_width = 800
                    img_height = 600
                    
                    img = Image.new('RGB', (img_width, img_height), color='white')
                    draw = ImageDraw.Draw(img)
                    
                    # Try to load a font, use default if not available
                    try:
                        font = ImageFont.truetype("arial.ttf", 14)
                        small_font = ImageFont.truetype("arial.ttf", 10)
                    except:
                        font = ImageFont.load_default()
                        small_font = ImageFont.load_default()
                    
                    # Draw layers
                    max_layers = min(len((st.session_state.current_model).layers), 15)  # Limit to 15 layers for clarity
                    layer_spacing = img_width // (max_layers + 1)
                    
                    for i in range(max_layers):
                        layer = (st.session_state.current_model).layers[i]
                        layer_type = layer.__class__.__name__
                        
                        # Get height for this layer type
                        height = layer_heights.get(layer_type, default_height)
                        
                        # Calculate position
                        x = (i + 1) * layer_spacing
                        y = img_height // 2
                        
                        # Choose color based on layer type
                        colors = {
                            'Conv2D': (100, 149, 237),  # Cornflower blue
                            'MaxPooling2D': (65, 105, 225),  # Royal blue
                            'Dense': (50, 205, 50),  # Lime green
                            'Dropout': (220, 220, 220),  # Light gray
                            'Flatten': (255, 165, 0),  # Orange
                            'BatchNormalization': (186, 85, 211),  # Medium orchid
                            'Input': (255, 99, 71),  # Tomato
                            'LSTM': (255, 215, 0),  # Gold
                            'GRU': (218, 165, 32),  # Goldenrod
                            'Bidirectional': (139, 69, 19),  # Saddle brown
                        }
                        
                        color = colors.get(layer_type, (200, 200, 200))
                        
                        # Draw the layer
                        draw.rectangle([x-40, y-height//2, x+40, y+height//2], fill=color, outline=(0, 0, 0))
                        
                        # Draw layer name and parameter count
                        draw.text((x-35, y-10), f"{layer_type}", fill=(0, 0, 0), font=font)
                        draw.text((x-35, y+10), f"Params: {layer.count_params():,}", fill=(0, 0, 0), font=small_font)
                        
                        # Draw connections
                        if i > 0:
                            prev_x = (i) * layer_spacing
                            prev_layer_type = (st.session_state.current_model).layers[i-1].__class__.__name__
                            prev_height = layer_heights.get(prev_layer_type, default_height)
                            
                            draw.line([(prev_x+40, img_height//2), (x-40, img_height//2)], fill=(0, 0, 0), width=2)
                    
                    # Convert PIL Image to bytes for Streamlit
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    buf.seek(0)
                    
                    # Display the image
                    st.image(buf, caption='Simplified Model Architecture', use_container_width =True)
                    
        # Provide download options
        st.markdown("## 📥 Export Options")
        
        col1, col2 = st.columns(2)
        with col1:
            # Option to download summary as text
            summary_text = f"""
            MODEL ARCHITECTURE SUMMARY
            =========================
            
            GENERAL INFORMATION:
            - Model: {model_info['Model Architecture']}
            - Total Parameters: {model_info['Total Parameters']}
            - Input Shape: {model_info['Input Shape']}
            
            AUDIO PARAMETERS:
            - Sample Rate: {model_info['Sample Rate']} Hz
            - Duration: {model_info['Duration']}
            - MFCC Features: {model_info['MFCC Features']}
            - Mel Bands: {model_info['Mel Bands']}
            
            CATEGORIES:
            {model_info['Categories']}
            
            SUBCATEGORIES:
            {model_info['Subcategories']}
            
            LAYER SUMMARY:
            {summary_str}
            """
            
            st.download_button(
                label="Download as Text",
                data=summary_text,
                file_name="model_architecture_summary.txt",
                mime="text/plain"
            )
            
        with col2:
            # Option to download as CSV
            csv = model_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="model_layers.csv",
                mime="text/csv"
            )
                
        st.subheader("Dataset Distribution")
                
        categories = ['Animals', 'Birds', 'Environment', 'user', 'Vehicles']
        subcategories = {'Animals': ['cat', 'dog', 'elephant', 'horse', 'lion'], 'Birds': ['crow', 'parrot', 'peacock', 'sparrow'], 'Environment': ['crowd', 'military', 'office', 'rainfall', 'traffic', 'wind'], 'user': ['lokesh_b'], 'Vehicles': ['airplane', 'bicycle', 'bike', 'bus', 'car', 'helicopter', 'train', 'truck']}
        
        category_counts = {'Animals': 3430, 'Birds': 3588, 'Environment': 6836, 'Vehicles': 9448}
        subcategory_counts = {'cat': 1032, 'dog': 596, 'elephant': 539, 'horse': 740, 'lion': 523, 'crow': 1095, 'parrot': 834, 'peacock': 497, 'sparrow': 1162, 'crowd': 918, 'military': 1107, 'office': 1376, 'rainfall': 1174, 'traffic': 1111, 'wind': 1150, 'airplane': 673, 'bicycle': 617, 'bike': 537, 'bus': 4221, 'car': 230, 'helicopter': 353, 'train': 2552, 'truck': 265}

        print("\n")
        print(category_counts)
        print("\n")
        print(subcategory_counts)
        print("\n")
        
        fig1 = px.bar(
            x=list(category_counts.keys()),
            y=list(category_counts.values()),
            labels={'x': 'Category', 'y': 'Number of Samples'},
            title='Samples per Category'
        )
        
        fig2 = px.bar(
            x=list(subcategory_counts.keys()),
            y=list(subcategory_counts.values()),
            labels={'x': 'Subcategory', 'y': 'Number of Samples'},
            title='Samples per Subcategory'
        )
        fig2.update_layout(xaxis_tickangle=-45)
        
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("New Files in Dataset")
        
        def new_file_function():
            
            data = []

            if not os.path.exists('DATASET_FLAC'):
                st.warning("Directory 'DATASET_FLAC' not found.")
            else:
                for dirname, _, files in os.walk('DATASET_FLAC'):
                    for filename in files:
                        full_path = os.path.join(dirname, filename)
                        path_parts = full_path.split(os.sep)
                        
                        if len(path_parts) >= 4:  
                            category = path_parts[1]
                            subcategory = path_parts[2]
                            data.append({
                                "Category": category,
                                "Subcategory": subcategory,
                                "Filename": filename
                            })

                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(
                        df,
                        use_container_width=True,  
                        hide_index=True  
                    )
                else:
                    st.warning("No New files found in the 'DATASET_FLAC' directory. Try adding some files.")
                    
        new_file_function()
        st.info("We have received your new data - audio files with category and subcategory ... ! The Models are updatable with the new files added to the dataset. Usually the update process takes 2 - 3 hours for each model. Hence, all the Models are updated with current data ONCE A MONTH. !")
        
    else:
        st.error("No model is currently loaded. Please select a model from the sidebar.")

st.markdown("---")  # Horizontal line

# Center-aligned footer text using HTML + CSS
st.markdown(
    """
    <div style="text-align: center; font-size: 16px;">
        <strong>Generic Audio Classifier Application</strong> | A modern and user flexible Streamlit app for envirnoment audio classification. | Lokesh Bhaskar
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import os
import cv2
import torch
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
import heapq
import threading

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Audio recorder for browser-based microphone recording
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# Text-to-Speech import
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Twilio import for call alerts
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# ============== TWILIO CONFIGURATION ==============
# Set your Twilio credentials here or use environment variables
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID', 'your_account_sid_here')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN', 'your_auth_token_here')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER', '+1234567890')  # Your Twilio phone number
# Use TWILIO_VERIFIED_PHONE_NUMBER as the default alert recipient
ALERT_PHONE_NUMBERS = os.environ.get('ALERT_PHONE_NUMBERS', os.environ.get('TWILIO_VERIFIED_PHONE_NUMBER', '+1234567890')).split(',')

# Debug: Print Twilio configuration status
print(f"Twilio Available: {TWILIO_AVAILABLE}")
print(f"Twilio Account SID: {TWILIO_ACCOUNT_SID[:10]}..." if TWILIO_ACCOUNT_SID != 'your_account_sid_here' else "Twilio Account SID: NOT CONFIGURED")
print(f"Twilio Phone Number: {TWILIO_PHONE_NUMBER}")
print(f"Alert Phone Numbers: {ALERT_PHONE_NUMBERS}")

def make_twilio_call(location_data, phone_numbers=None):
    """
    Make an automated phone call using Twilio when elephant is detected.
    Runs in a separate thread to avoid blocking the UI.
    """
    def _make_call():
        try:
            if not TWILIO_AVAILABLE:
                print("❌ Twilio not installed. Run: pip install twilio")
                return
            
            if TWILIO_ACCOUNT_SID == 'your_account_sid_here':
                print("❌ Twilio credentials not configured. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER.")
                return
            
            print(f"📞 Initiating Twilio call...")
            print(f"   From: {TWILIO_PHONE_NUMBER}")
            print(f"   To: {phone_numbers or ALERT_PHONE_NUMBERS}")
            
            client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
            # TwiML for the voice message
            twiml_message = f"""
            <Response>
                <Say voice="alice" language="en-US">
                    Alert! Alert! Elephant detected!
                    Location: {location_data['location_name']}.
                    Zone: {location_data['zone']}.
                    Coordinates: {location_data['coordinates'][0]}, {location_data['coordinates'][1]}.
                    Nearest response unit at {location_data['responder_name']}.
                    Distance: {location_data['distance_to_responder']} meters.
                    Immediate action required!
                    I repeat, elephant detected at {location_data['location_name']}.
                </Say>
                <Pause length="1"/>
                <Say voice="alice">Please respond immediately. This is an automated alert from the Elephant Detection System.</Say>
            </Response>
            """
            
            numbers_to_call = phone_numbers or ALERT_PHONE_NUMBERS
            
            for phone_number in numbers_to_call:
                phone_number = phone_number.strip()
                if phone_number:
                    try:
                        print(f"📞 Calling {phone_number}...")
                        call = client.calls.create(
                            twiml=twiml_message,
                            to=phone_number,
                            from_=TWILIO_PHONE_NUMBER
                        )
                        print(f"✅ Twilio call initiated to {phone_number}: SID={call.sid}")
                    except Exception as call_error:
                        print(f"❌ Failed to call {phone_number}: {call_error}")
                        
        except Exception as e:
            print(f"❌ Twilio Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run in separate thread to not block UI
    thread = threading.Thread(target=_make_call, daemon=True)
    thread.start()

def test_twilio_connection():
    """
    Test Twilio connection synchronously and return result for UI feedback.
    """
    result = {"success": False, "message": "", "details": ""}
    
    try:
        if not TWILIO_AVAILABLE:
            result["message"] = "Twilio not installed"
            result["details"] = "Run: pip install twilio"
            return result
        
        if TWILIO_ACCOUNT_SID == 'your_account_sid_here':
            result["message"] = "Twilio credentials not configured"
            result["details"] = "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN in .env file"
            return result
        
        # Validate SID format
        if not TWILIO_ACCOUNT_SID.startswith('AC'):
            result["message"] = "Invalid Account SID format"
            result["details"] = f"Account SID must start with 'AC', got '{TWILIO_ACCOUNT_SID[:2]}...'. Check your Twilio Console."
            return result
        
        # Try to create client and fetch account info
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        account = client.api.accounts(TWILIO_ACCOUNT_SID).fetch()
        
        result["success"] = True
        result["message"] = f"Connected to Twilio account: {account.friendly_name}"
        result["details"] = f"Account Status: {account.status}, Type: {account.type}"
        
    except Exception as e:
        result["message"] = f"Twilio connection failed"
        result["details"] = str(e)
    
    return result

def send_test_sms_sync(phone_number):
    """
    Send a test SMS synchronously and return result for UI feedback.
    """
    result = {"success": False, "message": "", "sid": ""}
    
    try:
        if not TWILIO_AVAILABLE:
            result["message"] = "Twilio not installed"
            return result
        
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        message = client.messages.create(
            body="🐘 TEST: Elephant Detection System - Twilio connection verified!",
            to=phone_number.strip(),
            from_=TWILIO_PHONE_NUMBER
        )
        
        result["success"] = True
        result["message"] = f"SMS sent successfully!"
        result["sid"] = message.sid
        
    except Exception as e:
        result["message"] = f"SMS failed: {str(e)}"
    
    return result

def send_test_call_sync(phone_number):
    """
    Send a test call synchronously and return result for UI feedback.
    """
    result = {"success": False, "message": "", "sid": ""}
    
    try:
        if not TWILIO_AVAILABLE:
            result["message"] = "Twilio not installed"
            return result
        
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        twiml_message = """
        <Response>
            <Say voice="alice" language="en-US">
                This is a test call from the Elephant Detection System.
                Your Twilio integration is working correctly.
                Thank you!
            </Say>
        </Response>
        """
        
        call = client.calls.create(
            twiml=twiml_message,
            to=phone_number.strip(),
            from_=TWILIO_PHONE_NUMBER
        )
        
        result["success"] = True
        result["message"] = f"Call initiated successfully!"
        result["sid"] = call.sid
        
    except Exception as e:
        result["message"] = f"Call failed: {str(e)}"
    
    return result

def send_twilio_sms(location_data, phone_numbers=None):
    """
    Send SMS alert using Twilio when elephant is detected.
    Runs in a separate thread to avoid blocking the UI.
    """
    def _send_sms():
        try:
            if not TWILIO_AVAILABLE:
                print("Twilio not installed. Run: pip install twilio")
                return
            
            if TWILIO_ACCOUNT_SID == 'your_account_sid_here':
                print("Twilio credentials not configured.")
                return
            
            client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
            sms_message = (
                f"🚨 ELEPHANT ALERT!\n"
                f"📍 Location: {location_data['location_name']}\n"
                f"🗺️ Zone: {location_data['zone']}\n"
                f"📐 Coords: ({location_data['coordinates'][0]}, {location_data['coordinates'][1]})\n"
                f"⏰ Time: {location_data['timestamp']}\n"
                f"🏃 Nearest: {location_data['responder_name']} ({location_data['distance_to_responder']}m)\n"
                f"🛤️ Path: {' → '.join(location_data['response_path'])}\n"
                f"⚠️ IMMEDIATE ACTION REQUIRED!"
            )
            
            numbers_to_text = phone_numbers or ALERT_PHONE_NUMBERS
            
            for phone_number in numbers_to_text:
                phone_number = phone_number.strip()
                if phone_number:
                    try:
                        message = client.messages.create(
                            body=sms_message,
                            to=phone_number,
                            from_=TWILIO_PHONE_NUMBER
                        )
                        print(f"Twilio SMS sent to {phone_number}: SID={message.sid}")
                    except Exception as sms_error:
                        print(f"Failed to SMS {phone_number}: {sms_error}")
                        
        except Exception as e:
            print(f"Twilio SMS Error: {e}")
    
    # Run in separate thread to not block UI
    thread = threading.Thread(target=_send_sms, daemon=True)
    thread.start()
# ============== END TWILIO CONFIGURATION ==============

# ============== SENSOR NETWORK & LOCATION DETECTION SYSTEM ==============

# Define sensor nodes with coordinates (simulating forest perimeter sensors)
SENSOR_NODES = {
    "Sensor_A": {"name": "North Gate", "coords": (0, 100), "zone": "Zone-1"},
    "Sensor_B": {"name": "East Watchtower", "coords": (100, 50), "zone": "Zone-2"},
    "Sensor_C": {"name": "South Checkpoint", "coords": (50, 0), "zone": "Zone-3"},
    "Sensor_D": {"name": "West Boundary", "coords": (0, 50), "zone": "Zone-4"},
    "Sensor_E": {"name": "Central Hub", "coords": (50, 50), "zone": "Zone-5"},
    "Sensor_F": {"name": "Northeast Corner", "coords": (80, 80), "zone": "Zone-2"},
    "Sensor_G": {"name": "Southwest Edge", "coords": (20, 20), "zone": "Zone-3"},
    "Sensor_H": {"name": "Control Station", "coords": (50, 100), "zone": "Zone-1"},
}

# Define graph edges (connections between sensors with distances)
SENSOR_GRAPH = {
    "Sensor_A": {"Sensor_D": 50, "Sensor_E": 70, "Sensor_H": 50},
    "Sensor_B": {"Sensor_E": 50, "Sensor_F": 36, "Sensor_C": 70},
    "Sensor_C": {"Sensor_G": 36, "Sensor_E": 50, "Sensor_B": 70},
    "Sensor_D": {"Sensor_A": 50, "Sensor_G": 36, "Sensor_E": 50},
    "Sensor_E": {"Sensor_A": 70, "Sensor_B": 50, "Sensor_C": 50, "Sensor_D": 50, 
                 "Sensor_F": 42, "Sensor_G": 42, "Sensor_H": 50},
    "Sensor_F": {"Sensor_B": 36, "Sensor_E": 42, "Sensor_H": 36},
    "Sensor_G": {"Sensor_C": 36, "Sensor_D": 36, "Sensor_E": 42},
    "Sensor_H": {"Sensor_A": 50, "Sensor_F": 36, "Sensor_E": 50},
}

def dijkstra_shortest_path(graph, start, end):
    """
    Dijkstra's algorithm to find shortest path between two sensors.
    Returns (distance, path) tuple.
    """
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        if current_node == end:
            break
            
        for neighbor, weight in graph.get(current_node, {}).items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    return distances[end], path

def find_nearest_responder_location(detection_sensor, responder_locations=None):
    """
    Find the nearest responder/control station from detection point using Dijkstra's.
    """
    if responder_locations is None:
        responder_locations = ["Sensor_H", "Sensor_E"]  # Control stations
    
    nearest = None
    min_distance = float('infinity')
    best_path = []
    
    for responder in responder_locations:
        distance, path = dijkstra_shortest_path(SENSOR_GRAPH, detection_sensor, responder)
        if distance < min_distance:
            min_distance = distance
            nearest = responder
            best_path = path
    
    return nearest, min_distance, best_path

def get_detection_location_info(sensor_id=None):
    """
    Get detailed location information for a detected elephant.
    If no sensor_id provided, randomly select one to simulate detection.
    """
    import random
    
    if sensor_id is None:
        sensor_id = random.choice(list(SENSOR_NODES.keys()))
    
    sensor_info = SENSOR_NODES.get(sensor_id, SENSOR_NODES["Sensor_E"])
    nearest_responder, distance, path = find_nearest_responder_location(sensor_id)
    responder_info = SENSOR_NODES.get(nearest_responder)
    
    location_data = {
        "detection_sensor": sensor_id,
        "location_name": sensor_info["name"],
        "coordinates": sensor_info["coords"],
        "zone": sensor_info["zone"],
        "nearest_responder": nearest_responder,
        "responder_name": responder_info["name"],
        "distance_to_responder": distance,
        "response_path": path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "alert_level": "HIGH" if "elephant" in sensor_info["name"].lower() else "CRITICAL"
    }
    
    return location_data

def speak_alert(message):
    """
    Use text-to-speech to announce elephant detection alert.
    Runs in a separate thread to avoid blocking the UI.
    """
    def _speak():
        try:
            if TTS_AVAILABLE:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)  # Speed of speech
                engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
                engine.say(message)
                engine.runAndWait()
                engine.stop()
            else:
                # Fallback: use system beep on Windows
                try:
                    import winsound
                    winsound.Beep(1000, 500)
                    winsound.Beep(1500, 500)
                    winsound.Beep(1000, 500)
                except:
                    pass
        except Exception as e:
            print(f"TTS Error: {e}")
    
    # Run in separate thread to not block UI
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()

def generate_alert_message(location_data, detection_type="visual"):
    """
    Generate alert message for speaker and notifications.
    """
    alert_msg = (
        f"Warning! Elephant detected at {location_data['location_name']}, "
        f"{location_data['zone']}. "
        f"Coordinates: {location_data['coordinates'][0]}, {location_data['coordinates'][1]}. "
        f"Nearest response unit at {location_data['responder_name']}, "
        f"Distance: {location_data['distance_to_responder']} meters. "
        f"Immediate action required!"
    )
    return alert_msg

# ============== END LOCATION DETECTION SYSTEM ==============

subcategory_with_emojis = {
    'cat': '🐱',
    'dog': '🐶',
    'elephant': '🐘',
    'horse': '🐴',
    'lion': '🦁',
    'crow': '🐦‍⬛',
    'parrot': '🦜',
    'peacock': '🦚',
    'sparrow': '🐦',
    'crowd': '👥',
    'office': '🏢',
    'rainfall': '🌧️',
    'wind': '🌬️',
    'traffic': '🚦',
    'military': '🪖',
    'airplane': '✈️',
    'bicycle': '🚲',
    'bike': '🏍️',
    'bus': '🚌',
    'car': '🚗',
    'helicopter': '🚁',
    'train': '🚆',
    'truck': '🚚'
}


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
DURATION = 5  
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
        'duration': 5,  
        'n_mfcc': 40,
        'n_mels': 128,
        'n_fft': 2048,
        'hop_length': 512,
        'batch_size': 32,
        'epochs': 20,
        'validation_split': 0.2,
        'random_state': 42,
        'num_test_samples': 10  
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
        category_dir = os.path.join(save_path, category)
        subcategory_dir = os.path.join(category_dir, subcategory)
        
        os.makedirs(subcategory_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sample_{timestamp}.wav"  
        destination = os.path.join(subcategory_dir, filename)

        try:
            data, samplerate = sf.read(audio_file) 
            sf.write(destination, data, samplerate)  
        except Exception as e:
            return False, f"Invalid WAV file: {e}"

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

# --- Elephant Detection Logic ---
@st.cache_resource
def load_yolo_model_cached(weights_path=None):
    try:
        # Check standard paths or absolute
        if weights_path and os.path.exists(weights_path):
            return torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    except Exception as e:
        st.error(f"YOLOv5 load error: {e}")
        return None

def process_detection_frame(frame, model):
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        df = results.pandas().xyxy[0]
        detected_animals = []
        
        for _, row in df.iterrows():
            name = row['name']
            conf = row['confidence']
            if name.lower() in ['elephant', 'pig']: 
                detected_animals.append(name)
                xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                label = f"{name.upper()} {conf:.2f}"
                cv2.putText(img_rgb, label, (xmin, max(ymin - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return img_rgb, detected_animals
    except Exception as e:
        return frame, []
# --------------------------------

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

    summary_str = []
    model.summary(print_fn=lambda x: summary_str.append(x))
    
    model_layers = []
    for layer in model.layers:
        try:
            output_shape = str(layer.output.shape)  
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

st.title("🔊 GAC -  Generic Audio Classifier")

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
        ["Home", "Classify Audio", "Elephant Detection", "Add Training Data", "Model Information"],  
        icons=["house", "soundwave", "camera-video", "folder-plus", "info-circle"],  
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
    
    st.markdown('<div class="card" style="background-color: #e3f2fd; text-align: center; padding: 2px;">', unsafe_allow_html=True)
    st.markdown("### Ready to classify your audio... ?")
    
if app_mode == "Classify Audio":
    st.header("Audio Classification")
    
    st.subheader("1️⃣ Audio Input")
    input_method = st.radio("Choose input method:", ["Upload Audio File", "🎙️ Record from Microphone"])

    if input_method == "Upload Audio File":
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
        if uploaded_file is not None:

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.audio_file = tmp_file.name
            
            st.audio(uploaded_file, format='audio/wav')
            st.success("✅ Audio file uploaded!")
    
    elif input_method == "🎙️ Record from Microphone":
        st.info("🎤 Click the microphone button below to start recording. Click again to stop.")
        
        # Recording settings
        col_settings1, col_settings2 = st.columns(2)
        with col_settings1:
            recording_duration = st.slider("Max Recording Duration (seconds)", 1, 30, 5)
        with col_settings2:
            sample_rate = st.selectbox("Sample Rate", [22050, 44100, 16000], index=0)
        
        if AUDIO_RECORDER_AVAILABLE:
            # Use audio_recorder_streamlit for browser-based recording
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="3x",
                sample_rate=sample_rate,
                pause_threshold=recording_duration
            )
            
            if audio_bytes:
                # Save the recorded audio to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes)
                    st.session_state.audio_file = tmp_file.name
                
                st.audio(audio_bytes, format='audio/wav')
                st.success("✅ Audio recorded successfully!")
                
                # Show recording info
                try:
                    audio_data, sr = librosa.load(st.session_state.audio_file, sr=None)
                    duration = len(audio_data) / sr
                    st.info(f"📊 Recording Info: Duration: {duration:.2f}s | Sample Rate: {sr} Hz")
                except Exception as e:
                    st.warning(f"Could not read audio info: {e}")
        else:
            # Fallback: Use sounddevice for direct recording
            st.warning("Browser-based recorder not available. Using direct microphone recording.")
            
            if 'recording_state' not in st.session_state:
                st.session_state.recording_state = False
            if 'recorded_audio' not in st.session_state:
                st.session_state.recorded_audio = None
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                if st.button("🔴 Start Recording", disabled=st.session_state.recording_state):
                    st.session_state.recording_state = True
                    st.info(f"🎙️ Recording for {recording_duration} seconds...")
                    
                    # Progress bar for recording
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Record audio using sounddevice
                        recorded_audio = sd.rec(
                            int(recording_duration * sample_rate),
                            samplerate=sample_rate,
                            channels=1,
                            dtype='float32'
                        )
                        
                        # Show progress
                        for i in range(recording_duration * 10):
                            time.sleep(0.1)
                            progress_bar.progress((i + 1) / (recording_duration * 10))
                            status_text.text(f"Recording... {(i + 1) / 10:.1f}s / {recording_duration}s")
                        
                        sd.wait()  # Wait for recording to complete
                        
                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            sf.write(tmp_file.name, recorded_audio, sample_rate)
                            st.session_state.audio_file = tmp_file.name
                            st.session_state.recorded_audio = recorded_audio
                        
                        progress_bar.progress(1.0)
                        status_text.text("")
                        st.success("✅ Recording completed!")
                        st.audio(st.session_state.audio_file, format='audio/wav')
                        st.info(f"📊 Recording Info: Duration: {recording_duration}s | Sample Rate: {sample_rate} Hz")
                        
                    except Exception as e:
                        st.error(f"Recording failed: {e}")
                        st.info("💡 Make sure your microphone is connected and browser has permission to access it.")
                    finally:
                        st.session_state.recording_state = False
            
            with col_rec2:
                if st.button("🗑️ Clear Recording"):
                    st.session_state.audio_file = None
                    st.session_state.recorded_audio = None
                    st.session_state.features = None
                    st.rerun()       
            
    def get_plot_bytes(fig):
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format="png")
        img_bytes.seek(0)
        return img_bytes
    
    if st.session_state.audio_file is not None and st.button("Extract Features"):
        with st.spinner("Extracting audio features..."):
            params = get_parameters()
            st.session_state.features = extract_features(st.session_state.audio_file, params)
            
            if st.session_state.features is not None:
                st.success("✅ Features extracted successfully!")

                waveform_fig = visualize_audio_waveform(st.session_state.features['audio'], st.session_state.features['sr'])
                mfcc_fig = visualize_mfcc(st.session_state.features['mfccs'])
                mel_spec_fig = visualize_mel_spectrogram(st.session_state.features['mel_spec'])
                
                st.subheader("2️⃣ Audio Visualizations")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("WaveForm Visualization")
                    st.pyplot(waveform_fig)
                    waveform_bytes = get_plot_bytes(waveform_fig)
                    st.download_button("Download Waveform", waveform_bytes, file_name="waveform.png", mime="image/png")
                    
                with col2:
                    st.write("MFCC and Mel Spectrogram")
                    
                    st.pyplot(mfcc_fig)
                    mfcc_bytes = get_plot_bytes(mfcc_fig)
                    st.download_button("Download MFCC", mfcc_bytes, file_name="mfcc.png", mime="image/png")
                
                    st.pyplot(mel_spec_fig)
                    mel_spec_bytes = get_plot_bytes(mel_spec_fig)
                    st.download_button("Download Mel Spectrogram", mel_spec_bytes, file_name="mel_spectrogram.png", mime="image/png")
                                   
    
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
                                Predicted Sub Category : <span style="text-transform: uppercase; color: #0D4715;">
                                {prediction_results['subcategory']} {subcategory_with_emojis[prediction_results['subcategory']]}
                            </span>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Check if elephant was detected in audio classification
                if prediction_results['subcategory'].lower() == 'elephant' and prediction_results['subcategory_confidence'] > 0.5:
                    st.markdown("---")
                    st.error("🚨 **ELEPHANT SOUND DETECTED!** 🐘")
                    
                    # Get location information
                    import random
                    audio_sensor = random.choice(list(SENSOR_NODES.keys()))
                    location_data = get_detection_location_info(audio_sensor)
                    
                    # Trigger speaker alert
                    alert_msg = f"Alert! Elephant sound detected at {location_data['location_name']}, {location_data['zone']}. " \
                               f"Nearest response unit at {location_data['responder_name']}. Immediate action required!"
                    speak_alert(alert_msg)
                    
                    # Show location popup
                    audio_alert_html = f"""
                    <div style="
                        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                        color: white;
                        padding: 20px;
                        border-radius: 10px;
                        margin: 10px 0;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                    ">
                        <h3 style="margin: 0 0 15px 0;">🔊 Audio-Based Elephant Detection Alert</h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <div>
                                <p><strong>📍 Detection Location:</strong> {location_data['location_name']}</p>
                                <p><strong>🗺️ Zone:</strong> {location_data['zone']}</p>
                                <p><strong>📐 Coordinates:</strong> ({location_data['coordinates'][0]}, {location_data['coordinates'][1]})</p>
                            </div>
                            <div>
                                <p><strong>🏃 Nearest Responder:</strong> {location_data['responder_name']}</p>
                                <p><strong>📏 Distance:</strong> {location_data['distance_to_responder']}m</p>
                                <p><strong>🛤️ Response Path:</strong> {' → '.join(location_data['response_path'])}</p>
                            </div>
                        </div>
                        <p style="margin-top: 10px;"><strong>⏰ Time:</strong> {location_data['timestamp']}</p>
                    </div>
                    """
                    st.markdown(audio_alert_html, unsafe_allow_html=True)
                    
                    # Show detailed location info
                    st.subheader("📍 Location Analysis (Shortest Path Algorithm)")
                    loc_col1, loc_col2, loc_col3 = st.columns(3)
                    with loc_col1:
                        st.metric("Detection Sensor", location_data['detection_sensor'])
                        st.metric("Location Name", location_data['location_name'])
                    with loc_col2:
                        st.metric("Zone", location_data['zone'])
                        st.metric("Coordinates", f"({location_data['coordinates'][0]}, {location_data['coordinates'][1]})")
                    with loc_col3:
                        st.metric("Nearest Responder", location_data['responder_name'])
                        st.metric("Response Distance", f"{location_data['distance_to_responder']}m")

elif app_mode == "Elephant Detection":
    st.header("🐘 Elephant Detection")
    st.markdown("Use **YOLOv5** to detect elephants in uploaded videos or live webcam feed.")
    
    # Initialize alert state in session
    if 'elephant_alert_shown' not in st.session_state:
        st.session_state.elephant_alert_shown = False
    if 'last_detection_location' not in st.session_state:
        st.session_state.last_detection_location = None
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []

    det_mode = st.radio("Input Source", ["Upload Video", "Webcam"], horizontal=True)
    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
    
    # Sensor selection for location simulation
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗺️ Location Settings")
    selected_sensor = st.sidebar.selectbox(
        "Active Detection Sensor",
        list(SENSOR_NODES.keys()),
        format_func=lambda x: f"{x} - {SENSOR_NODES[x]['name']}"
    )
    
    enable_speaker = st.sidebar.checkbox("🔊 Enable Speaker Alerts", value=True)
    enable_popup = st.sidebar.checkbox("📢 Enable Popup Notifications", value=True)
    
    # Twilio Alert Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("📞 Twilio Call Alerts")
    enable_twilio_call = st.sidebar.checkbox("📞 Enable Phone Call Alerts", value=False)
    enable_twilio_sms = st.sidebar.checkbox("💬 Enable SMS Alerts", value=False)
    
    # Twilio configuration in sidebar
    with st.sidebar.expander("⚙️ Twilio Configuration", expanded=True):
        if TWILIO_AVAILABLE:
            st.success("✅ Twilio package installed")
        else:
            st.error("❌ Twilio not installed. Run: pip install twilio")
        
        st.text(f"Account SID: {TWILIO_ACCOUNT_SID[:10]}...")
        st.text(f"From Number: {TWILIO_PHONE_NUMBER}")
        st.text(f"To Number: {ALERT_PHONE_NUMBERS}")
        
        custom_phone = st.text_input(
            "Alert Phone Number(s)", 
            placeholder="+1234567890, +0987654321",
            help="Enter phone numbers to receive alerts (comma-separated)"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔗 Test Connection"):
                with st.spinner("Testing Twilio connection..."):
                    result = test_twilio_connection()
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.info(result['details'])
                    else:
                        st.error(f"❌ {result['message']}")
                        st.warning(result['details'])
        
        with col2:
            if st.button("📞 Test Call"):
                target_phone = custom_phone.strip() if custom_phone else ALERT_PHONE_NUMBERS[0]
                with st.spinner(f"Calling {target_phone}..."):
                    result = send_test_call_sync(target_phone)
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.info(f"Call SID: {result['sid']}")
                    else:
                        st.error(f"❌ {result['message']}")
        
        with col3:
            if st.button("💬 Test SMS"):
                target_phone = custom_phone.strip() if custom_phone else ALERT_PHONE_NUMBERS[0]
                with st.spinner(f"Sending SMS to {target_phone}..."):
                    result = send_test_sms_sync(target_phone)
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.info(f"Message SID: {result['sid']}")
                    else:
                        st.error(f"❌ {result['message']}")
    
    # Display sensor network map
    with st.sidebar.expander("📍 View Sensor Network"):
        sensor_df = pd.DataFrame([
            {"Sensor": k, "Name": v["name"], "Zone": v["zone"], 
             "X": v["coords"][0], "Y": v["coords"][1]}
            for k, v in SENSOR_NODES.items()
        ])
        st.dataframe(sensor_df, hide_index=True)

    # Load YOLO model
    yolo_model = None
    try:
        with st.spinner("Loading YOLOv5 model... (first time may take a minute to download)"):
            yolo_model = load_yolo_model_cached()
    except Exception as e:
        st.error(f"Failed to load YOLOv5 model: {e}")

    if yolo_model is None:
        st.error("Could not load YOLOv5 model. Please check your internet connection (model downloads from GitHub on first run) and ensure `torch` is installed.")
    else:
        yolo_model.conf = conf_thresh
        st.success("YOLOv5 model loaded successfully!")
    
    # Function to display popup notification
    def show_elephant_alert_popup(location_data):
        """Display a prominent popup notification for elephant detection."""
        alert_html = f"""
        <div id="elephant-alert" style="
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 9999;
            background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            text-align: center;
            min-width: 400px;
            animation: pulse 1s infinite;
        ">
            <style>
                @keyframes pulse {{
                    0% {{ transform: translate(-50%, -50%) scale(1); }}
                    50% {{ transform: translate(-50%, -50%) scale(1.02); }}
                    100% {{ transform: translate(-50%, -50%) scale(1); }}
                }}
                @keyframes blink {{
                    0%, 100% {{ opacity: 1; }}
                    50% {{ opacity: 0.5; }}
                }}
            </style>
            <h1 style="margin: 0; font-size: 28px; animation: blink 0.5s infinite;">
                🚨 ELEPHANT DETECTED! 🚨
            </h1>
            <hr style="border-color: rgba(255,255,255,0.3); margin: 15px 0;">
            <div style="text-align: left; font-size: 16px;">
                <p><strong>📍 Location:</strong> {location_data['location_name']}</p>
                <p><strong>🗺️ Zone:</strong> {location_data['zone']}</p>
                <p><strong>📐 Coordinates:</strong> ({location_data['coordinates'][0]}, {location_data['coordinates'][1]})</p>
                <p><strong>⏰ Time:</strong> {location_data['timestamp']}</p>
                <hr style="border-color: rgba(255,255,255,0.3); margin: 10px 0;">
                <p><strong>🏃 Nearest Response:</strong> {location_data['responder_name']}</p>
                <p><strong>📏 Distance:</strong> {location_data['distance_to_responder']}m</p>
                <p><strong>🛤️ Response Path:</strong> {' → '.join(location_data['response_path'])}</p>
            </div>
        </div>
        """
        return alert_html
    
    # Function to handle elephant detection
    def handle_elephant_detection(detection_type="visual"):
        """Handle elephant detection: get location, trigger speaker, show popup, and Twilio alerts."""
        location_data = get_detection_location_info(selected_sensor)
        st.session_state.last_detection_location = location_data
        
        # Add to detection history
        st.session_state.detection_history.append({
            "time": location_data['timestamp'],
            "location": location_data['location_name'],
            "zone": location_data['zone'],
            "type": detection_type
        })
        
        # Trigger speaker alert
        if enable_speaker:
            alert_message = generate_alert_message(location_data, detection_type)
            speak_alert(alert_message)
        
        # Trigger Twilio phone call alert
        if enable_twilio_call:
            custom_numbers = [custom_phone] if custom_phone else None
            make_twilio_call(location_data, custom_numbers)
            print(f"📞 Twilio call alert triggered for detection at {location_data['location_name']}")
        
        # Trigger Twilio SMS alert
        if enable_twilio_sms:
            custom_numbers = [custom_phone] if custom_phone else None
            send_twilio_sms(location_data, custom_numbers)
            print(f"💬 Twilio SMS alert triggered for detection at {location_data['location_name']}")
        
        return location_data

    if det_mode == "Upload Video":
        st.subheader("📁 Upload a Video File")
        vid_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        if vid_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(vid_file.read())
            tfile.close()
            vpath = tfile.name

            st.video(vpath)

            if yolo_model is not None and st.button("🔍 Start Elephant Detection"):
                cap = cv2.VideoCapture(vpath)
                st_frame = st.empty()
                st_status = st.empty()
                st_alert_container = st.empty()
                st_location_info = st.empty()
                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0
                detection_triggered = False
                last_alert_time = 0
                alert_cooldown = 5  # seconds between alerts

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    p_frame, dets = process_detection_frame(frame, yolo_model)
                    st_frame.image(p_frame, channels="RGB", use_container_width=True)
                    
                    if dets:
                        current_time = time.time()
                        st_status.error(f"⚠️ DETECTED: {', '.join(set(dets))}")
                        
                        # Trigger alert with cooldown
                        if current_time - last_alert_time > alert_cooldown:
                            location_data = handle_elephant_detection("visual")
                            last_alert_time = current_time
                            
                            # Show popup notification
                            if enable_popup:
                                st_alert_container.markdown(show_elephant_alert_popup(location_data), unsafe_allow_html=True)
                            
                            # Show location details in dashboard
                            with st_location_info.container():
                                st.markdown("### 📍 Detection Location Details")
                                loc_col1, loc_col2, loc_col3 = st.columns(3)
                                with loc_col1:
                                    st.metric("Location", location_data['location_name'])
                                    st.metric("Zone", location_data['zone'])
                                with loc_col2:
                                    st.metric("Coordinates", f"({location_data['coordinates'][0]}, {location_data['coordinates'][1]})")
                                    st.metric("Distance to Response", f"{location_data['distance_to_responder']}m")
                                with loc_col3:
                                    st.metric("Nearest Responder", location_data['responder_name'])
                                    st.info(f"🛤️ Path: {' → '.join(location_data['response_path'])}")
                    else:
                        st_status.success("✅ Safe: No elephants detected")
                        st_alert_container.empty()
                        
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))
                cap.release()
                progress_bar.progress(1.0)
                st.success("✅ Finished processing video.")

    elif det_mode == "Webcam":
        st.subheader("📷 Live Webcam Detection")
        st.info("Check the box below to start your webcam. Uncheck to stop.")
        run_cam = st.checkbox("Start Camera")
        if run_cam:
            if yolo_model is None:
                st.error("Cannot start camera — YOLOv5 model not loaded.")
            else:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not open webcam. Make sure a camera is connected.")
                else:
                    st_frame = st.empty()
                    st_status = st.empty()
                    st_alert_container = st.empty()
                    st_location_info = st.empty()
                    last_alert_time = 0
                    alert_cooldown = 5  # seconds between alerts
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Lost webcam feed.")
                            break
                        p_frame, dets = process_detection_frame(frame, yolo_model)
                        st_frame.image(p_frame, channels="RGB", use_container_width=True)
                        
                        if dets:
                            current_time = time.time()
                            st_status.error(f"⚠️ DETECTED: {', '.join(set(dets))}")
                            
                            # Trigger alert with cooldown
                            if current_time - last_alert_time > alert_cooldown:
                                location_data = handle_elephant_detection("visual")
                                last_alert_time = current_time
                                
                                # Show popup notification
                                if enable_popup:
                                    st_alert_container.markdown(show_elephant_alert_popup(location_data), unsafe_allow_html=True)
                                
                                # Show location details in dashboard
                                with st_location_info.container():
                                    st.markdown("### 📍 Detection Location Details")
                                    loc_col1, loc_col2, loc_col3 = st.columns(3)
                                    with loc_col1:
                                        st.metric("Location", location_data['location_name'])
                                        st.metric("Zone", location_data['zone'])
                                    with loc_col2:
                                        st.metric("Coordinates", f"({location_data['coordinates'][0]}, {location_data['coordinates'][1]})")
                                        st.metric("Distance to Response", f"{location_data['distance_to_responder']}m")
                                    with loc_col3:
                                        st.metric("Nearest Responder", location_data['responder_name'])
                                        st.info(f"🛤️ Path: {' → '.join(location_data['response_path'])}")
                        else:
                            st_status.success("✅ Safe: No elephants detected")
                            st_alert_container.empty()
                    cap.release()
    
    # Display Detection History
    if st.session_state.detection_history:
        st.markdown("---")
        st.subheader("📋 Detection History")
        history_df = pd.DataFrame(st.session_state.detection_history)
        st.dataframe(history_df, hide_index=True, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.rerun()
    
    # Display Sensor Network Visualization
    st.markdown("---")
    st.subheader("🗺️ Sensor Network Map")
    
    # Create sensor network visualization using Plotly
    fig_network = go.Figure()
    
    # Add edges (connections between sensors)
    for sensor, connections in SENSOR_GRAPH.items():
        x0, y0 = SENSOR_NODES[sensor]["coords"]
        for connected_sensor, distance in connections.items():
            x1, y1 = SENSOR_NODES[connected_sensor]["coords"]
            fig_network.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color='lightgray', width=1),
                hoverinfo='none',
                showlegend=False
            ))
    
    # Add sensor nodes
    sensor_x = [SENSOR_NODES[s]["coords"][0] for s in SENSOR_NODES]
    sensor_y = [SENSOR_NODES[s]["coords"][1] for s in SENSOR_NODES]
    sensor_names = [f"{s}<br>{SENSOR_NODES[s]['name']}" for s in SENSOR_NODES]
    
    # Highlight active sensor
    colors = ['red' if s == selected_sensor else 'blue' for s in SENSOR_NODES]
    sizes = [20 if s == selected_sensor else 12 for s in SENSOR_NODES]
    
    fig_network.add_trace(go.Scatter(
        x=sensor_x, y=sensor_y,
        mode='markers+text',
        marker=dict(size=sizes, color=colors, symbol='circle'),
        text=[SENSOR_NODES[s]["name"] for s in SENSOR_NODES],
        textposition='top center',
        hovertext=sensor_names,
        hoverinfo='text',
        name='Sensors'
    ))
    
    fig_network.update_layout(
        title="Sensor Network Topology",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        showlegend=False,
        height=400,
        xaxis=dict(range=[-10, 110]),
        yaxis=dict(range=[-10, 110])
    )
    
    st.plotly_chart(fig_network, use_container_width=True)
    st.caption("🔴 Active detection sensor | 🔵 Other sensors")

elif app_mode == "Add Training Data":
    
    def reset_form_state():
        """Reset session state variables for form"""
        st.session_state.category_selection_complete = False
        st.session_state.subcategory_selection_complete = False
        st.session_state.selected_category = None
        st.session_state.selected_subcategory = None
        st.session_state.audio_path = None
    
    st.header("Add New Training Data")
    st.warning("The Streamlit version does not support audio recording. Please use the local device for audio recording. The recording feature is available in the file: [app_local_record.py](https://github.com/your-repo/audio_record.py)")

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
    
    with tab1:
        st.subheader("Step 1: Select or Create Category")
        
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
    
    with tab2:
        if not st.session_state.category_selection_complete:
            st.info("Please complete category selection first")
        else:
            st.subheader(f"Step 2: Select or Create Subcategory for '{st.session_state.selected_category}'")
            
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
    
    with tab3:
        if not st.session_state.category_selection_complete or not st.session_state.subcategory_selection_complete:
            st.info("Please complete category and subcategory selection first")
        else:
            st.subheader(f"Step 3: Add Audio to {st.session_state.selected_category}/{st.session_state.selected_subcategory}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                data_input_method = st.radio(
                    "Choose input method:", 
                    ["Upload Audio File"], 
                    key="data_input_method"
                )
            
            with col2:
                if data_input_method == "Upload Audio File":
                    data_file = st.file_uploader(
                        "Upload an audio file", 
                        type=["wav", "mp3"],
                        key="audio_upload"
                    )
                    
                    if data_file:
                        file_extension = os.path.splitext(data_file.name)[1].lower()
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                            tmp_file.write(data_file.getvalue())
                            temp_input_path = tmp_file.name

                        if file_extension == ".wav":
                            temp_wav_path = temp_input_path
                        else:
                            temp_wav_path = os.path.splitext(temp_input_path)[0] + ".wav"
                            audio = AudioSegment.from_mp3(temp_input_path) 
                            audio.export(temp_wav_path, format="wav")

                        st.session_state.audio_path = temp_wav_path

                        st.audio(temp_wav_path, format='audio/wav')                  

            if st.button("ADD TO DATASET", key="final_submit", type="primary"):
                if "audio_path" in st.session_state and st.session_state.audio_path:
                    
                    if not st.session_state.audio_path.lower().endswith('.wav'):
                        st.error("Only WAV format is supported. Please upload a WAV file.")
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
                                st.button("Add Another Audio Sample", key="reset_form", on_click=reset_form_state)
                            else:
                                st.error(f"Failed to add audio: {result}")
                else:
                    st.error("Please provide a WAV audio file first.")

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
            
            st.markdown("### 🖼️ Visual Architecture")
            
            if st.button("Generate Visual Architecture"):
                with st.spinner("Generating enhanced architecture visualization..."):
                    # Configuration for layer display
                    layer_heights = {
                        'Conv2D': 100,
                        'MaxPooling2D': 80, 
                        'Dense': 60,
                        'Dropout': 40,
                        'Flatten': 30,
                        'BatchNormalization': 40,
                        'Input': 120,
                        'LSTM': 110,
                        'GRU': 110,
                        'Bidirectional': 110,
                        'Activation': 35,
                        'Add': 50,
                        'Concatenate': 50,
                        'GlobalAveragePooling2D': 70,
                    }
                    
                    # Colors for different layer types (with improved color scheme)
                    colors = {
                        'Conv2D': (100, 149, 237),  # Cornflower Blue
                        'MaxPooling2D': (65, 105, 225),  # Royal Blue
                        'Dense': (50, 205, 50),  # Lime Green
                        'Dropout': (220, 220, 220),  # Light Gray
                        'Flatten': (255, 165, 0),  # Orange
                        'BatchNormalization': (186, 85, 211),  # Medium Orchid
                        'Input': (255, 99, 71),  # Tomato
                        'LSTM': (255, 215, 0),  # Gold
                        'GRU': (218, 165, 32),  # Goldenrod
                        'Bidirectional': (139, 69, 19),  # Saddle Brown
                        'Activation': (152, 251, 152),  # Pale Green
                        'Add': (240, 128, 128),  # Light Coral
                        'Concatenate': (135, 206, 250),  # Light Sky Blue
                        'GlobalAveragePooling2D': (221, 160, 221),  # Plum
                    }
                    
                    default_height = 60
                    default_color = (200, 200, 200)
                    
                    # Larger canvas for better visibility
                    img_width = 1200
                    img_height = 800
                    
                    # Create a high-resolution image with anti-aliasing
                    img = Image.new('RGB', (img_width, img_height), color='white')
                    draw = ImageDraw.Draw(img)
                    
                    # Try to load fonts with fallbacks
                    try:
                        title_font = ImageFont.truetype("arial.ttf", 24)
                        layer_font = ImageFont.truetype("arial.ttf", 16)
                        detail_font = ImageFont.truetype("arial.ttf", 12)
                    except:
                        try:
                            # Try system fonts if arial.ttf is not available
                            system_fonts = ImageFont.truetype(ImageFont.load_default().path, 24)
                            title_font = system_fonts
                            layer_font = ImageFont.truetype(ImageFont.load_default().path, 16)
                            detail_font = ImageFont.truetype(ImageFont.load_default().path, 12)
                        except:
                            title_font = ImageFont.load_default()
                            layer_font = ImageFont.load_default()
                            detail_font = ImageFont.load_default()
                    
                    # Draw title and legend
                    draw.text((img_width//2 - 150, 20), "Neural Network Architecture", fill=(0, 0, 0), font=title_font)
                    
                    # Create legend
                    legend_x = 50
                    legend_y = 80
                    legend_spacing = 30
                    legend_size = 20
                    
                    # Draw legend entries for the most common layer types
                    common_layers = ['Conv2D', 'MaxPooling2D', 'Dense', 'Dropout', 'Flatten', 'BatchNormalization']
                    for i, layer_type in enumerate(common_layers):
                        color = colors.get(layer_type, default_color)
                        
                        # Draw legend box
                        draw.rectangle([legend_x, legend_y + i*legend_spacing, 
                                       legend_x + legend_size, legend_y + i*legend_spacing + legend_size], 
                                       fill=color, outline=(0, 0, 0))
                        
                        # Draw legend text
                        draw.text((legend_x + legend_size + 10, legend_y + i*legend_spacing), 
                                 layer_type, fill=(0, 0, 0), font=detail_font)
                    
                    # Get the number of layers in the model
                    num_layers = len(st.session_state.current_model.layers)
                    
                    # Limit to a reasonable number for clarity, with a scroll warning if needed
                    max_display_layers = min(num_layers, 20)
                    
                    if num_layers > max_display_layers:
                        warning_text = f"Note: Showing first {max_display_layers} of {num_layers} layers"
                        draw.text((img_width//2 - 150, 60), warning_text, fill=(255, 0, 0), font=detail_font)
                    
                    # Calculate spacing between layers
                    layer_spacing = (img_width - 200) // (max_display_layers + 1)
                    
                    # Vertical position for the main flow
                    main_y = img_height // 2 + 50
                    
                    # Draw connections and layers
                    for i in range(max_display_layers):
                        layer = st.session_state.current_model.layers[i]
                        layer_type = layer.__class__.__name__
                        
                        # Get layer configuration info
                        height = layer_heights.get(layer_type, default_height)
                        color = colors.get(layer_type, default_color)
                        
                        # Calculate positions
                        x = 100 + (i + 1) * layer_spacing
                        y = main_y
                        
                        # Draw layer box with rounded corners
                        draw.rounded_rectangle([x-70, y-height//2, x+70, y+height//2], 
                                             radius=10, fill=color, outline=(0, 0, 0))
                        
                        # Draw layer name
                        draw.text((x-60, y-height//4-10), f"{layer_type}", fill=(0, 0, 0), font=layer_font)
                        
                        # Safely get output shape - Fix for the error
                        try:
                            if hasattr(layer, 'output_shape') and layer.output_shape is not None:
                                output_shape = str(layer.output_shape)
                                if len(output_shape) > 20:  # Truncate if too long
                                    output_shape = output_shape[:17] + "..."
                            else:
                                output_shape = "Unknown"
                        except:
                            output_shape = "Unknown"
                        
                        # Draw layer details - output shape
                        draw.text((x-60, y), f"Out: {output_shape}", fill=(0, 0, 0), font=detail_font)
                        
                        # Safely get parameters - some layers might not have parameters
                        try:
                            params = layer.count_params()
                        except:
                            params = 0
                            
                        # Draw parameter count
                        draw.text((x-60, y+height//4), f"Params: {params:,}", fill=(0, 0, 0), font=detail_font)
                        
                        # Draw connections between layers
                        if i > 0:
                            prev_x = 100 + i * layer_spacing
                            prev_layer_type = st.session_state.current_model.layers[i-1].__class__.__name__
                            prev_height = layer_heights.get(prev_layer_type, default_height)
                            
                            # Calculate connection points
                            start_x = prev_x + 70
                            start_y = main_y
                            end_x = x - 70
                            end_y = main_y
                            
                            # Draw an arrow between layers
                            draw.line([(start_x, start_y), (end_x, end_y)], fill=(0, 0, 0), width=3)
                            
                            # Draw arrowhead
                            arrow_size = 10
                            draw.polygon([(end_x, end_y), 
                                        (end_x - arrow_size, end_y - arrow_size//2),
                                        (end_x - arrow_size, end_y + arrow_size//2)], 
                                        fill=(0, 0, 0))
                    
                    # Add model summary at the bottom
                    summary_y = main_y + 200
                    
                    # Safely get total params
                    try:
                        total_params = st.session_state.current_model.count_params()
                    except:
                        total_params = 0
                    
                    # Safely extract model information
                    try:
                        import tensorflow.keras.backend as K
                        trainable_params = sum([K.count_params(w) for w in st.session_state.current_model.trainable_weights])
                        non_trainable_params = sum([K.count_params(w) for w in st.session_state.current_model.non_trainable_weights])
                    except:
                        trainable_params = 0
                        non_trainable_params = 0
                    
                    # Draw model summary
                    draw.text((50, summary_y), f"Model Summary:", fill=(0, 0, 0), font=title_font)
                    draw.text((50, summary_y + 40), f"Total parameters: {total_params:,}", fill=(0, 0, 0), font=layer_font)
                    draw.text((50, summary_y + 70), f"Trainable parameters: {trainable_params:,}", fill=(0, 0, 0), font=layer_font)
                    draw.text((50, summary_y + 100), f"Non-trainable parameters: {non_trainable_params:,}", fill=(0, 0, 0), font=layer_font)
                    
                    # Save the image
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    buf.seek(0)
                    
                    # Display the image
                    st.image(buf, caption='Enhanced Model Architecture Visualization', use_container_width=True)
    
        st.markdown("## 📥 Export Options")
        
        col1, col2 = st.columns(2)
        with col1:
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

st.markdown("---") 

st.markdown(
    """
    <div style="text-align: center; font-size: 16px;">
        <strong>Generic Audio Classifier Application</strong> | A modern and user flexible Streamlit app for envirnoment audio classification. | Lokesh Bhaskar
    </div>
    """,
    unsafe_allow_html=True
)

repo_url = "https://github.com/LokeshBhaskarNR/Generic-Audio-Classifier"
username_url = "https://github.com/LokeshBhaskarNR"

animated_html = f"""
    <style>
        @keyframes fadeIn {{
            0% {{ opacity: 0; transform: translateY(-10px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}

        .github-container {{
            text-align: center;
            margin-top: 20px;
            padding: 10px;
            border-radius: 10px;
            background-color: #262730;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            animation: fadeIn 1s ease-in-out;
        }}

        .github-container a {{
            text-decoration: none;
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            display: block;
            margin: 5px 0;
            transition: color 0.3s ease-in-out;
        }}

        .github-container a:hover {{
            color: #1E90FF;
        }}
    </style>
"""

st.markdown(animated_html, unsafe_allow_html=True)

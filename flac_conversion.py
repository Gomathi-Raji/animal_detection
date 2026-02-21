import os
from pydub import AudioSegment

SOURCE_DIR = "DATASET"
TARGET_DIR = "DATASET_FLAC"

def convert_wav_to_flac(source_dir, target_dir):
    for root, _, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        target_folder = os.path.join(target_dir, relative_path)
        os.makedirs(target_folder, exist_ok=True)

        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                flac_path = os.path.join(target_folder, file.replace(".wav", ".flac"))

                audio = AudioSegment.from_wav(wav_path)
                audio.export(flac_path, format="flac")

                print(f"Converted: {wav_path} → {flac_path}")

convert_wav_to_flac(SOURCE_DIR, TARGET_DIR)
print("\n✅ Conversion complete! All WAV files are now in FLAC format.")

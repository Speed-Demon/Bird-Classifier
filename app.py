import os
import io
import librosa
import numpy as np
import joblib
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

# Load the trained model 
try:
    model = joblib.load('/Users/rushikesh/Desktop/Jay/AOML/project/best_model_aryan.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the saved LabelEncoder

label_encoder = joblib.load("/Users/rushikesh/Desktop/Jay/AOML/project/top_100_species/label_encoder.pkl")

# Load the scaler
try:
    scaler = joblib.load('scaler.pkl')
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

def extract_features(audio_data, sr=22050, duration=5.0):
    """Extracts features from audio data."""
    try:
        y, sr = librosa.load(io.BytesIO(audio_data), sr=sr, duration=duration)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.mean(mfccs_delta, axis=1),
            np.mean(mfccs_delta2, axis=1),
            np.mean(chroma_stft, axis=1),
            np.mean(rms, axis=1),
            np.mean(spectral_centroid, axis=1),
            np.mean(spectral_bandwidth, axis=1),
            np.mean(spectral_rolloff, axis=1),
            np.mean(zero_crossing_rate, axis=1)
        ])
        return features
    except Exception as e:
        print(f"Error processing audio data: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_audio():
    if request.method == 'POST':
        try:
            if 'audio' not in request.files:
                return render_template('index.html', error='No audio file uploaded.')

            audio_file = request.files['audio']

            if audio_file.filename == '':
                return render_template('index.html', error='No audio file selected.')

            if audio_file:
                audio_data = audio_file.read()
                features = extract_features(audio_data)

                if features is not None:
                    if scaler is not None:
                        features_scaled = scaler.transform(features.reshape(1, -1))
                    else:
                        features_scaled = features.reshape(1, -1)

                    if model is not None:
                        prediction = model.predict(features_scaled)[0]

                        # labels = ['Black_footed_Albatross', 'Laysan_Albatross', 'Groove_billed_Ani', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Northern_Cardinal', 'Chuck_will_s_widow', 'Brandt_Cormorant', 'Bronzed_Cowbird', 'Cape_Glossy_Starling', 'Caspian_Tern', 'Common_Tern', 'Crested_Auklet', 'Dark_eyed_Junco', 'Downy_Woodpecker', 'Eastern_Towhee', 'Eurasian_Collared_Dove', 'Fairy_Tern', 'Finch', 'Forsters_Tern', 'Gadwall', 'Geococcyx', 'Golden_Winged_Warbler', 'Gray_Catbird', 'Gray_Kingbird', 'Great_Cormorant', 'Great_Grey_Shrike', 'Green_Jay', 'Guillemot', 'Heermann_Gull', 'Horned_Lark', 'House_Sparrow', 'Iceland_Gull', 'Ivory_Gull', 'Javan_Myna', 'Kentucky_Warbler', 'Kiwi', 'Laughing_Gull', 'Le_Conte_s_Thrasher', 'Least_Tern', 'Loggerhead_Shrike', 'Magnolia_Warbler', 'Marsh_Wren', 'Merlin', 'Mockingbird', 'Mourning_Dove', 'Nightingale']

                        species_name = label_encoder.inverse_transform([prediction])[0]

                        return render_template('index.html', prediction=species_name, audio_data=audio_data)
                    else:
                        return render_template('index.html', error='Model not loaded.')
                else:
                    return render_template('index.html', error='Error processing audio.')
        except Exception as e:
            print(f"Error during upload: {e}")
            return render_template('index.html', error=f'An error occurred: {e}')

    return render_template('index.html')

# @app.route('/audio')
# def play_audio():
#     audio_data = request.args.get('audio_data')
#     return send_file(io.BytesIO(audio_data), mimetype='audio/wav')

if __name__ == '__main__':
    app.run(debug=True)
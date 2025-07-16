import customtkinter as ctk
import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import joblib
import os
import parselmouth
import tkinter.filedialog as fd

# === Settings ===
MODEL_PATH = 'parkinsons_model.pkl'
RECORD_FILE = 'voice_input.wav'
DURATION = 5
SAMPLE_RATE = 44100

# === Voice Recording ===
def record_voice():
    try:
        result_label.configure(text="🎙️ Recording... Please speak.")
        app.update()
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        write(RECORD_FILE, SAMPLE_RATE, recording)
        result_label.configure(text="✅ Voice recorded successfully.")
    except Exception as e:
        result_label.configure(text=f"❌ Recording Error: {str(e)}")

# === Upload Recording ===
def upload_recording():
    try:
        file_path = fd.askopenfilename(title="Select WAV File", filetypes=[("WAV files", "*.wav")])
        if file_path:
            global RECORD_FILE
            RECORD_FILE = file_path
            result_label.configure(text="✅ File uploaded successfully.")
        else:
            result_label.configure(text="⚠️ No file selected.")
    except Exception as e:
        result_label.configure(text=f"❌ Upload Error: {str(e)}")

# === Feature Extraction ===
def extract_features(filename):
    try:
        sound = parselmouth.Sound(filename)
        pitch = sound.to_pitch()
        point_process = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
        harmonicity = sound.to_harmonicity_cc()

        def safe_call(obj_list, func, *args):
            try:
                return parselmouth.praat.call(obj_list, func, *args)
            except:
                return 0.0

        meanF0 = safe_call(pitch, "Get mean", 0, 0, "Hertz")
        minF0 = safe_call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        maxF0 = safe_call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        jitter_percent = safe_call([sound, point_process], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_abs = safe_call([sound, point_process], "Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rap = safe_call([sound, point_process], "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq = safe_call([sound, point_process], "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddp = 3 * rap if rap != 0.0 else 0.0
        shimmer = safe_call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = safe_call([sound, point_process], "Get shimmer (dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3 = safe_call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5 = safe_call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11 = safe_call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        dda = 3 * apq3 if apq3 != 0.0 else 0.0
        hnr = safe_call(harmonicity, "Get mean", 0, 0)
        nhr = 1 / (10 ** (hnr / 10)) if hnr > 0 else 0.0

        features = {
            'MDVP:Fo(Hz)': meanF0,
            'MDVP:Fhi(Hz)': maxF0,
            'MDVP:Flo(Hz)': minF0,
            'MDVP:Jitter(%)': jitter_percent,
            'MDVP:Jitter(Abs)': jitter_abs,
            'MDVP:RAP': rap,
            'MDVP:PPQ': ppq,
            'Jitter:DDP': ddp,
            'MDVP:Shimmer': shimmer,
            'MDVP:Shimmer(dB)': shimmer_db,
            'Shimmer:APQ3': apq3,
            'Shimmer:APQ5': apq5,
            'MDVP:APQ': apq11,
            'Shimmer:DDA': dda,
            'NHR': nhr,
            'HNR': hnr,
            'RPDE': 0.0,
            'DFA': 0.0,
            'spread1': 0.0,
            'spread2': 0.0,
            'D2': 0.0,
            'PPE': 0.0
        }

        return pd.DataFrame([features])
    except Exception as e:
        result_label.configure(text=f"❌ Feature Extraction Error: {str(e)}")
        return None

# === Prediction ===
def predict():
    if not os.path.exists(RECORD_FILE):
        result_label.configure(text="⚠️ Please record or upload a voice file.")
        return

    features = extract_features(RECORD_FILE)
    if features is None:
        return

    try:
        model = joblib.load(MODEL_PATH)
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred]

        if pred == 1:
            result = f"✅ Likely Healthy\nConfidence: {prob:.2f}"
        else:
            result = f"🧠 Parkinson's Detected\nConfidence: {prob:.2f}"

        result_label.configure(text=result)
    except Exception as e:
        result_label.configure(text=f"❌ Prediction Error: {str(e)}")

# === GUI Setup ===
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("🎙️ Parkinson's Voice Predictor")
app.geometry("480x380")
app.resizable(False, False)

title = ctk.CTkLabel(app, text="🎙️ Parkinson's Disease Voice Screening", font=ctk.CTkFont(size=20, weight="bold"))
title.grid(row=0, column=0, columnspan=2, pady=(20, 10), padx=10)

btn1 = ctk.CTkButton(app, text="🎙️ Record Voice", font=ctk.CTkFont(size=16,weight="bold"), command=record_voice, width=200, height=50)
btn1.grid(row=1, column=0, padx=20, pady=10, sticky="e")

btn_upload = ctk.CTkButton(app, text="📂 Upload Recording", font=ctk.CTkFont(size=16,weight="bold"), command=upload_recording, width=200, height=50)
btn_upload.grid(row=1, column=1, padx=20, pady=10, sticky="w")

btn2 = ctk.CTkButton(app, text="🔍 Predict", font=ctk.CTkFont(size=16, weight="bold"), command=predict, width=440, height=45)
btn2.grid(row=2, column=0, columnspan=2, pady=15)

result_label = ctk.CTkLabel(app, text="", font=ctk.CTkFont(size=16), text_color="#00ffaa", wraplength=450, justify="center")
result_label.grid(row=3, column=0, columnspan=2, pady=30)

btn3 = ctk.CTkButton(app, text="❌ Quit", command=app.destroy, width=120, height=35, fg_color="red", hover_color="#990000")
btn3.grid(row=4, column=0, columnspan=2)

footer = ctk.CTkLabel(app, text="© Shivam Verma", font=ctk.CTkFont(size=12), text_color="gray")
footer.grid(row=5, column=0, columnspan=2, pady=15)

app.mainloop()
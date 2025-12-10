# chatbot_gui.py
# Tkinter GUI pour ton chatbot émotionnel
# Place reponse.json dans le même dossier avant d'exécuter.

import os
import json
import random
from datetime import datetime
import unicodedata
import tkinter as tk
from tkinter import scrolledtext, END
import numpy as np

# ML libs (optionnel : datasets)
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
    has_tf = True
except Exception:
    has_tf = False

# Optional: use HuggingFace dataset if available to improve model
try:
    from datasets import load_dataset
    has_datasets = True
except Exception:
    has_datasets = False

# ---------------- files ----------------
RESPONSE_FILE = "reponse.json"
MEMORY_FILE = "memory.json"

# ---------------- UI settings ----------------
EMOJI_UI = {
    "fer7": "😄",
    "7zin": "😢",
    "ghaDab": "😡",
    "khouf": "😨",
    "7ob": "❤️",
    "neutre": "😐",
    "fatigue": "😴",
    "surprise": "😲",
    "motivation": "💪"
}
COLOR_UI = {
    "fer7": "#FFD700",
    "7zin": "#1E90FF",
    "ghaDab": "#FF4500",
    "khouf": "#9400D3",
    "7ob": "#FF1493",
    "neutre": "#DDDDDD",
    "fatigue": "#00CED1",
    "surprise": "#7CFC00",
    "motivation": "#32CD32"
}

# ---------------- Load responses ----------------
if not os.path.exists(RESPONSE_FILE):
    print(f"Erreur: {RESPONSE_FILE} introuvable dans le dossier courant.")
    raise SystemExit

with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
    responses_raw = json.load(f)

# Normalize responses to dict: {emotion: [advice, ...]}
responses = {}
if isinstance(responses_raw, dict):
    # assume mapping emotion -> list
    for k, v in responses_raw.items():
        if isinstance(v, list):
            responses[k] = v
        else:
            responses[k] = [v]
elif isinstance(responses_raw, list):
    # list of {"emotion": "...", "advice": "..."}
    for item in responses_raw:
        emo = item.get("emotion", "neutre")
        advice = item.get("advice") or item.get("reply") or item.get("text", "")
        responses.setdefault(emo, []).append(advice)
else:
    print("Format reponse.json non supporté.")
    raise SystemExit

# ---------------- Utility ----------------
def remove_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# ---------------- Prepare / train model ----------------
# We'll try to use French-emotion if available (big dataset) else fallback to tiny sample.
tokenizer = None
max_len = 20
unique_labels = []
model = None

def build_and_train_model(texts, labels, epochs=3, batch_size=128):
    global tokenizer, max_len, unique_labels, model
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_len = max(1, max(len(s) for s in sequences))
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    unique_labels = list(sorted(set(labels)))
    label_to_index = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_index[l] for l in labels])

    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        Bidirectional(LSTM(64)),
        Dense(len(unique_labels), activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(padded, y, epochs=epochs, batch_size=batch_size, verbose=0)
    print("Modèle entraîné (labels):", unique_labels)

# Try to load big dataset if available & tf present
if has_tf and has_datasets:
    try:
        print("Tentative de chargement du dataset JusteLeo/French-emotion (si présent)...")
        ds = load_dataset("JusteLeo/French-emotion")
        # labels are strings like 'joie','colère',...
        # mapping to our internal labels:
        map_lbl = {
            "joie": "fer7",
            "colère": "ghaDab",
            "tristesse": "7zin",
            "amour": "7ob",
            "peur": "khouf",
            "surprise": "surprise",
            "neutre": "neutre"
        }
        texts = [x["text"] for x in ds["train"]]
        labels = [map_lbl.get(remove_accents(x["label"]).lower(), "neutre") for x in ds["train"]]
        build_and_train_model(texts, labels, epochs=2, batch_size=256)
    except Exception as e:
        print("Impossible charger French-emotion:", e)
        # fallback below
if model is None and has_tf:
    # fallback minimal training so tokenizer/model exist (small examples)
    sample_texts = [
        "Je suis tellement heureux aujourd'hui !",
        "Je me sens triste et seul.",
        "J'ai peur de demain.",
        "Je suis en colère contre cette situation.",
        "Je suis fatigué et épuisé."
    ]
    sample_labels = ["fer7", "7zin", "khouf", "ghaDab", "fatigue"]
    build_and_train_model(sample_texts, sample_labels, epochs=60, batch_size=8)

# ---------------- Keywords rules for forcing emotion ----------------
POSITIVE_KEYWORDS = ["heureux", "content", "sourire", "joyeux", "ravi", "fer7"]
NEGATIVE_KEYWORDS = ["triste", "7zin", "déprimé", "mal", "fatigué", "fatigue", "peur", "stress",
                     "colère", "colere", "melancol", "sol", "seul"]

def rule_based_emotion(text):
    t = text.lower()
    for w in POSITIVE_KEYWORDS:
        if w in t:
            return "fer7"
    for w in NEGATIVE_KEYWORDS:
        if w in t:
            if any(x in t for x in ["triste", "7zin", "melancol", "seul", "sol", "fatigu"]):
                return "7zin"
            if "col" in w or "colere" in t or "colère" in t:
                return "ghaDab"
            if "peur" in t:
                return "khouf"
            if "fatig" in t:
                return "fatigue"
            return "7zin"
    return None

def predict_emotions(text):
    # first try rule-based
    forced = rule_based_emotion(text)
    if forced:
        return [(forced, 1.0)]
    # else model if available
    if has_tf and model is not None and tokenizer is not None:
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=max_len, padding='post')
        pred = model.predict(pad, verbose=0)[0]
        probs = [(unique_labels[i], float(pred[i])) for i in range(len(unique_labels))]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs
    # fallback neutral
    return [("neutre", 1.0)]

# ---------------- Choose reply ----------------
def choose_reply(emotions):
    dominant = emotions[0][0]
    pool = responses.get(dominant) or responses.get("neutre") or ["Je suis là pour toi."]
    return random.choice(pool), dominant

# ---------------- Memory ----------------
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
    except Exception:
        memory = []
else:
    memory = []

def save_memory(text, emotions, reply):
    memory.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "emotions": emotions,
        "reply": reply
    })
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

# ---------------- Fine-tune lightweight ----------------
def fine_tune_light(text, dominant):
    if has_tf and model is not None and tokenizer is not None:
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=max_len, padding='post')
        idx = unique_labels.index(dominant) if dominant in unique_labels else 0
        model.fit(pad, np.array([idx]), epochs=1, verbose=0)

# ---------------- Tkinter UI ----------------
root = tk.Tk()
root.title("AI Psychologue — GUI")
root.geometry("600x680")
root.configure(bg="#1b1b1b")

chat = scrolledtext.ScrolledText(root, bg="#111111", fg="#FFFFFF", font=("Helvetica", 12))
chat.pack(padx=12, pady=12, fill="both", expand=True)

entry_frame = tk.Frame(root, bg="#1b1b1b")
entry_frame.pack(fill="x", padx=12, pady=(0,12))

entry = tk.Entry(entry_frame, font=("Helvetica", 14))
entry.pack(side="left", fill="x", expand=True, padx=(0,8))

def insert_bot(text, emotion):
    emoji = EMOJI_UI.get(emotion, "🤖")
    color = COLOR_UI.get(emotion, "#AAAAAA")
    chat.insert(END, f"Bot {emoji}: {text}\n")
    # color the last line (simple approach)
    chat.tag_add(emotion, "end-2l", "end-1l")
    chat.tag_config(emotion, foreground=color)

def on_send(event=None):
    user_text = entry.get().strip()
    if not user_text:
        return
    # show user
    chat.insert(END, f"Toi: {user_text}\n", "user")
    chat.tag_config("user", foreground="#00E5FF")
    entry.delete(0, END)

    # predict
    emotions = predict_emotions(user_text)
    # print probs in chat header
    probs_str = " | ".join([f"{emo}: {prob:.2f}" for emo, prob in emotions[:4]])
    chat.insert(END, f"(Probabilités) {probs_str}\n", "meta")
    chat.tag_config("meta", foreground="#888888")

    # choose reply
    reply, dominant = choose_reply(emotions)
    insert_bot(reply, dominant)

    # save and fine-tune
    save_memory(user_text, emotions, reply)
    fine_tune_light(user_text, dominant)

entry.bind("<Return>", on_send)

send_btn = tk.Button(entry_frame, text="Envoyer", command=on_send, bg="#4CAF50", fg="white", font=("Helvetica", 12))
send_btn.pack(side="right")

# prefill welcome
chat.insert(END, "Bot 🤖: Salem! 7kili chnowa 7assitek tawa (tape 'exit' pour quitter)\n", "meta")
chat.tag_config("meta", foreground="#888888")

root.mainloop()

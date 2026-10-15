import io
import json
import wave

import matplotlib.pyplot as plt
import mido
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(page_title="AI Music Composer", page_icon="🎵")

with open("lstm_config.json") as f:
    CFG = json.load(f)
with open("vocab.json") as f:
    VOCAB = json.load(f)
with open("melodies_tokens.json") as f:
    MELODIES = json.load(f)

TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}
SEQ_LEN = CFG["seq_len"]


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden, layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers,
                            dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        out, _ = self.lstm(self.embed(x))
        return self.fc(self.drop(out[:, -1, :]))


@st.cache_resource
def load_model():
    m = MusicLSTM(len(VOCAB), CFG["embed_dim"], CFG["hidden"], CFG["layers"], CFG["dropout"])
    m.load_state_dict(torch.load("lstm_music.pt", map_location="cpu"))
    m.eval()
    return m


@torch.no_grad()
def generate(model, primer, length, temperature, seed):
    gen = torch.Generator().manual_seed(int(seed))
    ids = [TOK2ID[t] for t in primer][-SEQ_LEN:]
    out = []
    for _ in range(length):
        x = torch.tensor([ids[-SEQ_LEN:]])
        probs = F.softmax(model(x)[0] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=gen).item()
        ids.append(nxt)
        out.append(ID2TOK[nxt])
    return out


def synth_wav(tokens, bpm=100, sr=22050):
    spb = 60.0 / bpm
    chunks = []
    for t in tokens:
        p, d = t.split("_")
        freq = 440.0 * 2 ** ((int(p) - 69) / 12)
        n = max(int(sr * float(d) * spb), 1)
        x = np.arange(n) / sr
        tone = (0.6 * np.sin(2 * np.pi * freq * x)
                + 0.25 * np.sin(2 * np.pi * 2 * freq * x)
                + 0.1 * np.sin(2 * np.pi * 3 * freq * x))
        env = np.ones(n)
        a = max(min(int(0.01 * sr), n // 4), 1)
        r = max(min(int(0.08 * sr), n // 2), 1)
        env[:a] = np.linspace(0, 1, a)
        env[-r:] *= np.linspace(1, 0, r)
        chunks.append(tone * env * 0.3)
    audio = np.concatenate(chunks)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def tokens_to_midi_bytes(tokens, bpm=100, ppq=480):
    mid = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    for t in tokens:
        p, d = t.split("_")
        track.append(mido.Message("note_on", note=int(p), velocity=80, time=0))
        track.append(mido.Message("note_off", note=int(p), velocity=0, time=int(float(d) * ppq)))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def piano_roll_figure(tokens):
    t0, xs, ys, ws = 0.0, [], [], []
    for t in tokens:
        p, d = t.split("_")
        d = float(d)
        xs.append(t0)
        ys.append(int(p))
        ws.append(d)
        t0 += d
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.barh(ys, ws, left=xs, height=0.8, color="darkorange")
    ax.set_xlabel("Time (beats)")
    ax.set_ylabel("MIDI pitch")
    fig.tight_layout()
    return fig


model = load_model()

st.title("AI Music Composer (LSTM)")
st.write(
    "An LSTM neural network trained on Bach chorale melodies. It starts from the first "
    f"{SEQ_LEN} notes of a chosen chorale and composes the notes that follow."
)

chorale = st.number_input("Starting chorale number", min_value=1, max_value=len(MELODIES), value=1, step=1)
length = st.slider("Number of notes to compose", 16, 128, 64)
temperature = st.slider("Creativity (temperature)", 0.5, 1.5, 1.0, 0.1,
                        help="Low = safer and more repetitive. High = more adventurous and more mistakes.")
bpm = st.slider("Tempo (beats per minute)", 60, 160, 100)
seed = st.number_input("Random seed", value=42, step=1)
include_primer = st.checkbox("Also play the starting notes (these are real Bach notes)", value=False)

if st.button("Compose"):
    primer = MELODIES[int(chorale) - 1][:SEQ_LEN]
    composed = generate(model, primer, length, temperature, seed)
    st.session_state["tokens"] = (primer + composed) if include_primer else composed
    st.session_state["bpm"] = bpm

if "tokens" in st.session_state:
    tokens = st.session_state["tokens"]
    play_bpm = st.session_state["bpm"]
    st.subheader("Your melody")
    fig = piano_roll_figure(tokens)
    st.pyplot(fig)
    plt.close(fig)
    st.audio(synth_wav(tokens, play_bpm), format="audio/wav")
    st.download_button("Download as MIDI", data=tokens_to_midi_bytes(tokens, play_bpm),
                       file_name="composed_melody.mid", mime="audio/midi")
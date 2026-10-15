# AI Music Composer (LSTM)

An LSTM neural network trained on Bach chorale melodies. Pick a starting chorale and the model composes the notes that follow. The app shows a piano roll, plays the melody in the browser, and lets you download it as MIDI.

**Live demo:** (add your Streamlit link here)

## How it works
- **Data:** 370 soprano melodies from the Bach chorales in the `music21` corpus, transposed to C major or A minor and converted to "pitch_duration" tokens (about 19,000 notes, 175 unique tokens).
- **Baseline:** an order-2 Markov chain.
- **Model:** a 2-layer LSTM (64-dim embedding, 128 hidden units, dropout 0.3) that predicts the next note from the previous 16.
- **Generation:** temperature sampling, starting from the first 16 notes of a chosen chorale.

## Results (validation set: 37 melodies, 1,530 notes)
| Model | Next-note accuracy |
|---|---|
| Markov baseline | 0.367 |
| LSTM | 0.501 |

LSTM validation loss 1.999 (perplexity 7.4).

| 64-note melodies | In scale % | Unique pitches | Avg jump | Max jump |
|---|---|---|---|---|
| Real melodies | 96.4 | 8.4 | 1.92 | 6.8 |
| Markov baseline | 95.0 | 13.8 | 1.85 | 8.7 |
| LSTM (temp 0.7) | 98.0 | 10.2 | 1.68 | 7.6 |
| LSTM (temp 1.0) | 95.2 | 11.0 | 1.99 | 8.1 |
| LSTM (temp 1.3) | 92.2 | 12.5 | 2.74 | 15.7 |

## Run locally
```
python -m venv venv
venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install streamlit numpy matplotlib mido
streamlit run streamlit_app.py
```

## Files
- `streamlit_app.py`: the web app
- `lstm_music.pt`, `vocab.json`, `lstm_config.json`: trained model and settings
- `melodies_tokens.json`: tokenized melodies (supply the starting notes)
- `notebooks/`: data preparation, Markov baseline and LSTM training

## Limitations
Small dataset (one voice from Bach chorales), a short 16-note context, melody only, and results from a single training run.

## Author
Arshad Anwar, BS Computer Science (AI), UET Mardan. Internship project.

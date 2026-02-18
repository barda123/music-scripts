import os
import sys
import numpy as np
import librosa
import soundfile as sf
import pyrubberband as pyrb
from pathlib import Path

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']


def peak_normalise(y, target_peak=0.999):
    """
    Peak-normalize audio to target_peak.
    """
    peak = np.max(np.abs(y))

    if peak == 0:
        return y

    return (y / peak) * target_peak

def detect_root_chroma(y, sr):
    """
    Detect root pitch using chroma CQT.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Average over time
    chroma_mean = np.mean(chroma, axis=1)

    root_index = np.argmax(chroma_mean)

    return root_index  # 0=C, 1=C#, ...


def process_file(filepath, output_dir):
    print(f"\nProcessing: {filepath}")

    y, sr = sf.read(filepath)

    # Ensure float32
    if y.dtype != np.float32:
        y = y.astype(np.float32)

    # Ensure (samples, channels)
    if y.ndim == 1:
        y = y[:, np.newaxis]

    # Mix to mono for analysis only
    y_mono = np.mean(y, axis=1)

    # Trim silence
    y_trim, _ = librosa.effects.trim(y_mono, top_db=40)

    if len(y_trim) < sr * 0.05:
        print("  Too short after trimming. Skipping.")
        return

    root_index = detect_root_chroma(y_trim, sr)
    root_name = NOTE_NAMES[root_index]

    shift = -root_index

    print(f"  Detected root: {root_name}")
    print(f"  Transposing by {shift} semitones")

    y_shifted = pyrb.pitch_shift(
        y,
        sr,
        n_steps=shift,
        rbargs={
            "quality": "high",
            "transients": "crisp"
        }
    )

    y_normalised = peak_normalise(y_shifted)

    output_path = Path(output_dir) / Path(filepath).name
    sf.write(output_path, y_normalised, sr)

    print(f"  Saved: {output_path}")


def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    wav_files = list(Path(input_folder).rglob("*.wav"))

    if not wav_files:
        print("No WAV files found.")
        return

    for wav in wav_files:
        process_file(str(wav), output_folder)


if __name__ == "__main__":

    main(sys.argv[1], sys.argv[2])

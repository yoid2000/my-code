import bob.kaldi
import numpy as np
import soundfile as sf

def identify_voice_frames(wav_path):
    # Load wav file
    signal, sample_rate = sf.read(wav_path)
    
    # Extract MFCC features using bob.kaldi
    mfcc = bob.kaldi.mfcc(signal, sample_rate)
    
    # Apply Voice Activity Detection (VAD) to MFCC features
    vad = bob.kaldi.vad(mfcc)
    
    # Extract frames with voice
    voice_frames = signal[vad]
    
    return voice_frames, sample_rate

def write_voice_frames(voice_frames, sample_rate, output_path):
    # Write voice frames to output wav file
    sf.write(output_path, voice_frames, sample_rate)

# Input and output file paths
input_wav = 'input.wav'
output_wav = 'output.wav'

# Identify voice frames
voice_frames, sample_rate = identify_voice_frames(input_wav)

# Write voice frames to output file
write_voice_frames(voice_frames, sample_rate, output_wav)

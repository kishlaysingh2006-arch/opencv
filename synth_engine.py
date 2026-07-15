import numpy as np
import sounddevice as sd
import time

class HeadlessSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.amplitude = 0.3  
        self.current_frequencies = []
        
        self.notes = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
        }
        
        self.chords = {
            'maj': [0, 4, 7],       
            'm': [0, 3, 7],         
            '7': [0, 4, 7, 10],     
            'maj7': [0, 4, 7, 11],  
            'm7': [0, 3, 7, 10],    
            'dim': [0, 3, 6],       
            'aug': [0, 4, 8],       
            'sus4': [0, 5, 7]       
        }
        
        self.phase = 0.0

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback
        )
        self.stream.start()

    def set_chord(self, root_note, chord_type):
        if root_note == "OFF":
            self.current_frequencies = []
            return

        base_freq = self.notes.get(root_note)
        if not base_freq:
            return

        intervals = self.chords.get(chord_type, [0]) 
        self.current_frequencies = [base_freq * (2 ** (n / 12)) for n in intervals]

    def _audio_callback(self, outdata, frames, time, status):
        if status:
            print(status, flush=True)
            
        if not self.current_frequencies:
            outdata[:] = np.zeros((frames, 1))
            return

        t = (np.arange(frames) + self.phase) / self.sample_rate
        self.phase += frames
        
        wave = np.zeros(frames)
        num_freqs = len(self.current_frequencies)
        
        if num_freqs > 0:
            for freq in self.current_frequencies:
                wave += np.sin(2 * np.pi * freq * t)
            
            wave = (wave / num_freqs) * self.amplitude
        
        outdata[:] = wave.reshape(-1, 1)

    def stop(self):
        self.stream.stop()
        self.stream.close()

if __name__ == "__main__":
    print("Initializing Synth Engine...")
    synth = HeadlessSynth()
    
    try:
        print("Playing C Major...")
        synth.set_chord('C', 'maj')
        time.sleep(2)
        
        print("Silencing...")
        synth.set_chord("OFF", "")
        time.sleep(1)
        
    except KeyboardInterrupt:
        print("\nExiting manually...")
    finally:
        synth.stop()
        print("Audio stream closed safely.")
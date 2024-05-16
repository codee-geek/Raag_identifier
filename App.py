import os
from flask import Flask, request, render_template
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MAX_FILES = 4

def delete_old_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    if len(files) > MAX_FILES:
        # Sort files by creation time and delete the oldest ones
        files = sorted(files, key=lambda x: os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], x)))
        for file in files[:-MAX_FILES]:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

@app.route('/', methods=['GET', 'POST'])
def get_files():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = 'f{}.wav'.format(len(os.listdir(app.config['UPLOAD_FOLDER'])) + 1)  # Unique filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Delete old files if necessary
            delete_old_files()

            # Process the uploaded file
            notes, waveform_plot_path = process_audio(file_path)
            
            # Render the template with the updated data
            return render_template('upload_form.html', notes=notes, waveform_plot=waveform_plot_path)

    # If GET request or if there's an error with the POST request, render the upload form
    return render_template('upload_form.html', notes=[], waveform_plot=None)
def process_audio(file_path):
    sample_rate, waveform = wavfile.read(file_path)
    window_size = 1024
    overlap = 512
    threshold = 0.5

    distinct_frequencies = []
    for window in create_windows(waveform, window_size, overlap):
        frequencies, magnitudes = get_frequencies(window, sample_rate)
        peak_frequencies = find_peak_frequencies(frequencies, magnitudes, threshold)
        distinct_frequencies.extend(remove_duplicate_consecutive_frequencies(peak_frequencies))

    distinct_frequencies = np.unique(distinct_frequencies)
    notes = [frequency_to_note(freq) for freq in distinct_frequencies]
    unique_notes = set(notes)

    waveform_plot_filename = 'w{}.png'.format(len(os.listdir('static')) + 1)  # Unique waveform plot filename
    waveform_plot_path = os.path.join('static', waveform_plot_filename)
    draw_waveform(waveform, sample_rate, 'Waveform', waveform_plot_path)

    return unique_notes, waveform_plot_path


def create_windows(waveform, window_size, overlap):
    num_samples = len(waveform)
    start = 0
    while start < num_samples:
        yield waveform[start:start + window_size]
        start += window_size - overlap

def get_frequencies(window, sample_rate):
    fft_result = np.fft.fft(window)
    frequencies = np.fft.fftfreq(len(window)) * sample_rate
    magnitudes = np.abs(fft_result)
    return frequencies[:len(window)//2], magnitudes[:len(window)//2]

def find_peak_frequencies(frequencies, magnitudes, threshold=0.5):
    peak_indices = np.where(magnitudes > threshold * np.max(magnitudes))[0]
    peak_frequencies = frequencies[peak_indices]
    return peak_frequencies

def draw_waveform(waveform, sample_rate, title, filename):
    time = np.arange(0, len(waveform)) / sample_rate
    plt.plot(time, waveform)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig(filename)
    plt.close()

def remove_duplicate_consecutive_frequencies(frequencies):
    if len(frequencies) == 0:
        return []
    
    unique_frequencies = [frequencies[0]]
    for freq in frequencies[1:]:
        if freq != unique_frequencies[-1]:
            unique_frequencies.append(freq)
    return unique_frequencies

def frequency_to_note(frequency):
    notes_mapping = {
        261.63: 'Sa',  # Middle C
        277.18: 'Re',
        293.66: 'Ga',
        311.13: 'Ma',
        329.63: 'Pa',
        349.23: 'Dha',
        369.99: 'Ni',
        392.00: 'Sa',  # Higher octave Sa
        415.30: 'Re',
        440.00: 'Ga',
        466.16: 'Ma',
        493.88: 'Pa',
    }
    nearest_freq = min(notes_mapping.keys(), key=lambda x: abs(x - frequency))
    return notes_mapping[nearest_freq]

if __name__ == '__main__':
    app.run(debug=True)

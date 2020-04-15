import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import gan as g
import tensorflow as tf
import pygame
import keras.backend.tensorflow_backend as tb
from keras.models import load_model

app = Flask(__name__)

genres = {
    0: 'any',
    1: 'rock',
    2: 'funk',
    3: 'hiphop',
    4: 'jazz',
    5: 'latin',
    6: 'afrobeat',
    7: 'soul',
    8: 'reggae',
    9: 'pop',
    10: 'any_2'
}

pygame.init()

@app.route('/')
def home():
    return render_template('index.html', prediction_text='Choose a genre and generate some drums!')

#Generates a drum track based on the selected genre
@app.route('/generate',methods=['POST'])
def generate():
    print('Generate')

    tb._SYMBOLIC_SCOPE.value = False

    values = [x for x in request.form.values()]
    genre = int(values[0])
    
    #Load the generator's model from its stored location
    model_name = '../drum_gan/models/model_' + genres[genre] + '.h5'
    model = load_model(model_name)
    
    #drum_df = pd.read_csv("../drum_gan/data/drum_tracks.csv")
    #midi_list = drum_df.midi_filename.tolist()
    #notes = g.get_notes(midi_list)
    
    #Loads the notes that were recieved when the model was originally trained
    notes = pickle.load(open('../drum_gan/data/notes/'+genres[genre]+'.txt', 'rb'))

    #Generate the notes for the new drumtrack
    #Create a new MIDI file and save it to be played
    predictions = g.generate(model, notes)
    midi_result = g.create_midi(predictions)
    midi_name = "../drum_gan/results/result_midi.mid"
    midi_result.write('../drum_gan/results/result_midi.mid')
    return render_template('index.html', prediction_text='{} Drumtrack Ready'.format(midi_name))

#Plays the generated drumtrack
@app.route('/play',methods=['POST'])
def play():
    clock = pygame.time.Clock()
    music_file = "../drum_gan/results/result_midi.mid"
    
    #Loads the MIDI file to pygame and attempts to play it
    try:
        pygame.mixer.music.load(music_file)
        print ("Music file %s loaded!" % music_file)
    except pygame.error:
        print ("File %s not found! (%s)" % (music_file, pygame.get_error()))
        return render_template('index.html', prediction_text='Midi was not playable')
    pygame.mixer.music.play()

    #Plays the track until completion
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick()
    return render_template('index.html', prediction_text='Drumtrack finished!')

#Manually stops the MIDI file from playing
@app.route('/stop',methods=['POST'])
def stop():
    pygame.mixer.music.stop()
    return render_template('index.html', prediction_text='Drumtrack stopped!')

#TODO: Remove as it may not be useful
@app.route('/results',methods=['POST'])
def results():
    print('results')

    #data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])
    #output = prediction[0]
    #return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
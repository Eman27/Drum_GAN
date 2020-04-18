import os
import uuid
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import gan as g
import model as m
import tensorflow as tf
import keras.backend.tensorflow_backend as tb
from keras.models import load_model
from absl import logging
logging._warn_preinit_stderr = 0
logging.set_verbosity(logging.DEBUG)

app = Flask(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

genres = {
    0: 'any',
    1: 'rock',
    2: 'funk',
    3: 'jazz',
    4: 'latin'
}

@app.route('/')
def home():
    return render_template('index.html', prediction_text='Choose a genre and generate some drums!')

#Generates a drum track based on the selected genre
@app.route('/generate',methods=['POST'])
def generate():
    logging.info('Begin generating track..')

    tb._SYMBOLIC_SCOPE.value = False

    values = [x for x in request.form.values()]

    genre = int(values[0])
    
    #Load the generator's model from its stored location
    model_name = ROOT_DIR+'/models/model_' + genres[genre] + '.h5'
    model = load_model(model_name)
    
    #Loads the notes that were recieved when the model was originally trained
    logging.info('Retrieve notes from %s', genre)
    notes = pickle.load(open(ROOT_DIR+'/data/notes/'+genres[genre]+'.txt', 'rb'))

    #Generate the notes for the new drumtrack
    #Create a new MIDI file and save it to be played
    logging.info('Generate notes for the new track')
    predictions = g.generate(model, notes)
    midi_result = g.create_midi(predictions)
    filename = str(uuid.uuid4())+".mid"
    midi_name = ROOT_DIR+"/static/"+filename
    #midi_name = ROOT_DIR+"/static/result_midi.mid"
    midi_result.write(midi_name)
    logging.info('New track is created, (%s)',midi_name)
    return render_template('index.html', prediction_text='Drumtrack Generated!', value=filename)

#API
@app.route('/gen',methods=['POST'])
def gen():
    tb._SYMBOLIC_SCOPE.value = False    
    values = [x for x in request.form.values()]
    if len(values) > 1:
       logging.debug("ERROR: Invalid number of genres, expected only one genre or none for all genres")
       return jsonify("Error: Invalid number of inputs")
    elif len(values) == 1 and values[0] != '':
        genre = values[0]
    else:
        genre = 'any'
    logging.info("Begin generating track")

    genre = values[0]
    
    #Load the generator's model from its stored location
    model_name = ROOT_DIR+'/models/model_' + genre + '.h5'
    model = load_model(model_name)
    
    #Loads the notes that were recieved when the model was originally trained
    logging.info('Retrieve notes from %s', genre)
    notes = pickle.load(open(ROOT_DIR+'/data/notes/'+genre+'.txt', 'rb'))
    #Generate the notes for the new drumtrack
    #Create a new MIDI file and save it to be played
    logging.info('Generate notes for the new track')
    predictions = g.generate(model, notes)
    midi_result = g.create_midi(predictions)
    midi_name = ROOT_DIR+"/static/result_midi.mid"
    midi_result.write(midi_name)
    logging.info('New track is created, (%s)',midi_name)
    return jsonify(midi_name)

@app.route('/train',methods=['POST'])
def train():
    values = [x for x in request.form.values()]
    if len(values) > 1:
       logging.debug("ERROR: Invalid number of genres, expected only one genre or none for all genres")
       return jsonify("Error: Invalid number of inputs")
    elif len(values) == 1 and values[0] != '':
        genre = values[0]
        logging.info("Begin training for genre: %s", genre)
    else:
        genre = 'any'
        logging.info("Begin training with all genres")
    tb._SYMBOLIC_SCOPE.value = True
    m.train_gan(genre)
    return jsonify("Training complete.")

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import gan as g

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate',methods=['POST'])
def generate():
    print('Generate')
    #output = 'Generate'
    genre = [x for x in request.form.values()]
    drum_df = pd.read_csv("../drum_gan/data/drum_tracks.csv")
    midi_list = drum_df.midi_filename.tolist()
    notes = g.get_notes(midi_list)
    prediction = model.generate(notes)
    midi_result = g.create_midi(prediction)
    midi_result.write('results/result_midi.mid')
    #print('midi result')   
    return render_template('index.html', prediction_text='It has {}'.format(prediction))

@app.route('/results',methods=['POST'])
def results():
    print('results')

    #data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])
    #output = prediction[0]
    #return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Dropout, Activation
from keras.layers import LSTM, Bidirectional, BatchNormalization, ZeroPadding2D
from keras.models import Sequential, Model, load_model
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
import pretty_midi
import pickle

MIDI_PATH = "../drum_gan/data/"

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
    9: 'pop'
}

roland_to_gm = {
    36 : 36,
    38 : 38,
    40 : 38,
    37 : 38,
    48 : 50,
    50 : 50,
    45 : 47,
    47 : 47,
    43 : 43,
    58 : 43,
    46 : 46,
    26 : 46,
    42 : 42,
    22 : 42,
    44 : 42,
    49 : 49,
    55 : 49,
    57 : 49,
    52 : 49,
    51 : 51,
    59 : 51,
    53 : 51
}

#Parses through a list of midi files and retrieves all of their notes
def get_notes(midi_list):
    notes = []
    for file in midi_list:
        midi = pretty_midi.PrettyMIDI(MIDI_PATH + file)
        for instrument in midi.instruments:
            if instrument.is_drum:
                for note in instrument.notes:
                    #convert Roland MIDI pitch to GM pitch for certain instruments
                    if int(note.pitch) == 22:
                        note.pitch = 42
                    elif int(note.pitch) == 26:
                        note.pitch = 46
                    notes.append((note.pitch))
    return notes

#TODO: REMOVE as it doesnt seem to generate as good of pieces
#Test what happens
def get_notes_new(midi_list):
    notes = []
    for file in midi_list:
        midi = pretty_midi.PrettyMIDI(MIDI_PATH + file)
        for instrument in midi.instruments:
            if instrument.is_drum:
                for note in instrument.notes:
                    note.pitch = roland_to_gm[int(note.pitch)]
                    notes.append((note.pitch))
    return notes

#Prepares the inputs and outputs for the model to train
def prepare_sequences(notes, n_vocab):
    sequence_length = 100
    print("\n**Preparing sequences for training**")

    #List of unique chords and notes
    pitchnames = sorted(set(i for i in notes))

    print("Pitchnames (unique notes/chords from 'notes') at length {}: {}".format(len(pitchnames),pitchnames))
    
    #Enumerate pitchnames into dictionary embedding
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    print("Note to integer embedding created at length {}".format(len(note_to_int)))

    network_input = []
    network_output = []

    #i equals total notes less declared sequence length of LSTM (ie 5000 - 100)
    #Sequence input for each i is list of notes i to end of sequence length (ie 0-100 for i = 0)
    #Sequence output for each i is single note at i + sequence length (ie 100 for i = 0)
    for i in range(0, len(notes) - sequence_length,1):
        sequence_in = notes[i:i + sequence_length] # 100
        sequence_out = notes[i + sequence_length] # 1

        #Enumerate notes and chord sequences with note_to_int enumerated encoding
        #Network input/output is a list of encoded notes and chords based on note_to_int encoding
        #If 100 unique notes/chords, the encoding will be between 0-100
        input_add = [note_to_int[char] for char in sequence_in]
        network_input.append(input_add) # sequence length
        output_add = note_to_int[sequence_out]
        network_output.append(output_add) # single note

    print("Network input and output created with (pre-transform) lengths {} and {}".format(len(network_input),len(network_output)))
    
    n_patterns = len(network_input) # notes less sequence length
    print("Lengths. N Vocab: {} N Patterns: {} Pitchnames: {}".format(n_vocab,n_patterns, len(pitchnames)))
    print("\n**Reshaping for training**")

    #Convert network input/output from lists to numpy arrays
    #Reshape input to (notes less sequence length, sequence length)
    network_input_r = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    #Normalize input
    network_input_r = (network_input_r - (float(n_vocab) / 2)) / (float(n_vocab) / 2)
    
    #Reshape output to (notes less sequence length, unique notes/chords)    
    network_output_r = np_utils.to_categorical(network_output)

    print("Reshaping network input to (notes - sequence length, sequence length) {}".format(network_input_r.shape))
    print("Reshaping network output to (notes - sequence length, unique notes) {}".format(network_output_r.shape))
    return network_input_r, network_output_r, n_patterns, n_vocab, pitchnames

#Creates a new MIDI file from the generated notes
def create_midi(notes):
    new_midi_data = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="Midi Drums" )

    #Manually set time and step for the track
    time = 0
    step = 0.21
    len_notes = len(notes)
    vec_arr = np.random.uniform(50,125,len_notes)
    delta_arr = np.random.uniform(0.2,0.5,len_notes)

    #Generate pretty_midi Note objects for each note
    for i,note_number in enumerate(notes):
        myNote = pretty_midi.Note(velocity=int(vec_arr[i]), pitch=int(note_number), start=time, end=time+delta_arr[i])
        drum.notes.append(myNote)
        time += step
    new_midi_data.instruments.append(drum)
    return new_midi_data

class GAN():
    def __init__(self, rows):
        self.seq_length = rows
        self.seq_shape = (self.seq_length, 1)
        self.latent_dim = 1000
        self.disc_loss = []
        self.gen_loss =[]
        
        optimizer = Adam(0.0002, 0.5)

        #Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        #Build the generator
        self.generator = self.build_generator()

        #The generator takes noise as input and generates note sequence
        
        z = Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)

        #For the combined model we will only train the generator
        self.discriminator.trainable = False

        #The discriminator takes generated images as input and determines validity
        validity = self.discriminator(generated_seq)

        #The combined model (stacked generator and discriminator)
        #Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        

    #Builds the discriminator model
    def build_discriminator(self):

        model = Sequential()
        model.add(LSTM(512, input_shape=self.seq_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(512)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        seq = Input(shape=self.seq_shape)
        validity = model(seq)

        return Model(seq, validity)

    #Builds the generator model
    def build_generator(self):

        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.seq_shape), activation='tanh'))
        model.add(Reshape(self.seq_shape))
        model.add(Dropout(0.2))
        model.summary()
        
        noise = Input(shape=(self.latent_dim,))
        seq = model(noise)

        return Model(noise, seq)

    def train(self, genre_dataset, genre, epochs, batch_size=128, sample_interval=25):

        #Load and convert the data
        notes = get_notes(genre_dataset)
        n_vocab = len(set(notes))
        X_train, y_train, n_patterns, n_vocab, pitchnames = prepare_sequences(notes, n_vocab)

        #Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        #Training the model
        for epoch in range(epochs):

            #Training the discriminator
            #Select a random batch of note sequences
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            #Generate a batch of new note sequences
            
            gen_seqs = self.generator.predict(noise)
            

            #Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            #Training the Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            #Train the generator (to have the discriminator label samples as real)
            g_loss = self.combined.train_on_batch(noise, real)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        
        #Save generator's model so the app can generate new tracks
        print('Training complete. Saving model.')
        if genre != None:
            self.generator.save('../drum_gan/models/model_' + genre + '.h5')

            #Save notes to reduce time retrieving them for the app
            pickle.dump(notes,open('../drum_gan/data/notes/any_' + genre + '.txt','wb'))
        else:
            self.generator.save('../drum_gan/models/model_any.h5')

            #Save notes to reduce time retrieving them for the app
            pickle.dump(notes,open('../drum_gan/data/notes/any.txt','wb'))
            
#Generates the notes for the new drum track using the generator model that was trained
def generate(model, input_notes):
    #Get pitch names and store in a dictionary
    notes = input_notes
    
    pitchnames = sorted(set(item for item in notes))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    #Use random noise to generate sequences
    noise = np.random.normal(0, 1, (1, 1000))
    length = len(pitchnames) / 2
    
    predictions = model.predict(noise)
    pred_notes = [x*length+length for x in predictions[0]]
    pred_notes = [int_to_note[int(x)] for x in pred_notes]
    notess = []
    for x in pred_notes:
        notess.append(int(x))
    return notess
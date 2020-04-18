
import sys
import numpy as np
import pandas as pd
import pickle
import gan as g
from absl import logging
logging._warn_preinit_stderr = 0
logging.set_verbosity(logging.DEBUG)

#Path for the groove dataset
DATA_PATH = "../drum_gan/data/info.csv"

def train_gan(input_genre):

    #Set genre to None if we are using all genres for training.
    if input_genre == 'any':
        genre = None
    else:
        genre = input_genre

    groove_df = pd.read_csv(DATA_PATH)

    #Remove rows that are only drum fills
    groove_df = groove_df[groove_df.beat_type != 'fill']
    short = groove_df[groove_df.duration <=30]

    single_styles = groove_df[groove_df['duration'] <= 20]

    styles = groove_df['style'].value_counts()

    #Add the multi-style tracks to their first substyle.
    for s in styles.index:
        if '/' in s:
            style_a,style_b = s.split('/')
            if style_a in styles.index:
                split_style = groove_df.query('style=="'+s+'"')
                groove_df = groove_df.replace({'style':{s:style_a}})
            if style_b in styles.index:
                split_style = groove_df.query('style=="'+s+'"')
                groove_df = groove_df.replace({'style':{s:style_b}})


    #Find number of notes per style
    styles = groove_df['style'].value_counts()
    style_durations = pd.DataFrame(columns=['style','max','min','sum'])
    for s in styles.index:
        style_df = groove_df.query('style=="'+s+'"')
        style_durations = style_durations.append({'style': s, 'max': style_df['duration'].max(),
                                                 'min': style_df['duration'].min(), 'sum': int(style_df['duration'].sum())},
                                                 ignore_index=True)

    #Removed styles that do not have significant duration size
    styles_removed = style_durations[style_durations['sum'] < 100]

    #Generates MIDI list for each genre
    genre_midi_list = {}
    for s in styles.index:
        style_df = groove_df.query('style=="'+s+'"')
        genre_midi_list[s] = style_df.midi_filename.tolist()

    #Determine if the model is on all of the styles or one specified
    if genre is None:
        print('Saving full note list')
        r_list = groove_df.midi_filename.tolist()
    else:
        print('Saving note list for genre: %s', genre)
        r_list = genre_midi_list[genre]

    #Generate GAN model and begin training
    print('Begin training GAN model')
    gan = g.GAN(rows=100)
    gan.train(genre_dataset=r_list, genre=genre, epochs=5, batch_size=32, sample_interval=1)


def main(argv):

    genre = None
    if len(argv) == 1:
        genre = argv[0]

    groove_df = pd.read_csv(DATA_PATH)

    #Remove rows that are only drum fills
    groove_df = groove_df[groove_df.beat_type != 'fill']
    short = groove_df[groove_df.duration <=30]

    #groove_df = groove_df[groove_df.audio_filename.isna() == False]

    single_styles = groove_df[groove_df['duration'] <= 20]

    styles = groove_df['style'].value_counts()

    #Add the multi-style tracks to their first substyle.
    logging.info('Parsing multi-styled genres and re-distributing')
    for s in styles.index:
        if '/' in s:
            style_a,style_b = s.split('/')
            if style_a in styles.index:
                split_style = groove_df.query('style=="'+s+'"')
                #test_df = test_df.replace({'style':{s:style_a}})
                groove_df = groove_df.replace({'style':{s:style_a}})
            if style_b in styles.index:
                split_style = groove_df.query('style=="'+s+'"')
                #test_df = test_df.replace({'style':{s:style_b}})
                groove_df = groove_df.replace({'style':{s:style_b}})

    #Find number of notes per style
    styles = groove_df['style'].value_counts()
    style_durations = pd.DataFrame(columns=['style','max','min','sum'])
    for s in styles.index:
        style_df = groove_df.query('style=="'+s+'"')
        style_durations = style_durations.append({'style': s, 'max': style_df['duration'].max(),
                                                 'min': style_df['duration'].min(), 'sum': int(style_df['duration'].sum())},
                                                 ignore_index=True)

    #Removed styles that do not have significant duration size
    styles_removed = style_durations[style_durations['sum'] < 100]

    #Generates MIDI list for each genre
    genre_midi_list = {}
    for s in styles.index:
        style_df = groove_df.query('style=="'+s+'"')
        genre_midi_list[s] = style_df.midi_filename.tolist()

    #Determine if the model is on all of the styles or one specified
    if genre is None:
        logging.info('Saving full note list')
        r_list = groove_df.midi_filename.tolist()
    #Check that specified genre is valid
    elif genre not in styles.index:
        logging.debug('ERROR: Genre is not valid')
        return 1
    else:
        logging.info('Saving note list for genre: %s', genre)
        r_list = genre_midi_list[genre]

    #Generate GAN model and begin training
    logging.info('Begin training GAN model')
    gan = g.GAN(rows=100)
    gan.train(genre_dataset=r_list, genre=genre, epochs=5, batch_size=32, sample_interval=1)

if __name__ == "__main__":
    main(sys.argv[1:])
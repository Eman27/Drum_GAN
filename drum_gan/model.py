
import numpy as np
import pandas as pd

import scipy.stats as stats
import matplotlib.pyplot as plot
import sklearn
import pickle

import gan as g

#Path for the groove dataset

DATA_PATH = "/home/mark/repos/Springboard/data/info.csv"

def main():

	groove_df = pd.read_csv(DATA_PATH)

	#Remove rows that are only fills
	groove_df = groove_df[groove_df.beat_type != 'fill']
	short = groove_df[groove_df.duration <=30]

	groove_df = groove_df[groove_df.audio_filename.isna() == False]

	single_styles = groove_df[groove_df['duration'] <= 20]

	styles = groove_df['style'].value_counts()

	#style_durations.sort_values(by=['sum'], ascending='True')

	#Add the multi-style to the first substyle.
	#test_df = groove_df
	for s in styles.index:
	    if '/' in s:
	        print(s)
	        style_a,style_b = s.split('/')
	        print(s.split('/'))
	        if style_a in styles.index:
	            print(style_a)
	            split_style = groove_df.query('style=="'+s+'"')
	            #test_df = test_df.replace({'style':{s:style_a}})
	            groove_df = groove_df.replace({'style':{s:style_a}})
	        if style_b in styles.index:
	            print(style_b)
	            split_style = groove_df.query('style=="'+s+'"')
	            #test_df = test_df.replace({'style':{s:style_b}})
	            groove_df = groove_df.replace({'style':{s:style_b}})

	groove_df.to_csv('drum_tracks.csv')

	styles = groove_df['style'].value_counts()
	#styles = test_df['style'].value_counts()
	style_durations = pd.DataFrame(columns=['style','max','min','sum'])
	for s in styles.index:
	    style_df = groove_df.query('style=="'+s+'"')
	    style_durations = style_durations.append({'style': s, 'max': style_df['duration'].max(), 
	                                             'min': style_df['duration'].min(), 'sum': int(style_df['duration'].sum())},
	                                             ignore_index=True)

	styles_removed = style_durations[style_durations['sum'] < 100]

	#Generates MIDI list for each genre
	genre_midi_list = {}
	for s in styles.index:
	    style_df = groove_df.query('style=="'+s+'"')
	    genre_midi_list[s] = style_df.midi_filename.tolist()

	r_list = groove_df.midi_filename.tolist()

	print('Begin training GAN model')
	gan = g.GAN(rows=100)
	gan.train(genre_dataset=r_list, epochs=10, batch_size=32, sample_interval=1)

	print('Training complete. Saving model.')

	pickle.dump(gan, open('model.pkl','wb'))

	model = pickle.load(open('model.pkl','rb'))
	notes = g.get_notes(r_list)
	print(model.generate(notes))

if __name__ == "__main__":
	main()
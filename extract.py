import numpy as np

'''
Data Extraction: Parses through specified file and outputs a numpy array of dictionaries. 
Each dictionary corresponds to a single example. Dictionary format is:

{
	"source" : source,
	"reference": reference,
	"candidate": candidate,
	"bleu": bleu,
	"label": label
}
'''
def extract(filename):
	data = []
	f = open(filename, "r")
	raw = f.read()
	paragraph_split = raw.split('\n\n')
	for example in paragraph_split:
		sample = {}
		line_split = example.splitlines()
		for i, line in enumerate(line_split):
			if i == 0:
				sample["source"] = line
			if i == 1:
				sample["reference"] = line
			if i == 2:
				sample["candidate"] = line
			if i == 3:
				sample["bleu"] = line
			if i == 4:
				sample["label"] = line
		data.append(sample)
	np_data = np.array(data)
	return np_data




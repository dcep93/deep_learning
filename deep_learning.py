from __future__ import print_function

import six.moves.cPickle as pickle
import csv
import gzip
import os
import timeit

import numpy

from models import *

def load_data(dataset, raw_options=None):
	options = {
		'regression': False,
		'header': True,
		'output_value_is_first':False,
		'upsampling':True
	}
	if raw_options:
		options.update(raw_options)

	print('... loading data')

	if os.path.split(dataset)[-1] == 'mnist.pkl.gz':
		with gzip.open(dataset, 'rb') as fh:
			loaded = pickle.load(fh)

		data = numpy.vstack([numpy.c_[i] for i in loaded])
		lookup_dict = {}
		n_out = 1 if options['regression'] else 10

	else:
		with open(dataset) as fh:
			values = [i for i in csv.reader(fh)]

		if options['header']:
			values = values[1:]

		raw_data = numpy.array(values)

		if options['output_value_is_first']:
			raw_data = numpy.roll(raw_data,-1,1)

		data, lookup_dict, n_out = _format_data(raw_data, options['regression'])

		if options['upsampling'] and not options['regression']:
			data = _upsample(data)

	numpy.random.shuffle(data)
			
	loaded_data = {'data':data,'lookup_dict':lookup_dict,'n_out':n_out}

	return loaded_data


def easy_deep_learning(dataset,load_data_options=None,depth=3,build_options=None,train_options=None,predict_count=100):
	loaded_data = load_data(dataset, load_data_options)
	data = loaded_data['data']

	n_in = data.shape[1]-1
	n_out = loaded_data['n_out']

	ratio = float(n_out)/n_in
	features = [int(n_in * ratio**i) for i in range(depth)]

	if n_out == 1:
		model = Regression.build(n_in,features,build_options)
	else:
		model = Classification.build(n_in,n_out,features,build_options)

	model.train(data,train_options)

	results = model.predict(numpy.random.choice(data, predict_count), loaded_data['lookup_dict'])

	return {'loaded_data':loaded_data,'model':model,'results':results}


def _format_data(raw_data,regression):
	x = numpy.column_stack([_unlabel(i) for i in raw_data[:,:-1].T])

	if regression:
		lookup_dict = {}
		y = raw_data[:,-1].astype('float')
		n_out = 1

	else:
		raw_y = raw_data[:,-1]
		values = list(set(raw_y))
		d_reverse = {values[i]:i for i in range(len(values))}
		y = [d_reverse[i] for i in raw_y]
		lookup_dict = {j:i for (i,j) in d_reverse.items()}
		n_out = len(lookup_dict)

	data = numpy.column_stack([x, y])

	return (data, lookup_dict, n_out)


def _unlabel(row):
	try:
		return row.astype('float')
	except ValueError:
		labels = list(set(row))
		lookup_dict = {labels[i]:[int(j==i) for j in range(len(labels))] for i in range(len(labels))}
		return numpy.array([lookup_dict[i] for i in row])


def _upsample(original):
	data = original[original[:,-1].argsort()]

	split = numpy.split(data,numpy.where(numpy.diff(data[:,-1]))[0]+1)

	max_count = max([len(i) for i in split])

	add = numpy.vstack([numpy.repeat(i,[max_count/len(i) - 1 + int(j < (max_count % len(i))) for j in range(len(i))],axis=0) for i in split])

	return numpy.append(original,add,axis=0)

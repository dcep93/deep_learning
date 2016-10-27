from __future__ import print_function

import abc

import six.moves.cPickle as pickle
import timeit

import numpy

import theano
import theano.tensor as T

class MLP(object):
	def __init__(self,n_in,n_out,features=[],**raw_options=None):
		options = {
			"rng": None,
			"L_reg": [0.00,0.0001],
			"learning_rate": 0.01
		}
		if raw_options:
			options.update(raw_options)

		self.n_in = n_in
		self.n_out = n_out
		self.features = features

		self.rng = numpy.random.RandomState(1234) if options['rng'] is None else options['rng']
		self.L_reg = numpy.array(options['L_reg'])
		self.learning_rate = options['learning_rate']

		self.layers = []

		for layer_n_out in features:
			self.new_layer(layer_n_out)

		self.new_layer(n_out)


	@staticmethod
	def load(model_dump='best_model.pkl'):
		with open(model_dump,'rb') as fh:
			model = pickle.load(fh)

		return model

	
	def dump(self,model_dump='best_model.pkl'):
		with open(model_dump,'wb') as fh:
			pickle.dump(self,fh)

		return self


	def new_layer(self,n_out,bounds=None):
		n_in = self.layers[-1].n_out if self.layers else self.n_in

		if bounds is not None:
			layer = TanhLayer(n_in,n_out,bounds=bounds,rng=self.rng)
		else:
			layer = Layer(n_in,n_out)

		self.layers.append(layer)

		return self


	def _get_variables(self):
		input_variable = T.matrix('x')
		output_variable = T.vector('y')

		params = []
		L_cost = 0
		output = input_variable

		for layer in self.layers:
			W = theano.shared(value=layer.W_values,borrow=True)
			b = theano.shared(value=layer.b_values,borrow=True)

			params += [W,b]

			L_cost += sum(layer.L(W,b) * self.L_reg)

			output = layer.output(output,W,b)

		variables = {
			'input_variable':input_variable,
			'output_variable':output_variable,
			'output':output,
			'params':params,
			'L_cost':L_cost
		}
		variables.update(self._class_variables(variables))

		return variables

	
	def train(self,data,**raw_options=None):
		options = {
			"batch_size": 20,
			"max_epochs": 10,
			"distribution": {'train':5./7,'valid':1./7,'test':1./7},
			"model_dump"='best_model.pkl'
		}
		if raw_options:
			options.update(raw_options)

		print('... training the model')

		partitioned_data = self._partition(data,options['distribution'])

		n_train_batches = partitioned_data['train'].shape[0].eval() // options['batch_size']
		n_valid_batches = partitioned_data['valid'].shape[0].eval() // options['batch_size']
		n_test_batches  = partitioned_data['test'] .shape[0].eval() // options['batch_size']

		variables = self._get_variables()

		total_cost = variables['L_cost'] + variables['cost']
		updates = [(param, param - self.learning_rate * T.grad(total_cost,param)) for param in variables['params']]

		index = T.lscalar() 

		test_model = theano.function(
			inputs=[index],
			outputs=variables['score'],
			givens={
				variables['input_variable']:  partitioned_data['test'][:,:-1][index * options['batch_size']: (index + 1) * options['batch_size']],
				variables['output_variable']: partitioned_data['test'][:, -1][index * options['batch_size']: (index + 1) * options['batch_size']]
			}
		)

		validate_model = theano.function(
			inputs=[index],
			outputs=variables['score'],
			givens={
				variables['input_variable']:  partitioned_data['valid'][:,:-1][index * options['batch_size']: (index + 1) * options['batch_size']],
				variables['output_variable']: partitioned_data['valid'][:, -1][index * options['batch_size']: (index + 1) * options['batch_size']]
			}
		)

		train_model = theano.function(
			inputs=[index],
			updates=updates,
			givens={
				variables['input_variable']:  partitioned_data['train'][:,:-1][index * options['batch_size']: (index + 1) * options['batch_size']],
				variables['output_variable']: partitioned_data['train'][:, -1][index * options['batch_size']: (index + 1) * options['batch_size']]
			}
		)

		n_epochs = 5
		epoch_increase = 2
		improvement_threshold = 0.995

		best_validation_score = 0.
		test_score = 0.
		start_time = timeit.default_timer()

		for epoch in range(options['max_epochs']):

			for minibatch_index in range(n_train_batches):
				train_model(minibatch_index)

			validation_score = numpy.mean([validate_model(i) for i in range(n_valid_batches)])

			print('epoch %i, validation score %f%%' % (epoch+1, validation_score*100.))

			if validation_score > best_validation_score:
				if validation_score > best_validation_score * improvement_threshold:
					n_epochs = max(n_epochs, epoch * epoch_increase)

				best_validation_score = validation_score

				if n_test_batches > 0:
					test_scores = [test_model(i) for i in range(n_test_batches)]
					test_score = numpy.mean(test_scores)

					print('\tepoch %i, test score of best model %f%%' % (epoch+1, test_score*100.))

				if options['model_dump']:
					self.dump(options['model_dump'])

			if epoch > n_epochs:
				break

		end_time = timeit.default_timer()

		print('Optimization complete with best validation score of %f%%, test score %f%%' % (best_validation_score * 100., test_score * 100))

		print('The code run for %d epochs, with %f epochs/sec totalling %.1fs' % (epoch+1, (epoch+1.) / (end_time - start_time), end_time - start_time))

		return self

	
	def predict(self,data,lookup_dict={}):
		variables = self._get_variables()

		predict_model = theano.function( 
			inputs=[variables['input_variable']], 
			outputs=variables['label']
		)

		f = numpy.vectorize(lambda bin_num: lookup_dict[bin_num] if bin_num in lookup_dict else bin_num)

		x = data[:,:-1]
		y = f(data[:,-1])

		predicted_values = f(predict_model(x))

		results = {'Predicted values':predicted_values,'Actual values': y}

		return results

	def score(self, data):
		variables = self._get_variables()

		score_model = theano.function(
			inputs=[variables['input_variable'], variables['output_variable']],
			outputs=variables['score']
		)

		x = data[:,:-1]
		y = data[:,-1]

		s = score_model(x, y)
		return s


	@staticmethod
	def _partition(data,distribution):
		s = sum(distribution.values())
		distribution = {i:distribution[i]/float(s) for i in distribution}

		partitioned_data = {}
		n = len(data)
		start = 0
		for i in distribution:
			end = int(start + distribution[i]*n)
			partitioned_data[i] = theano.shared(data[start:end],borrow=True)
			start = end
		return partitioned_data

	@abc.abstractmethod
	def _class_variables(cls,variables):
		pass


class Regression(MLP):
	def __init__(self, n_in, features=[], **raw_options=None):
		options = {
			"bounds": None
		}
		if raw_options:
			options.update(raw_options)

		super(Regression, self).__init__(n_in, 1, features, options)
		self.layers[-1].bounds = options["bounds"]


	@staticmethod
	def _class_variables(cls,variables):
		label = variables['output'].flatten()

		output_variable = variables['output_variable']
		cost = T.mean((label - output_variable)**2) 


		param = 1
		score = (param/(cost+param))

		return {
			"label": label,
			"cost": cost,
			"score": score
		}


class Classification(MLP):
	@staticmethod
	def _class_variables(cls,variables):
		p_y_given_x = T.nnet.softmax(variables['output'])
		label = T.argmax(p_y_given_x, axis=1)

		output_variable = variables['output_variable']
		cost = -T.mean(T.log(p_y_given_x)[T.arange(output_variable.shape[0]), output_variable.astype('int32')])

		score = T.mean(T.eq(label,output_variable))

		return {
			"label": label,
			"cost": cost,
			"score": score
		}


class Layer(object):
	def __init__(self, n_in, n_out, rng=None, bounds=None):
		if rng is None:
			self.W_values = numpy.zeros((n_in,n_out),dtype=theano.config.floatX)
		else:
			bound = numpy.sqrt(6. / (n_in + n_out))
			self.W_values = numpy.asarray(rng.uniform(low=-bound,high=bound,size=(n_in,n_out)))

		self.b_values = numpy.zeros((n_out,),dtype=theano.config.floatX)

		self.n_in = n_in

		self.n_out = n_out


	def L(self,W,b):
		return numpy.array([abs(W).sum(),(W ** 2).sum()])

	def output(self,output,W,b):
		out = T.dot(output,W)+b
		if self.bounds is not None:
			sigm = (T.tanh(out) + 1) / 2
			out = (sigm * (self.bounds[1] - self.bounds[0])) + self.bounds[0]
		return out

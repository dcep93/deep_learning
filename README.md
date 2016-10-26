## deep\_learning documentation

### Typical use:
```python
import pandas



dataset = 'mnist.pkl.gz'
loaded_data = deep_learning.load_data(dataset,regression=False)
data = loaded_data['data']

n_in = data.shape[1]-1
n_out = loaded_data['n_out']

model = deep_learning.Classification.build(n_in,n_out,[500])
model.train(data,batch_size = 20, max_epochs = 10)

results = model.predict(data[:1000])
print(pandas.DataFrame(results).T)





dataset = 'heart.csv'
loaded_data = deep_learning.load_data(dataset,regression=True,header=True)
data = loaded_data['data']

n_in = data.shape[1]-1

model = deep_learning.Regression.build(n_in,[10])
model.train(data,batch_size = 10, max_epochs = 1000)

results = model.predict(data[:100],loaded_data['d'])
print(pandas.DataFrame(results).T)





dataset = 'credit.csv'
output = easy_deep_learning(dataset,regression=True,output_value_is_first=True,depth=3,max_epochs=5,predict_count=100,bounds=[0,1])

```

### Classes and Functions:
```python
class MLP(object) - abstract class
	methods
		__init__(rng=None, L_reg=[0.00,0.0001], learning_rate=0.01,**kwargs)
			args
				numpy.random.RandomState rng
				iter<float> L_reg -> cost = sum([layer.L * L_reg for layer in layers])
				float learning_rate -> multiplicative factor of gradient when updating parameters

		@classmethod
		build(n_in, n_out, features=[], **kwargs)
			args
				int n_in
				int n_out
				iter<int> features -> sizes of hidden layers

		@staticmethod
		load(model_dump='best_model.pkl')
			args
				str model_dump -> path to load model

		dump(model_dump='best_model.pkl')
			args
				str model_dump -> path to dump model

		new_layer(n_out, tanh=False)
			adds new layer to model

			args
				int n_out
				bool tanh

		train(data, batch_size, max_epochs, distribution={'train':5./7,'valid':1./7,'test':1./7}, model_dump='best_model.pkl') - trains the model using the provided data
			args
				iter<iter<float>> data -> each row is the data point followed by the true value
				int batch_size
				int max_epochs
				dict<(train|valid|test),float> distribution -> partitions data
				str model_dump

		predict(data, d={})
			returns dictionary of the model's predicted values using input x, along with the actual values y

			args
				iter<iter<float>> data -> each row is the data point followed by the true value
				dict<float,str> d -> transformation of bin to label


	attributes
		numpy.random.RandomState rng
		iter<float> L_reg -> coefficients for L of each layer when calculating cost
		float learning_rate -> multiplicative factor of gradient when updating parameters
		iter<Layer> layers
		bool built -> True when the model has been built
		int n_in
		int n_out
		iter<int> features


class Regression(MLP)
	methods
		@classmethod
		build(n_in, features=[], bounds=[-numpy.inf,numpy.inf], **kwargs)
			args
				int n_in
				iter<int> features
				iter<float> bounds -> clamp for output


class Classification(MLP)


class Layer(object)
	methods
		__init__(n_in,n_out,rng=None)
			args
				int n_in
				int n_out
				numpy.random.RandomState rng

		L(W,b)
			args
				TheanoMatrix<float> W
				TheanoMatrix<float> b

		output(output,W,b)
			args
				TheanoMatrix<float> output
				TheanoMatrix<float> W
				TheanoVector<float> b

	attributes
		NumpyMatrix<float> W_values
		NumpyVector<float> b_values
		int n_in
		int n_out


class TanhLayer(Layer)


def load_data(dataset,regression,header=True,true_value_is_last=True,upsampling=False)
	returns dictionary{'data': numpy matrix<float> data, d: dictionary<int,string> transformation of bin to label, 'n_out': number of output bins, or 1 for regression}

	args
		str dataset
		bool regression
		bool header -> whether or not to ignore first line
		bool output_value_is_first -> true if output is the first value, false if the output is the last value in each row
		bool upsampling -> if true and the model is classification, upsamples data to have an equal number of data points for each bin


def easy_deep_learning(dataset,regression=False,depth=3,max_epochs=10,batch_size=20,predict_count=100,**kwargs)
	returns dictionary {loaded_data, model, results}

	args
		str dataset
		bool regression
		int depth
		int max_epochs
		int batch_size
		int predict_count
```

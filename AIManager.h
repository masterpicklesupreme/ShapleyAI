#pragma once
#include <vector>
#include <time.h>
#include <algorithm>
#include <random>

using namespace std;

// they have to be like this with the name then function because earlier i got an error that the internet told me that meant i was trying to redefine a reserved name, it was tanh i think
enum ActivationFunction { relu_function, leakyrelu_function, sigmoid_function, tanh_function, no_function };

struct Layer
{
public:
	ActivationFunction activationFunction;
	int layerID;
	int numOfNeurons;

	vector<vector<double>> weights;
	vector<double> biases;
	vector<double> activations;
};

class Network
{
	default_random_engine re{};
	void PopulateLayerWithValues(int lowWeightRange, int highWeightRange, int lowBiasRange, int highBiasRange, Layer& layer, Layer& lastLayer)
	{
		//srand(time(NULL));
		uniform_real_distribution<double> weightRange{ (double)lowWeightRange, (double)highWeightRange };
		uniform_real_distribution<double> biasRange{ (double)lowBiasRange, (double)highBiasRange };

		vector<double> temp;
		//int weightRange = highWeightRange - lowWeightRange;
		//int biasRange = lowBiasRange - highBiasRange;
		for (int y = 0; y < lastLayer.numOfNeurons; y++)
		{
			for (int x = 0; x < layer.numOfNeurons; x++)
			{
				//temp.push_back(rand() % (weightRange + 1) + lowWeightRange);
				//temp.push_back(lowWeightRange + (rand() / (RAND_MAX / (weightRange))));
				temp.push_back(weightRange(re));
			}
			layer.weights.push_back(temp);
			temp.clear();
		}

		for (int y = 0; y < layer.numOfNeurons; y++)
		{
			//layer.biases.push_back(rand() % (biasRange + 1) + lowBiasRange);
			//layer.biases.push_back(lowBiasRange + (rand() / (RAND_MAX / (biasRange))));
			layer.biases.push_back(biasRange(re));
		}
	}
	
public:
	vector<Layer> layers;
	int networkSize;
	int networkSizeMinus1;
	// default range is -100 to 100 for weights and -50 to 50 for biases.
	Network(vector<int>& listOfLayerLengths, ActivationFunction hiddenLayerActivationFunc, ActivationFunction outputLayerActivationFunc)
	{
		re.seed(time(NULL));
		vector<double> temp;
		int weightRange = 200;
		int biasRange = 100;
		networkSize = listOfLayerLengths.size();
		networkSizeMinus1 = networkSize - 1;
		
		for (int i = 0; i < networkSize; i++)
		{
			if (i == 0)
			{
				layers.push_back(Layer());
				layers[i].layerID = i;
				layers[i].numOfNeurons = listOfLayerLengths[i];
				continue;
			}
			else if (i == networkSizeMinus1)
			{
				layers.push_back(Layer());
				layers[i].layerID = i;
				layers[i].numOfNeurons = listOfLayerLengths[i];
				layers[i].activationFunction = outputLayerActivationFunc;
				PopulateLayerWithValues(-100, 100, -50, 50, layers[i], layers[i - 1]);

				//for (int y = 0; y < layers[i - 1].numOfNeurons; y++)
				//{
				//	for (int x = 0; x < layers[i].numOfNeurons; x++)
				//	{
				//		//temp.push_back(rand() % (weightRange + 1) + lowWeightRange);
				//		temp.push_back(-100 + (rand() / (RAND_MAX / (weightRange))));
				//	}
				//	layers[i].weights.push_back(temp);
				//	temp.clear();
				//}

				//for (int y = 0; y < layers[i].numOfNeurons; y++)
				//{
				//	//layer.biases.push_back(rand() % (biasRange + 1) + lowBiasRange);
				//	layers[i].biases.push_back(-50 + (rand() / (RAND_MAX / (biasRange))));
				//}

				continue;
			}
			else
			{
				layers.push_back(Layer());
				layers[i].layerID = i;
				layers[i].numOfNeurons = listOfLayerLengths[i];
				layers[i].activationFunction = hiddenLayerActivationFunc;
				PopulateLayerWithValues(-100, 100, -50, 50, layers[i], layers[i - 1]);

				//for (int y = 0; y < layers[i - 1].numOfNeurons; y++)
				//{
				//	for (int x = 0; x < layers[i].numOfNeurons; x++)
				//	{
				//		//temp.push_back(rand() % (weightRange + 1) + lowWeightRange);
				//		temp.push_back(-100 + (rand() / (RAND_MAX / (weightRange))));
				//	}
				//	layers[i].weights.push_back(temp);
				//	temp.clear();
				//}

				//for (int y = 0; y < layers[i].numOfNeurons; y++)
				//{
				//	//layer.biases.push_back(rand() % (biasRange + 1) + lowBiasRange);
				//	layers[i].biases.push_back(-50 + (rand() / (RAND_MAX / (biasRange))));
				//}
			}
		}
	}

	// note that ranges are inclusive (i think) hopefully it doesnt matter for your application
	Network(vector<int>& listOfLayerLengths, ActivationFunction hiddenLayerActivationFunc, ActivationFunction outputLayerActivationFunc, int lowWeightRange, int highWeightRange, int lowBiasRange, int highBiasRange)
	{
		re.seed(time(NULL));
		vector<double> temp;
		int weightRange = highWeightRange - lowWeightRange;
		int biasRange = lowBiasRange - highBiasRange;
		networkSize = listOfLayerLengths.size();
		networkSizeMinus1 = networkSize - 1;

		for (int i = 0; i < networkSize; i++)
		{
			if (i == 0)
			{
				layers.push_back(Layer());
				layers[i].layerID = i;
				layers[i].numOfNeurons = listOfLayerLengths[i];
				continue;
			}
			else if (i == networkSizeMinus1)
			{
				layers.push_back(Layer());
				layers[i].layerID = i;
				layers[i].numOfNeurons = listOfLayerLengths[i];
				layers[i].activationFunction = outputLayerActivationFunc;
				PopulateLayerWithValues(lowWeightRange, highWeightRange, lowBiasRange, highBiasRange, layers[i], layers[i - 1]);

				//for (int y = 0; y < layers[i - 1].numOfNeurons; y++)
				//{
				//	for (int x = 0; x < layers[i].numOfNeurons; x++)
				//	{
				//		//temp.push_back(rand() % (weightRange + 1) + lowWeightRange);
				//		temp.push_back(lowWeightRange + (rand() / (RAND_MAX / (weightRange))));
				//	}
				//	layers[i].weights.push_back(temp);
				//	temp.clear();
				//}

				//for (int y = 0; y < layers[i].numOfNeurons; y++)
				//{
				//	//layer.biases.push_back(rand() % (biasRange + 1) + lowBiasRange);
				//	layers[i].biases.push_back(lowBiasRange + (rand() / (RAND_MAX / (biasRange))));
				//}

				continue;
			}
			else
			{
				layers.push_back(Layer());
				layers[i].layerID = i;
				layers[i].numOfNeurons = listOfLayerLengths[i];
				layers[i].activationFunction = hiddenLayerActivationFunc;
				PopulateLayerWithValues(lowWeightRange, highWeightRange, lowBiasRange, highBiasRange, layers[i], layers[i - 1]);

				//for (int y = 0; y < layers[i - 1].numOfNeurons; y++)
				//{
				//	for (int x = 0; x < layers[i].numOfNeurons; x++)
				//	{
				//		//temp.push_back(rand() % (weightRange + 1) + lowWeightRange);
				//		temp.push_back(lowWeightRange + (rand() / (RAND_MAX / (weightRange))));
				//	}
				//	layers[i].weights.push_back(temp);
				//	temp.clear();
				//}

				//for (int y = 0; y < layers[i].numOfNeurons; y++)
				//{
				//	//layer.biases.push_back(rand() % (biasRange + 1) + lowBiasRange);
				//	layers[i].biases.push_back(lowBiasRange + (rand() / (RAND_MAX / (biasRange))));
				//}

			}
		}
	}

	// default activaton function is leaky relu for hidden and output layers.
	Network(vector<int>& listOfLayerLengths, int lowWeightRange, int highWeightRange, int lowBiasRange, int highBiasRange)
	{
		re.seed(time(NULL));
		vector<double> temp;
		int weightRange = highWeightRange - lowWeightRange;
		int biasRange = lowBiasRange - highBiasRange;
		networkSize = listOfLayerLengths.size();
		networkSizeMinus1 = networkSize - 1;

		for (int i = 0; i < networkSize; i++)
		{
			if (i == 0)
			{
				layers.push_back(Layer());
				layers[i].layerID = i;
				layers[i].numOfNeurons = listOfLayerLengths[i];
				continue;
			}
			else if (i == networkSizeMinus1)
			{
				layers.push_back(Layer());
				layers[i].layerID = i;
				layers[i].numOfNeurons = listOfLayerLengths[i];
				layers[i].activationFunction = leakyrelu_function;
				PopulateLayerWithValues(lowWeightRange, highWeightRange, lowBiasRange, highBiasRange, layers[i], layers[i - 1]);

				//for (int y = 0; y < layers[i - 1].numOfNeurons; y++)
				//{
				//	for (int x = 0; x < layers[i].numOfNeurons; x++)
				//	{
				//		//temp.push_back(rand() % (weightRange + 1) + lowWeightRange);
				//		temp.push_back(lowWeightRange + (rand() / (RAND_MAX / (weightRange))));
				//	}
				//	layers[i].weights.push_back(temp);
				//	temp.clear();
				//}

				//for (int y = 0; y < layers[i].numOfNeurons; y++)
				//{
				//	//layer.biases.push_back(rand() % (biasRange + 1) + lowBiasRange);
				//	layers[i].biases.push_back(lowBiasRange + (rand() / (RAND_MAX / (biasRange))));
				//}

				continue;
			}
			else
			{
				layers.push_back(Layer());
				layers[i].layerID = i;
				layers[i].numOfNeurons = listOfLayerLengths[i];
				layers[i].activationFunction = leakyrelu_function;
				PopulateLayerWithValues(lowWeightRange, highWeightRange, lowBiasRange, highBiasRange, layers[i], layers[i - 1]);

				//for (int y = 0; y < layers[i - 1].numOfNeurons; y++)
				//{
				//	for (int x = 0; x < layers[i].numOfNeurons; x++)
				//	{
				//		//temp.push_back(rand() % (weightRange + 1) + lowWeightRange);
				//		temp.push_back(lowWeightRange + (rand() / (RAND_MAX / (weightRange))));
				//	}
				//	layers[i].weights.push_back(temp);
				//	temp.clear();
				//}

				//for (int y = 0; y < layers[i].numOfNeurons; y++)
				//{
				//	//layer.biases.push_back(rand() % (biasRange + 1) + lowBiasRange);
				//	layers[i].biases.push_back(lowBiasRange + (rand() / (RAND_MAX / (biasRange))));
				//}

			}
		}
	}

	// all default stuff which is leaky relu for everything and -100 to 100 weight range -50 to 50 bias range.
	Network(vector<int>& listOfLayerLengths)
	{
		re.seed(time(NULL));
		vector<double> temp;
		int weightRange = 200;
		int biasRange = 100;
		networkSize = listOfLayerLengths.size();
		networkSizeMinus1 = networkSize - 1;

		for (int i = 0; i < networkSize; i++)
		{
			if (i == 0)
			{
				layers.push_back(Layer());
				layers[i].layerID = i;
				layers[i].numOfNeurons = listOfLayerLengths[i];
				continue;
			}
			else if (i == networkSizeMinus1)
			{
				layers.push_back(Layer());
				layers[i].layerID = i;
				layers[i].numOfNeurons = listOfLayerLengths[i];
				layers[i].activationFunction = leakyrelu_function;
				PopulateLayerWithValues(-100, 100, -50, 50, layers[i], layers[i - 1]);

				//for (int y = 0; y < layers[i - 1].numOfNeurons; y++)
				//{
				//	for (int x = 0; x < layers[i].numOfNeurons; x++)
				//	{
				//		//temp.push_back(rand() % (weightRange + 1) + lowWeightRange);
				//		temp.push_back(-100 + (rand() / (RAND_MAX / (weightRange))));
				//	}
				//	layers[i].weights.push_back(temp);
				//	temp.clear();
				//}

				//for (int y = 0; y < layers[i].numOfNeurons; y++)
				//{
				//	//layer.biases.push_back(rand() % (biasRange + 1) + lowBiasRange);
				//	layers[i].biases.push_back(-50 + (rand() / (RAND_MAX / (biasRange))));
				//}

				continue;
			}
			else
			{
				layers.push_back(Layer());
				layers[i].layerID = i;
				layers[i].numOfNeurons = listOfLayerLengths[i];
				layers[i].activationFunction = leakyrelu_function;
				PopulateLayerWithValues(-100, 100, -50, 50, layers[i], layers[i - 1]);

				//for (int y = 0; y < layers[i - 1].numOfNeurons; y++)
				//{
				//	for (int x = 0; x < layers[i].numOfNeurons; x++)
				//	{
				//		//temp.push_back(rand() % (weightRange + 1) + lowWeightRange);
				//		temp.push_back(-100 + (rand() / (RAND_MAX / (weightRange))));
				//	}
				//	layers[i].weights.push_back(temp);
				//	temp.clear();
				//}

				//for (int y = 0; y < layers[i].numOfNeurons; y++)
				//{
				//	//layer.biases.push_back(rand() % (biasRange + 1) + lowBiasRange);
				//	layers[i].biases.push_back(-50 + (rand() / (RAND_MAX / (biasRange))));
				//}
			}
		}
	}

	// d2 must be a variable/constant, not just a number, or this won't work. this is because d2 is using a reference of the passed double.
	double HigherDouble(double d1, double& d2)
	{
		if (d1 > d2)
		{
			return d1;
		}
		else
		{
			return d2;
		}
	}

	vector<double> RunNetwork(vector<double> inputs)
	{
		layers[0].activations = inputs;
		vector<double> calculatedLayer;
		//vector<double> outputLayer;
		for (int i = 1; i < layers.size(); i++)
		{
			// initialise calculated layer as the biases. this is the same as doing the shit with the weights then adding biases, as the method i use to do the shit with the weights
			//is by adding. however, this is more efficient as computer doesnt have to loop through stuff more
			calculatedLayer = layers[i].biases;

			// reason for this here being the last layer num of neurons is because that is the same number as there are vectors in the matrix; referencing some variable is a little more
			//efficient than getting size of the weights matrix
			for (int x = 0; x < layers[i - 1].numOfNeurons; x++)
			{
				// here is same but the lengths of all these vectors would be the number of how many neurons in current layers
				for (int y = 0; y < layers[i].numOfNeurons; y++)
				{
					// even now, while writing this, i barely understand how it works. i have a vague, thin thread of logic in my head that suggests it might.
					calculatedLayer[y] += layers[i - 1].activations[x] * layers[i].weights[x][y];
				}
			}

			// apply activation function
			switch (layers[i].activationFunction)
			{
			case relu_function:
				for (int x = 0; x < layers[i].numOfNeurons; x++)
				{
					calculatedLayer[x] = HigherDouble(0, calculatedLayer[x]);
				}
				break;
			case leakyrelu_function:
				for (int x = 0; x < layers[i].numOfNeurons; x++)
				{
					calculatedLayer[x] = HigherDouble(calculatedLayer[x] * 0.5, calculatedLayer[x]);
				}
				break;
			case sigmoid_function:
				for (int x = 0; x < layers[i].numOfNeurons; x++)
				{
					calculatedLayer[x] = 1 / (1 + exp(-calculatedLayer[x]));
				}
				break;
			case tanh_function:
				for (int x = 0; x < layers[i].numOfNeurons; x++)
				{
					calculatedLayer[x] = tanh(calculatedLayer[x]);
				}
				break;
			}
			layers[i].activations = calculatedLayer;
			layers[i - 1].activations.clear();
			calculatedLayer.clear();
		}
		//outputLayer = layers[networkSizeMinus1].activations;
		//layers[networkSizeMinus1].activations.clear();
		return layers[networkSizeMinus1].activations;
	}

	vector<double> RunNetworkForBackpropagation(vector<double> inputs, vector<vector<double>> &valuesBeforeApplyingActivationFunction)
	{
		layers[0].activations = inputs;
		vector<double> calculatedLayer;
		//vector<double> outputLayer;
		for (int i = 1; i < layers.size(); i++)
		{
			// initialise calculated layer as the biases. this is the same as doing the shit with the weights then adding biases, as the method i use to do the shit with the weights
			//is by adding. however, this is more efficient as computer doesnt have to loop through stuff more
			calculatedLayer = layers[i].biases;

			// reason for this here being the last layer num of neurons is because that is the same number as there are vectors in the matrix; referencing some variable is a little more
			//efficient than getting size of the weights matrix
			for (int x = 0; x < layers[i - 1].numOfNeurons; x++)
			{
				// here is same but the lengths of all these vectors would be the number of how many neurons in current layers
				for (int y = 0; y < layers[i].numOfNeurons; y++)
				{
					// even now, while writing this, i barely understand how it works. i have a vague, thin thread of logic in my head that suggests it might.
					calculatedLayer[y] += layers[i - 1].activations[x] * layers[i].weights[x][y];
				}
			}
			valuesBeforeApplyingActivationFunction.push_back(calculatedLayer);
			// apply activation function
			switch (layers[i].activationFunction)
			{
			case relu_function:
				for (int x = 0; x < layers[i].numOfNeurons; x++)
				{
					calculatedLayer[x] = HigherDouble(0, calculatedLayer[x]);
				}
				break;
			case leakyrelu_function:
				for (int x = 0; x < layers[i].numOfNeurons; x++)
				{
					calculatedLayer[x] = HigherDouble(calculatedLayer[x] * 0.5, calculatedLayer[x]);
				}
				break;
			case sigmoid_function:
				for (int x = 0; x < layers[i].numOfNeurons; x++)
				{
					calculatedLayer[x] = 1 / (1 + exp(-calculatedLayer[x]));
				}
				break;
			case tanh_function:
				for (int x = 0; x < layers[i].numOfNeurons; x++)
				{
					calculatedLayer[x] = tanh(calculatedLayer[x]);
				}
				break;
			}
			layers[i].activations = calculatedLayer;
			//layers[i - 1].activations.clear();
			calculatedLayer.clear();
		}
		//outputLayer = layers[networkSize - 1].activations;
		//layers[networkSize - 1].activations.clear();
		//return outputLayer;
		return layers[networkSizeMinus1].activations;
	}
};

static class NetworkHolder
{
	vector<Network> networks;
};
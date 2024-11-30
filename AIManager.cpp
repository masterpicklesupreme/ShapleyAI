#include <vector>
#include <time.h>
#include <algorithm>

using namespace std;

// they have to be like this with the name then function because earlier i got an error that the internet told me that meant i was trying to redefine a reserved name, it was tanh i
//think
//enum ActivationFunction { relu_function, leakyrelu_function, sigmoid_function, tanh_function };
//
//struct Layer
//{
//public:
//	ActivationFunction activationFunction;
//	int layerID;
//	int numOfNeurons;
//
//	vector<vector<double>> weights;
//	vector<double> biases;
//	vector<double> activations;
//};
//
//class Network
//{
//	void PopulateLayerWithValues(int lowWeightRange, int highWeightRange, int lowBiasRange, int highBiasRange, Layer& layer, Layer& lastLayer)
//	{
//		vector<double> temp;
//		int weightRange = highWeightRange - lowWeightRange;
//		int biasRange = lowBiasRange - highBiasRange;
//		for (int i = 0; i < lastLayer.numOfNeurons; i++)
//		{
//			for (int x = 0; x < layer.numOfNeurons; x++)
//			{
//				//temp.push_back(rand() % (weightRange + 1) + lowWeightRange);
//				temp.push_back(lowWeightRange + (rand() / (RAND_MAX / (weightRange))));
//			}
//			layer.weights.push_back(temp);
//		}
//
//		for (int i = 0; i < layer.numOfNeurons; i++)
//		{
//			//layer.biases.push_back(rand() % (biasRange + 1) + lowBiasRange);
//			layer.biases.push_back(lowBiasRange + (rand() / (RAND_MAX / (biasRange))));
//		}
//	}
//public:
//	vector<Layer> layers;
//
//	// default range is -100 to 100 for weights and -50 to 50 for biases.
//	Network(vector<int>& listOfLayerLengths, ActivationFunction& hiddenLayerActivationFunc, ActivationFunction& outputLayerActivationFunc)
//	{
//		srand(time(NULL));
//		for (int i = 0; i < listOfLayerLengths.size(); i++)
//		{
//			if (i == 0)
//			{
//				layers.push_back(Layer());
//				layers[i].layerID = i;
//				layers[i].numOfNeurons = listOfLayerLengths[i];
//				continue;
//			}
//			else if (i == listOfLayerLengths.size() - 1)
//			{
//				layers.push_back(Layer());
//				layers[i].layerID = i;
//				layers[i].numOfNeurons = listOfLayerLengths[i];
//				layers[i].activationFunction = outputLayerActivationFunc;
//				PopulateLayerWithValues(-100, 100, -50, 50, layers[i], layers[i - 1]);
//				continue;
//			}
//			else
//			{
//				layers.push_back(Layer());
//				layers[i].layerID = i;
//				layers[i].numOfNeurons = listOfLayerLengths[i];
//				layers[i].activationFunction = hiddenLayerActivationFunc;
//				PopulateLayerWithValues(-100, 100, -50, 50, layers[i], layers[i - 1]);
//			}
//		}
//	}
//
//	// note that ranges are inclusive (i think) hopefully it doesnt matter for your application
//	Network(vector<int>& listOfLayerLengths, ActivationFunction& hiddenLayerActivationFunc, ActivationFunction& outputLayerActivationFunc, int lowWeightRange, int highWeightRange, int lowBiasRange, int highBiasRange)
//	{
//		srand(time(NULL));
//		for (int i = 0; i < listOfLayerLengths.size(); i++)
//		{
//			if (i == 0)
//			{
//				layers.push_back(Layer());
//				layers[i].layerID = i;
//				layers[i].numOfNeurons = listOfLayerLengths[i];
//				continue;
//			}
//			else if (i == listOfLayerLengths.size() - 1)
//			{
//				layers.push_back(Layer());
//				layers[i].layerID = i;
//				layers[i].numOfNeurons = listOfLayerLengths[i];
//				layers[i].activationFunction = outputLayerActivationFunc;
//				PopulateLayerWithValues(lowWeightRange, highWeightRange, lowBiasRange, highBiasRange, layers[i], layers[i - 1]);
//				continue;
//			}
//			else
//			{
//				layers.push_back(Layer());
//				layers[i].layerID = i;
//				layers[i].numOfNeurons = listOfLayerLengths[i];
//				layers[i].activationFunction = hiddenLayerActivationFunc;
//				PopulateLayerWithValues(lowWeightRange, highWeightRange, lowBiasRange, highBiasRange, layers[i], layers[i - 1]);
//			}
//		}
//	}
//
//	// default activaton function is relu for hidden and output layers.
//	Network(vector<int>& listOfLayerLengths, int lowWeightRange, int highWeightRange, int lowBiasRange, int highBiasRange)
//	{
//		ActivationFunction hiddenLayerActivationFunc = leakyrelu_function;
//		ActivationFunction outputLayerActivationFunc = leakyrelu_function;
//		srand(time(NULL));
//		for (int i = 0; i < listOfLayerLengths.size(); i++)
//		{
//			if (i == 0)
//			{
//				layers.push_back(Layer());
//				layers[i].layerID = i;
//				layers[i].numOfNeurons = listOfLayerLengths[i];
//				continue;
//			}
//			else if (i == listOfLayerLengths.size() - 1)
//			{
//				layers.push_back(Layer());
//				layers[i].layerID = i;
//				layers[i].numOfNeurons = listOfLayerLengths[i];
//				layers[i].activationFunction = outputLayerActivationFunc;
//				PopulateLayerWithValues(lowWeightRange, highWeightRange, lowBiasRange, highBiasRange, layers[i], layers[i - 1]);
//				continue;
//			}
//			else
//			{
//				layers.push_back(Layer());
//				layers[i].layerID = i;
//				layers[i].numOfNeurons = listOfLayerLengths[i];
//				layers[i].activationFunction = hiddenLayerActivationFunc;
//				PopulateLayerWithValues(lowWeightRange, highWeightRange, lowBiasRange, highBiasRange, layers[i], layers[i - 1]);
//			}
//		}
//	}
//
//	// d2 must be a variable/constant, not just a number, or this won't work.
//	double HigherDouble(double d1, double& d2)
//	{
//		if (d1 > d2)
//		{
//			return d1;
//		}
//		else
//		{
//			return d2;
//		}
//	}
//
//	vector<double> RunNetwork(vector<double> inputs)
//	{
//		layers[0].activations = inputs;
//		vector<double> calculatedLayer;
//		for (int i = 1; i < layers.size(); i++)
//		{
//			// initialise calculated layer as the biases. this is the same as doing the shit with the weights then adding biases, as the method i use to do the shit with the weights
//			//is by adding. however, this is more efficient as computer doesnt have to loop through stuff more
//			calculatedLayer = layers[i].biases;
//
//			// reason for this here being the last layer num of neurons is because that is the same number as there are vectors in the matrix; referencing some variable is a little more
//			//efficient than getting size of the weights matrix
//			for (int x = 0; x < layers[i-1].numOfNeurons; x++)
//			{
//				// here is same but the lengths of all these vectors would be the number of how many neurons in current layers
//				for (int y = 0; y < layers[i].numOfNeurons; y++)
//				{
//					// even now, while writing this, i barely understand how it works. i have a vague, thin thread of logic in my head that suggests it might.
//					calculatedLayer[y] += layers[i - 1].activations[x] * layers[i].weights[x][y];
//				}
//			}
//			
//			// apply activation function
//			switch (layers[i].activationFunction)
//			{
//			case relu_function:
//				for (int x = 0; x < layers[i].numOfNeurons; x++)
//				{
//					calculatedLayer[x] = HigherDouble(0, calculatedLayer[x]);
//				}
//				break;
//			case leakyrelu_function:
//				for (int x = 0; x < layers[i].numOfNeurons; x++)
//				{
//					calculatedLayer[x] = HigherDouble(calculatedLayer[x] * 0.01, calculatedLayer[x]);
//				}
//				break;
//			case sigmoid_function:
//				for (int x = 0; x < layers[i].numOfNeurons; x++)
//				{
//					calculatedLayer[x] = 1/(1+exp(-calculatedLayer[x]));
//				}
//				break;
//			case tanh_function:
//				for (int x = 0; x < layers[i].numOfNeurons; x++)
//				{
//					calculatedLayer[x] = tanh(calculatedLayer[x]);
//				}
//				break;
//			}
//			calculatedLayer.clear();
//		}
//	}
//};
//
//static class NetworkHolder
//{
//	vector<Network> networks;
//};
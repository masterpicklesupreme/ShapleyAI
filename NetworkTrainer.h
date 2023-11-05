#pragma once
#include "AIManager.h"
#include <vector>

static class BackpropagationTrainer
{
public:
	void TrainBackpropagation(Network& network, vector<vector<double>>& inputTrainingData, vector<vector<double>>& outputTrainingData)
	{
		double learnRate = 0.001;

		vector<double> networkOutput;
		vector<double> lastLayerActivationDerivatives;
		vector<double> currentLayerActivationDerivatives;
		vector<double> activationBeforeFunctionDerivatives;

		vector<vector<vector<vector<double>>>> allNetworkWeightDerivatives;
		vector<vector<vector<double>>> networkWeightDerivatives;
		vector<vector<double>> layerWeightDerivatives;
		vector<double> tempWeightDerivatives;

		vector<vector<vector<double>>> allNetworkBiasDerivatives;
		vector<vector<double>> networkBiasDerivatives;
		vector<double> layerBiasDerivatives;

		// dont mind me!!! it would take ages and be really annoying to explain this, and it doesnt really matter, promise! (its for efficiency)
		int networkSizeMinus2 = network.networkSizeMinus1 - 1;

		// same with me!
		int layerNumMinus1;

		// this is passed into the runnetwork method as a reference, the runnetwork method populates it.
		vector<vector<double>> valuesBeforeApplyingActivationFunction;

		for (int currentTrainingExample = 0; currentTrainingExample < inputTrainingData.size(); currentTrainingExample++)
		{
			// run the network on the data to get the output
			valuesBeforeApplyingActivationFunction.clear();
			networkOutput = network.RunNetworkForBackpropagation(inputTrainingData[currentTrainingExample], valuesBeforeApplyingActivationFunction);
			
			for (int neuron = 0; neuron < network.layers[network.networkSizeMinus1].numOfNeurons; neuron++)
			{
				// get the derivatives for cost in terms of last layer neurons
				lastLayerActivationDerivatives.push_back(2 * (network.layers[network.networkSizeMinus1].activations[neuron] - outputTrainingData[currentTrainingExample][neuron]));

				// get derivatives for last layer neurons in terms of the values for the activations before applying activation function
				switch (network.layers[network.networkSizeMinus1].activationFunction)
				{
				case relu_function:
					if (valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron] < 0.0)
					{
						activationBeforeFunctionDerivatives.push_back(0.0);
					}
					else
					{
						activationBeforeFunctionDerivatives.push_back(1.0);
					}
					break;
				case leakyrelu_function:
					if (valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron] < 0.0)
					{
						activationBeforeFunctionDerivatives.push_back(0.5);
					}
					else
					{
						activationBeforeFunctionDerivatives.push_back(1.0);
					}
					break;
				case sigmoid_function:
					activationBeforeFunctionDerivatives.push_back(
						exp(valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron]) /
						pow( exp(valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron]) + 1.0, 2.0)
					);
					break;
				case tanh_function:
					activationBeforeFunctionDerivatives.push_back(1.0-pow(tanh(valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron]), 2.0));
					break;
				}
			}
			
			for (int lastLayerNeuron = 0; lastLayerNeuron < network.layers[networkSizeMinus2].numOfNeurons; lastLayerNeuron++)
			{
				// see: inside the next for loop, beneath where the shit with the weights is done for an explanation of this line
				currentLayerActivationDerivatives.push_back(0.0);
				for (int currentLayerNeuron = 0; currentLayerNeuron < network.layers[network.networkSizeMinus1].numOfNeurons; currentLayerNeuron++)
				{
					// weights are not immediately being updated, neither is anything else. this is because all the derivatives of everything have to be averaged across multiple training examples.
					tempWeightDerivatives.push_back(network.layers[networkSizeMinus2].activations[lastLayerNeuron] * activationBeforeFunctionDerivatives[currentLayerNeuron] * lastLayerActivationDerivatives[currentLayerNeuron]);
					
					// this is done because the next layer's activations affect the cost function through multiple avenues. so, the derivative must be summed up among every avenue.
					currentLayerActivationDerivatives[lastLayerNeuron] += (network.layers[network.networkSizeMinus1].weights[lastLayerNeuron][currentLayerNeuron] * activationBeforeFunctionDerivatives[currentLayerNeuron] * lastLayerActivationDerivatives[currentLayerNeuron]);
					//network.layers[network.networkSizeMinus1].weights[lastLayerNeuron][currentLayerNeuron] -= network.layers[network.networkSizeMinus1 - 1].activations[lastLayerNeuron] * activationBeforeFunctionDerivatives[currentLayerNeuron] * activationDerivatives[currentLayerNeuron];
				}
				layerWeightDerivatives.push_back(tempWeightDerivatives);
				tempWeightDerivatives.clear();
			}

			// loop through biases separately from weights and activations because weights and activations go through current layer neurons for every previous layer neuron. not needed for biases
			for (int neuron = 0; neuron < network.layers[network.networkSizeMinus1].numOfNeurons; neuron++)
			{
				//network.layers[network.networkSizeMinus1].biases[neuron] -= activationBeforeFunctionDerivatives[neuron] * activationDerivatives[neuron];
				layerBiasDerivatives.push_back(activationBeforeFunctionDerivatives[neuron] * lastLayerActivationDerivatives[neuron]);
			}

			// put current layer weights and biases into the whole network weights and biases so that this layer is saved. then the next layer can be calculated
			networkWeightDerivatives.insert(networkWeightDerivatives.begin(), layerWeightDerivatives);
			networkBiasDerivatives.insert(networkBiasDerivatives.begin(), layerBiasDerivatives);

			layerWeightDerivatives.clear();
			layerBiasDerivatives.clear();

			lastLayerActivationDerivatives = currentLayerActivationDerivatives;
			currentLayerActivationDerivatives.clear();

			activationBeforeFunctionDerivatives.clear();
			
			// loop through layers going BACKwards hahaha get it cause its backpropagation and we are going backwards hahahah. anyway the last layer is different to the others, thats why its not in the
			// loop.
			// stops at 1 because the first layer doesnt have any weights or biases; its the input layer.
			for (int layerNum = network.networkSizeMinus1; layerNum >= 1; layerNum--)
			{
				layerNumMinus1 = layerNum - 1;
				for (int neuron = 0; neuron < network.layers[layerNum].numOfNeurons; neuron++)
				{
					switch (network.layers[layerNum].activationFunction)
					{
					case relu_function:
						if (valuesBeforeApplyingActivationFunction[layerNumMinus1][neuron] < 0.0)
						{
							activationBeforeFunctionDerivatives.push_back(0.0);
						}
						else
						{
							activationBeforeFunctionDerivatives.push_back(1.0);
						}
						break;
					case leakyrelu_function:
						if (valuesBeforeApplyingActivationFunction[layerNumMinus1][neuron] < 0.0)
						{
							activationBeforeFunctionDerivatives.push_back(0.5);
						}
						else
						{
							activationBeforeFunctionDerivatives.push_back(1.0);
						}
						break;
					case sigmoid_function:
						activationBeforeFunctionDerivatives.push_back(
							exp(valuesBeforeApplyingActivationFunction[layerNumMinus1][neuron]) /
							pow(exp(valuesBeforeApplyingActivationFunction[layerNumMinus1][neuron]) + 1.0, 2.0)
						);
						break;
					case tanh_function:
						activationBeforeFunctionDerivatives.push_back(1.0 - pow(tanh(valuesBeforeApplyingActivationFunction[layerNumMinus1][neuron]), 2.0));
						break;
					}
				}

				for (int lastLayerNeuron = 0; lastLayerNeuron < network.layers[layerNumMinus1].numOfNeurons; lastLayerNeuron++)
				{
					currentLayerActivationDerivatives.push_back(0.0);
					for (int currentLayerNeuron = 0; currentLayerNeuron < network.layers[layerNum].numOfNeurons; currentLayerNeuron++)
					{
						tempWeightDerivatives.push_back(network.layers[layerNumMinus1].activations[lastLayerNeuron] * activationBeforeFunctionDerivatives[currentLayerNeuron] * lastLayerActivationDerivatives[currentLayerNeuron]);

						currentLayerActivationDerivatives[lastLayerNeuron] += (network.layers[layerNum].weights[lastLayerNeuron][currentLayerNeuron] * activationBeforeFunctionDerivatives[currentLayerNeuron] * lastLayerActivationDerivatives[currentLayerNeuron]);
					}
					layerWeightDerivatives.push_back(tempWeightDerivatives);
					tempWeightDerivatives.clear();
				}

				for (int neuron = 0; neuron < network.layers[network.networkSizeMinus1].numOfNeurons; neuron++)
				{
					layerBiasDerivatives.push_back(activationBeforeFunctionDerivatives[neuron] * lastLayerActivationDerivatives[neuron]);
				}

				networkWeightDerivatives.insert(networkWeightDerivatives.begin(), layerWeightDerivatives);
				networkBiasDerivatives.insert(networkBiasDerivatives.begin(), layerBiasDerivatives);

				layerWeightDerivatives.clear();
				layerBiasDerivatives.clear();

				lastLayerActivationDerivatives = currentLayerActivationDerivatives;
				currentLayerActivationDerivatives.clear();

				activationBeforeFunctionDerivatives.clear();
			}

			allNetworkWeightDerivatives.push_back(networkWeightDerivatives);
			allNetworkBiasDerivatives.push_back(networkBiasDerivatives);

			networkBiasDerivatives.clear();
			networkWeightDerivatives.clear();

			// clear activations of network
			//for (int layer = 0; layer < network.networkSize; layer++)
			//{
			//	network.layers[layer].activations.clear();
			//}
		}

		// average all weight derivatives
		vector<vector<double>> temp2DDerivatives;
		vector<double> temp1DDerivatives;
		double tempDerivative = 0;
		int size1;
		int size2;
		int size3 = allNetworkWeightDerivatives.size();

		// add em up
		for (int x = 0; x < network.networkSize; x++)
		{
			size1 = allNetworkWeightDerivatives[0][x].size();
			for (int y = 0; y < size1; y++)
			{
				size2 = allNetworkWeightDerivatives[0][x][y].size();
				for (int z = 0; z < size2; z++)
				{
					for (int a = 0; a < size3; a++)
					{
						tempDerivative += allNetworkWeightDerivatives[a][x][y][z];
					}
					temp1DDerivatives.push_back(tempDerivative);
					tempDerivative = 0;
				}
				temp2DDerivatives.push_back(temp1DDerivatives);
				temp1DDerivatives.clear();
			}
			networkWeightDerivatives.push_back(temp2DDerivatives);
			temp2DDerivatives.clear();
		}

		// divide em
		//cout << networkWeightDerivatives[0].size();
		for (int x = 1; x < network.networkSize; x++)
		{
			size1 = networkWeightDerivatives[x].size();
			for (int y = 0; y < size1; y++)
			{
				size2 = networkWeightDerivatives[x][y].size();
				for (int z = 0; z < size2; z++)
				{
					networkWeightDerivatives[x][y][z] = networkWeightDerivatives[x][y][z] / size3;
					if (isnan(networkWeightDerivatives[x][y][z]) || !(networkWeightDerivatives[x][y][z] == networkWeightDerivatives[x][y][z]))
					{
						cout << "weight after divide ";
					}
					
					// apply the weight change to the network
					network.layers[x].weights[y][z] += (networkWeightDerivatives[x][y][z] * learnRate);
				}
			}
		}

		// average all bias derivatives
		size3 = allNetworkBiasDerivatives.size();
		size1 = allNetworkBiasDerivatives[0].size();

		// add em up
		for (int x = 0; x < size1; x++)
		{
			size2 = allNetworkBiasDerivatives[0][x].size();
			for (int y = 0; y < size2; y++)
			{
				for (int z = 0; z < size3; z++)
				{
					tempDerivative += allNetworkBiasDerivatives[z][x][y];
				}
				temp1DDerivatives.push_back(tempDerivative);
				tempDerivative = 0;
			}
			networkBiasDerivatives.push_back(temp1DDerivatives);
			temp1DDerivatives.clear();
		}

		// divide em
		//cout << size3;
		for (int x = 1; x < network.networkSize; x++)
		{
			size2 = networkBiasDerivatives[x].size();
			for (int y = 0; y < size2; y++)
			{
				//networkBiasDerivatives[x][y] = networkBiasDerivatives[x][y] / size3;
				if (isnan(networkBiasDerivatives[x][y]) || !(networkBiasDerivatives[x][y] == networkBiasDerivatives[x][y]))
				{
					cout << "before divide ";
				}
				networkBiasDerivatives[x][y] /= size3;
				if (isnan(networkBiasDerivatives[x][y]) || !(networkBiasDerivatives[x][y] == networkBiasDerivatives[x][y]))
				{
					cout << "after divide ";
				}

				// apply bias change to the network
				network.layers[x].biases[y] += (networkBiasDerivatives[x][y] * learnRate);
			}
		}
	}
};

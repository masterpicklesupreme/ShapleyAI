#pragma once
#include "AIManager.h"
#include <vector>

static class BackpropagationTrainer
{
public:
	void TrainBackpropagation(Network& network, vector<vector<double>>& inputTrainingData, vector<vector<double>>& outputTrainingData)
	{
		// change this to instead be given as an argument...
		double learnRate = 0.01;

		//vector<double> networkOutput; 
		//vector<double> lastLayerActivationDerivatives;
		//vector<double> currentLayerActivationDerivatives;
		//vector<double> activationBeforeFunctionDerivatives;

		//vector<vector<vector<vector<double>>>> allNetworkWeightDerivatives;
		//vector<vector<vector<double>>> networkWeightDerivatives;
		//vector<vector<double>> layerWeightDerivatives;
		//vector<double> tempWeightDerivatives;

		//vector<vector<vector<double>>> allNetworkBiasDerivatives;
		//vector<vector<double>> networkBiasDerivatives;
		//vector<double> layerBiasDerivatives;

		//// dont mind me!!! it would take ages and be really annoying to explain this, and it doesnt really matter, promise! (its for efficiency)
		//int networkSizeMinus2 = network.networkSizeMinus1 - 1;

		//// same with me!
		//int layerNumMinus1;

		//// this is passed into the runnetwork method as a reference, the runnetwork method populates it.
		//vector<vector<double>> valuesBeforeApplyingActivationFunction;

		// this is very bad, change it to be an argument given in the function instead. only written like this here and now to get something bloody working already.
		int numberOfSteps = 50;
		for (int step = 0; step < numberOfSteps; step++)
		{
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

				//std::cout << "size of values before applying activation function: " << valuesBeforeApplyingActivationFunction.size() << endl << endl;
				//std::cout << "output layer size: " << network.layers[network.networkSizeMinus1].numOfNeurons << endl;

				for (int neuron = 0; neuron < network.layers[network.networkSizeMinus1].numOfNeurons; neuron++)
				{
					// get the derivatives for cost in terms of last layer neurons
					lastLayerActivationDerivatives.push_back(2 * (network.layers[network.networkSizeMinus1].activations[neuron] - outputTrainingData[currentTrainingExample][neuron]));

					//std::cout << 2 * (network.layers[network.networkSizeMinus1].activations[neuron] - outputTrainingData[currentTrainingExample][neuron]) << "   ";

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
							//std::cout << "before activ func: " << valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron] << " after activ func: " << network.layers[network.networkSizeMinus1].activations[neuron] << endl;
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
								pow(exp(valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron]) + 1.0, 2.0)
							);
							break;
						case tanh_function:
							activationBeforeFunctionDerivatives.push_back(1.0 - pow(tanh(valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron]), 2.0));
							break;
					}
				}

				// this for loop is just debug, you can remove it if you want
				/*std::cout << endl << endl;
				std::cout << lastLayerActivationDerivatives.size()<< endl;
				for (int der = 0; der < lastLayerActivationDerivatives.size(); der++)
				{
					std::cout << lastLayerActivationDerivatives[der] << "   ";
				}
				std::cout << endl << endl << endl;*/

				// this for loop gets the derivatives of the weights in terms of the cost and the next layer's activations in terms of the cost
				for (int lastLayerNeuron = 0; lastLayerNeuron < network.layers[networkSizeMinus2].numOfNeurons; lastLayerNeuron++)
				{
					// there used to be a shitty useless explanation of this, instead i have graced you with a better explanation now: need to start off each element here with 0 as it is additively...
					//added to inside the for loop. see just below the tempweightderivatives stuff inside this loop.
					currentLayerActivationDerivatives.push_back(0.0);
					for (int currentLayerNeuron = 0; currentLayerNeuron < network.layers[network.networkSizeMinus1].numOfNeurons; currentLayerNeuron++)
					{
						//std::cout << network.layers[networkSizeMinus2].activations[lastLayerNeuron] * activationBeforeFunctionDerivatives[currentLayerNeuron] * lastLayerActivationDerivatives[currentLayerNeuron] << "    ";
						//std::cout << lastLayerActivationDerivatives[currentLayerNeuron] << "    ";
						// now get the derivatives of the weight in terms of the cost
						// weights are not immediately being updated, neither is anything else. this is because all the derivatives of everything have to be averaged across multiple training examples.
						tempWeightDerivatives.push_back(network.layers[networkSizeMinus2].activations[lastLayerNeuron]
							* activationBeforeFunctionDerivatives[currentLayerNeuron]
							* lastLayerActivationDerivatives[currentLayerNeuron]);

						// now get the derivatives of the next layers activations in terms of the cost
						// this is done because the next layer's activations affect the cost function through multiple avenues. so, the derivative must be summed up among every avenue.
						currentLayerActivationDerivatives[lastLayerNeuron] += (network.layers[network.networkSizeMinus1].weights[lastLayerNeuron][currentLayerNeuron]
							* activationBeforeFunctionDerivatives[currentLayerNeuron]
							* lastLayerActivationDerivatives[currentLayerNeuron]);
						//network.layers[network.networkSizeMinus1].weights[lastLayerNeuron][currentLayerNeuron] -= network.layers[network.networkSizeMinus1 - 1].activations[lastLayerNeuron] * activationBeforeFunctionDerivatives[currentLayerNeuron] * activationDerivatives[currentLayerNeuron];
					}
					layerWeightDerivatives.push_back(tempWeightDerivatives);
					tempWeightDerivatives.clear();
				}

				// get the derivatives of the biases in terms of the cost
				// loop through biases separately from weights and activations because weights and activations go through current layer neurons for every previous layer neuron. not needed for biases
				for (int neuron = 0; neuron < network.layers[network.networkSizeMinus1].numOfNeurons; neuron++)
				{
					//network.layers[network.networkSizeMinus1].biases[neuron] -= activationBeforeFunctionDerivatives[neuron] * activationDerivatives[neuron];
					layerBiasDerivatives.push_back(activationBeforeFunctionDerivatives[neuron] * lastLayerActivationDerivatives[neuron]);
				}

				// put current layer weights and biases into the whole network weights and biases so that this layer is saved. then the next layer can be calculated
				networkWeightDerivatives.insert(networkWeightDerivatives.begin(), layerWeightDerivatives);
				networkBiasDerivatives.insert(networkBiasDerivatives.begin(), layerBiasDerivatives);

				// this is for debug pls delete
				/*for (int i = 0; i < networkWeightDerivatives.size(); i++)
				{
					std::cout << "i: " << i << endl;
					for (int x = 0; x < networkWeightDerivatives[i].size(); x++)
					{
						std::cout << "x: " << x << endl;
						for (int y = 0; y < networkWeightDerivatives[i][x].size(); y++)
						{
							std::cout << "y: " << y << " ";
						}
						std::cout << endl;
					}
					std::cout << endl << endl;
				}*/

				layerWeightDerivatives.clear();
				layerBiasDerivatives.clear();

				lastLayerActivationDerivatives = currentLayerActivationDerivatives;
				currentLayerActivationDerivatives.clear();

				activationBeforeFunctionDerivatives.clear();

				// loop through layers going BACKwards hahaha get it cause its backpropagation and we are going backwards hahahah. anyway the last layer is different to the others, thats why its not in the
				// loop.
				// stops at 1 because the first layer doesnt have any weights or biases; its the input layer.
				for (int layerNum = networkSizeMinus2; layerNum >= 1; layerNum--)
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
							tempWeightDerivatives.push_back(network.layers[layerNumMinus1].activations[lastLayerNeuron]
								* activationBeforeFunctionDerivatives[currentLayerNeuron]
								* lastLayerActivationDerivatives[currentLayerNeuron]);

							currentLayerActivationDerivatives[lastLayerNeuron] += (network.layers[layerNum].weights[lastLayerNeuron][currentLayerNeuron]
								* activationBeforeFunctionDerivatives[currentLayerNeuron]
								* lastLayerActivationDerivatives[currentLayerNeuron]);
						}
						layerWeightDerivatives.push_back(tempWeightDerivatives);
						tempWeightDerivatives.clear();
					}

					for (int neuron = 0; neuron < network.layers[layerNum].numOfNeurons; neuron++)
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

				lastLayerActivationDerivatives.clear();

				allNetworkWeightDerivatives.push_back(networkWeightDerivatives);
				allNetworkBiasDerivatives.push_back(networkBiasDerivatives);

				networkBiasDerivatives.clear();
				networkWeightDerivatives.clear();

				// clear activations of network
				/*for (int layer = 0; layer < network.networkSize; layer++)
				{
					network.layers[layer].activations.clear();
				}*/
			}

			// average all weight derivatives
			vector<vector<double>> temp2DDerivatives;
			vector<double> temp1DDerivatives;
			double tempDerivative = 0;
			int size1;
			int size2;
			int size3 = allNetworkWeightDerivatives.size();
			std::cout << "num of layers of allNetworkWeightDerivatives's networks" << allNetworkWeightDerivatives[0].size() << endl;

			// add em up (x is layers, y is neurons behind current layer, z is neurons in current layer)
			for (int x = 0; x < network.networkSize - 1; x++)
			{
				size1 = allNetworkWeightDerivatives[0][x].size();
				//std::cout << "x: " << x << endl;
				for (int y = 0; y < size1; y++)
				{
					size2 = allNetworkWeightDerivatives[0][x][y].size();
					//std::cout << "No. of neurons: " << size2 << endl;
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
			//std::cout << "time for network weight derivatives times the learn rate" << endl;
			for (int x = 0; x < network.networkSize - 1; x++)
			{
				size1 = networkWeightDerivatives[x].size();
				for (int y = 0; y < size1; y++)
				{
					size2 = networkWeightDerivatives[x][y].size();

					for (int z = 0; z < size2; z++)
					{
						networkWeightDerivatives[x][y][z] = networkWeightDerivatives[x][y][z] / size3;
						//std::cout << networkWeightDerivatives[x][y][z] * learnRate << "   ";

						// apply the weight change to the network
						network.layers[x + 1].weights[y][z] -= (networkWeightDerivatives[x][y][z] * learnRate);
					}
				}
			}

			//std::cout << endl << endl << "biases now" << endl;

			// average all bias derivatives
			size3 = allNetworkBiasDerivatives.size();
			size1 = allNetworkBiasDerivatives[0].size();
			//std::cout << "size3: " << size3 << ", size1: " << size1 << endl << endl;
			// x could potentially be each layer, then y each neuron, then z each training example... no guarantee
			// add em up
			for (int x = 0; x < size1; x++)
			{

				size2 = allNetworkBiasDerivatives[0][x].size();
				//std::cout << "No. of neurons: " << size2 << endl;
				//std::cout << "x: " << x << endl;
				//std::cout << "size2: " << size2 << endl;
				for (int y = 0; y < size2; y++)
				{
					//std::cout << "y: " << y << endl;

					for (int z = 0; z < size3; z++)
					{
						tempDerivative += allNetworkBiasDerivatives[z][x][y];
						//std::cout << "z: " << z << "   ";
					}
					temp1DDerivatives.push_back(tempDerivative);
					tempDerivative = 0;
					//std::cout << endl;
				}
				networkBiasDerivatives.push_back(temp1DDerivatives);
				temp1DDerivatives.clear();
				//std::cout << endl << endl;
			}

			// divide em
			//cout << size3;
			for (int x = 0; x < network.networkSize - 1; x++)
			{
				size2 = networkBiasDerivatives[x].size();
				for (int y = 0; y < size2; y++)
				{
					//networkBiasDerivatives[x][y] = networkBiasDerivatives[x][y] / size3;
					networkBiasDerivatives[x][y] /= size3;
					//std::cout << networkBiasDerivatives[x][y] * learnRate << "   ";
					// apply bias change to the network
					network.layers[x + 1].biases[y] -= (networkBiasDerivatives[x][y] * learnRate);
				}
			}
		}

		//for (int currentTrainingExample = 0; currentTrainingExample < inputTrainingData.size(); currentTrainingExample++)
		//{
		//	// run the network on the data to get the output
		//	valuesBeforeApplyingActivationFunction.clear();
		//	networkOutput = network.RunNetworkForBackpropagation(inputTrainingData[currentTrainingExample], valuesBeforeApplyingActivationFunction);

		//	std::cout << "size of values before applying activation function: " << valuesBeforeApplyingActivationFunction.size() << endl << endl;
		//	//std::cout << "output layer size: " << network.layers[network.networkSizeMinus1].numOfNeurons << endl;

		//	for (int neuron = 0; neuron < network.layers[network.networkSizeMinus1].numOfNeurons; neuron++)
		//	{
		//		// get the derivatives for cost in terms of last layer neurons (there is some problem here)
		//		lastLayerActivationDerivatives.push_back(2 * (network.layers[network.networkSizeMinus1].activations[neuron] - outputTrainingData[currentTrainingExample][neuron]));

		//		//std::cout << 2 * (network.layers[network.networkSizeMinus1].activations[neuron] - outputTrainingData[currentTrainingExample][neuron]) << "   ";

		//		// get derivatives for last layer neurons in terms of the values for the activations before applying activation function
		//		switch (network.layers[network.networkSizeMinus1].activationFunction)
		//		{
		//		case relu_function:
		//			if (valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron] < 0.0)
		//			{
		//				activationBeforeFunctionDerivatives.push_back(0.0);
		//			}
		//			else
		//			{
		//				activationBeforeFunctionDerivatives.push_back(1.0);
		//			}
		//			break;
		//		case leakyrelu_function:
		//			//std::cout << "before activ func: " << valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron] << " after activ func: " << network.layers[network.networkSizeMinus1].activations[neuron] << endl;
		//			if (valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron] < 0.0)
		//			{
		//				activationBeforeFunctionDerivatives.push_back(0.5);
		//			}
		//			else
		//			{
		//				activationBeforeFunctionDerivatives.push_back(1.0);
		//			}
		//			break;
		//		case sigmoid_function:
		//			activationBeforeFunctionDerivatives.push_back(
		//				exp(valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron]) /
		//				pow( exp(valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron]) + 1.0, 2.0)
		//			);
		//			break;
		//		case tanh_function:
		//			activationBeforeFunctionDerivatives.push_back(1.0-pow(tanh(valuesBeforeApplyingActivationFunction[networkSizeMinus2][neuron]), 2.0));
		//			break;
		//		}
		//	}
		//	
		//	// this for loop is just debug, you can remove it if you want
		//	/*std::cout << endl << endl;
		//	std::cout << lastLayerActivationDerivatives.size()<< endl;
		//	for (int der = 0; der < lastLayerActivationDerivatives.size(); der++)
		//	{
		//		std::cout << lastLayerActivationDerivatives[der] << "   ";
		//	}
		//	std::cout << endl << endl << endl;*/

		//	for (int lastLayerNeuron = 0; lastLayerNeuron < network.layers[networkSizeMinus2].numOfNeurons; lastLayerNeuron++)
		//	{
		//		// there used to be a shitty useless explanation of this, instead i have graced you with a better explanation now: need to start off each element here with 0 as it is additively...
		//		//added to inside the for loop. see just below the tempweightderivatives stuff inside this loop.
		//		currentLayerActivationDerivatives.push_back(0.0);
		//		for (int currentLayerNeuron = 0; currentLayerNeuron < network.layers[network.networkSizeMinus1].numOfNeurons; currentLayerNeuron++)
		//		{
		//			//std::cout << network.layers[networkSizeMinus2].activations[lastLayerNeuron] * activationBeforeFunctionDerivatives[currentLayerNeuron] * lastLayerActivationDerivatives[currentLayerNeuron] << "    ";
		//			//std::cout << lastLayerActivationDerivatives[currentLayerNeuron] << "    ";
		//			// weights are not immediately being updated, neither is anything else. this is because all the derivatives of everything have to be averaged across multiple training examples.
		//			tempWeightDerivatives.push_back(network.layers[networkSizeMinus2].activations[lastLayerNeuron]
		//				* activationBeforeFunctionDerivatives[currentLayerNeuron]
		//				* lastLayerActivationDerivatives[currentLayerNeuron]);
		//			
		//			// this is done because the next layer's activations affect the cost function through multiple avenues. so, the derivative must be summed up among every avenue.
		//			currentLayerActivationDerivatives[lastLayerNeuron] += (network.layers[network.networkSizeMinus1].weights[lastLayerNeuron][currentLayerNeuron]
		//				* activationBeforeFunctionDerivatives[currentLayerNeuron]
		//				* lastLayerActivationDerivatives[currentLayerNeuron]);
		//			//network.layers[network.networkSizeMinus1].weights[lastLayerNeuron][currentLayerNeuron] -= network.layers[network.networkSizeMinus1 - 1].activations[lastLayerNeuron] * activationBeforeFunctionDerivatives[currentLayerNeuron] * activationDerivatives[currentLayerNeuron];
		//		}
		//		layerWeightDerivatives.push_back(tempWeightDerivatives);
		//		tempWeightDerivatives.clear();
		//	}

		//	// loop through biases separately from weights and activations because weights and activations go through current layer neurons for every previous layer neuron. not needed for biases
		//	for (int neuron = 0; neuron < network.layers[network.networkSizeMinus1].numOfNeurons; neuron++)
		//	{
		//		//network.layers[network.networkSizeMinus1].biases[neuron] -= activationBeforeFunctionDerivatives[neuron] * activationDerivatives[neuron];
		//		layerBiasDerivatives.push_back(activationBeforeFunctionDerivatives[neuron] * lastLayerActivationDerivatives[neuron]);
		//	}

		//	// put current layer weights and biases into the whole network weights and biases so that this layer is saved. then the next layer can be calculated
		//	networkWeightDerivatives.insert(networkWeightDerivatives.begin(), layerWeightDerivatives);
		//	networkBiasDerivatives.insert(networkBiasDerivatives.begin(), layerBiasDerivatives);

		//	// this is for debug pls delete
		//	/*for (int i = 0; i < networkWeightDerivatives.size(); i++)
		//	{
		//		std::cout << "i: " << i << endl;
		//		for (int x = 0; x < networkWeightDerivatives[i].size(); x++)
		//		{
		//			std::cout << "x: " << x << endl;
		//			for (int y = 0; y < networkWeightDerivatives[i][x].size(); y++)
		//			{
		//				std::cout << "y: " << y << " ";
		//			}
		//			std::cout << endl;
		//		}
		//		std::cout << endl << endl;
		//	}*/

		//	layerWeightDerivatives.clear();
		//	layerBiasDerivatives.clear();

		//	lastLayerActivationDerivatives = currentLayerActivationDerivatives;
		//	currentLayerActivationDerivatives.clear();

		//	activationBeforeFunctionDerivatives.clear();
		//	
		//	// loop through layers going BACKwards hahaha get it cause its backpropagation and we are going backwards hahahah. anyway the last layer is different to the others, thats why its not in the
		//	// loop.
		//	// stops at 1 because the first layer doesnt have any weights or biases; its the input layer.
		//	for (int layerNum = networkSizeMinus2; layerNum >= 1; layerNum--)
		//	{
		//		layerNumMinus1 = layerNum - 1;
		//		for (int neuron = 0; neuron < network.layers[layerNum].numOfNeurons; neuron++)
		//		{
		//			switch (network.layers[layerNum].activationFunction)
		//			{
		//			case relu_function:
		//				if (valuesBeforeApplyingActivationFunction[layerNumMinus1][neuron] < 0.0)
		//				{
		//					activationBeforeFunctionDerivatives.push_back(0.0);
		//				}
		//				else
		//				{
		//					activationBeforeFunctionDerivatives.push_back(1.0);
		//				}
		//				break;
		//			case leakyrelu_function:
		//				if (valuesBeforeApplyingActivationFunction[layerNumMinus1][neuron] < 0.0)
		//				{
		//					activationBeforeFunctionDerivatives.push_back(0.5);
		//				}
		//				else
		//				{
		//					activationBeforeFunctionDerivatives.push_back(1.0);
		//				}
		//				break;
		//			case sigmoid_function:
		//				activationBeforeFunctionDerivatives.push_back(
		//					exp(valuesBeforeApplyingActivationFunction[layerNumMinus1][neuron]) /
		//					pow(exp(valuesBeforeApplyingActivationFunction[layerNumMinus1][neuron]) + 1.0, 2.0)
		//				);
		//				break;
		//			case tanh_function:
		//				activationBeforeFunctionDerivatives.push_back(1.0 - pow(tanh(valuesBeforeApplyingActivationFunction[layerNumMinus1][neuron]), 2.0));
		//				break;
		//			}
		//		}

		//		for (int lastLayerNeuron = 0; lastLayerNeuron < network.layers[layerNumMinus1].numOfNeurons; lastLayerNeuron++)
		//		{
		//			currentLayerActivationDerivatives.push_back(0.0);
		//			for (int currentLayerNeuron = 0; currentLayerNeuron < network.layers[layerNum].numOfNeurons; currentLayerNeuron++)
		//			{
		//				tempWeightDerivatives.push_back(network.layers[layerNumMinus1].activations[lastLayerNeuron]
		//					* activationBeforeFunctionDerivatives[currentLayerNeuron]
		//					* lastLayerActivationDerivatives[currentLayerNeuron]);

		//				currentLayerActivationDerivatives[lastLayerNeuron] += (network.layers[layerNum].weights[lastLayerNeuron][currentLayerNeuron]
		//					* activationBeforeFunctionDerivatives[currentLayerNeuron]
		//					* lastLayerActivationDerivatives[currentLayerNeuron]);
		//			}
		//			layerWeightDerivatives.push_back(tempWeightDerivatives);
		//			tempWeightDerivatives.clear();
		//		}

		//		for (int neuron = 0; neuron < network.layers[layerNum].numOfNeurons; neuron++)
		//		{
		//			layerBiasDerivatives.push_back(activationBeforeFunctionDerivatives[neuron] * lastLayerActivationDerivatives[neuron]);
		//		}

		//		networkWeightDerivatives.insert(networkWeightDerivatives.begin(), layerWeightDerivatives);
		//		networkBiasDerivatives.insert(networkBiasDerivatives.begin(), layerBiasDerivatives);

		//		layerWeightDerivatives.clear();
		//		layerBiasDerivatives.clear();

		//		lastLayerActivationDerivatives = currentLayerActivationDerivatives;
		//		currentLayerActivationDerivatives.clear();

		//		activationBeforeFunctionDerivatives.clear();
		//	}

		//	lastLayerActivationDerivatives.clear();

		//	allNetworkWeightDerivatives.push_back(networkWeightDerivatives);
		//	allNetworkBiasDerivatives.push_back(networkBiasDerivatives);

		//	networkBiasDerivatives.clear();
		//	networkWeightDerivatives.clear();

		//	// clear activations of network
		//	/*for (int layer = 0; layer < network.networkSize; layer++)
		//	{
		//		network.layers[layer].activations.clear();
		//	}*/
		//} 

		//// average all weight derivatives
		//vector<vector<double>> temp2DDerivatives;
		//vector<double> temp1DDerivatives;
		//double tempDerivative = 0;
		//int size1;
		//int size2;
		//int size3 = allNetworkWeightDerivatives.size();
		//std::cout << "num of layers of allNetworkWeightDerivatives's networks" << allNetworkWeightDerivatives[0].size() << endl;

		//// add em up (x is layers, y is neurons behind current layer, z is neurons in current layer)
		//for (int x = 0; x < network.networkSize-1; x++)
		//{
		//	size1 = allNetworkWeightDerivatives[0][x].size();
		//	std::cout << "x: " << x << endl;
		//	for (int y = 0; y < size1; y++)
		//	{
		//		size2 = allNetworkWeightDerivatives[0][x][y].size();
		//		//std::cout << "No. of neurons: " << size2 << endl;
		//		for (int z = 0; z < size2; z++)
		//		{
		//			for (int a = 0; a < size3; a++)
		//			{
		//				tempDerivative += allNetworkWeightDerivatives[a][x][y][z];
		//			}
		//			temp1DDerivatives.push_back(tempDerivative);
		//			tempDerivative = 0;
		//		}
		//		temp2DDerivatives.push_back(temp1DDerivatives);
		//		temp1DDerivatives.clear();
		//	}
		//	networkWeightDerivatives.push_back(temp2DDerivatives);
		//	temp2DDerivatives.clear();
		//}

		//// divide em
		////cout << networkWeightDerivatives[0].size();
		//std::cout << "time for network weight derivatives times the learn rate" << endl;
		//for (int x = 0; x < network.networkSize-1; x++)
		//{
		//	size1 = networkWeightDerivatives[x].size();
		//	for (int y = 0; y < size1; y++)
		//	{
		//		size2 = networkWeightDerivatives[x][y].size();
		//		
		//		for (int z = 0; z < size2; z++)
		//		{
		//			networkWeightDerivatives[x][y][z] = networkWeightDerivatives[x][y][z] / size3;
		//			//std::cout << networkWeightDerivatives[x][y][z] * learnRate << "   ";

		//			// apply the weight change to the network
		//			network.layers[x+1].weights[y][z] += (networkWeightDerivatives[x][y][z] * learnRate);
		//		}
		//	}
		//}

		//std::cout << endl << endl << "biases now" << endl;

		//// average all bias derivatives
		//size3 = allNetworkBiasDerivatives.size();
		//size1 = allNetworkBiasDerivatives[0].size();
		////std::cout << "size3: " << size3 << ", size1: " << size1 << endl << endl;
		//// x could potentially be each layer, then y each neuron, then z each training example... no guarantee
		//// add em up
		//for (int x = 0; x < size1; x++)
		//{
		//	
		//	size2 = allNetworkBiasDerivatives[0][x].size();
		//	std::cout << "No. of neurons: " << size2 << endl;
		//	//std::cout << "x: " << x << endl;
		//	//std::cout << "size2: " << size2 << endl;
		//	for (int y = 0; y < size2; y++)
		//	{
		//		//std::cout << "y: " << y << endl;
		//		
		//		for (int z = 0; z < size3; z++)
		//		{
		//			tempDerivative += allNetworkBiasDerivatives[z][x][y];
		//			//std::cout << "z: " << z << "   ";
		//		}
		//		temp1DDerivatives.push_back(tempDerivative);
		//		tempDerivative = 0;
		//		//std::cout << endl;
		//	}
		//	networkBiasDerivatives.push_back(temp1DDerivatives);
		//	temp1DDerivatives.clear();
		//	//std::cout << endl << endl;
		//}

		//// divide em
		////cout << size3;
		//for (int x = 0; x < network.networkSize-1; x++)
		//{
		//	size2 = networkBiasDerivatives[x].size();
		//	for (int y = 0; y < size2; y++)
		//	{
		//		//networkBiasDerivatives[x][y] = networkBiasDerivatives[x][y] / size3;
		//		networkBiasDerivatives[x][y] /= size3;
		//		//std::cout << networkBiasDerivatives[x][y] * learnRate << "   ";
		//		// apply bias change to the network
		//		network.layers[x+1].biases[y] += (networkBiasDerivatives[x][y] * learnRate);
		//	}
		//}
	}
};

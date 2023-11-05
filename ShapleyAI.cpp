#include <iostream>
#include <vector>
#include "ShapleyValueGenerator.h"
#include "AIManager.h"
#include "NetworkTrainer.h"
#include <random>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

int main()
{
    default_random_engine re;
    re.seed(time(NULL));

    ShapleyValueGenerator shapleyValueGenerator;
    vector<double> characterTable = shapleyValueGenerator.CreateRandomCharacterTable(re);

    std::cout << "characteristic function: " << endl;
    for (int i = 0; i < characterTable.size(); i++)
    {
        std::cout << characterTable[i] << " ";
    }
    std::cout << endl << endl;

    vector<double> shapleyValues = shapleyValueGenerator.CalculateShapleyValues(characterTable);

    std::cout << "correct shapley values: " << endl;
    for (int i = 0; i < shapleyValues.size(); i++)
    {
        std::cout << shapleyValues[i] << " ";
    }
    std::cout << endl;
    vector<int> testNetwork = { 65, 50, 50, 6 };;

    Network network = Network(testNetwork, leakyrelu_function, leakyrelu_function, -2, 1, -4, 2);

    vector<double> testInputs = characterTable;
    for (int i = testInputs.size(); i < 64 ; i++)
    {
        testInputs.push_back(0.0);
    }
    testInputs.push_back(shapleyValueGenerator.numOfAgents);

    vector<double> results = network.RunNetwork(testInputs);

    std::cout << "network's shapley values: " << endl;
    for (int i = 0; i < results.size(); i++)
    {
        std::cout << results[i] << " ";
    }
    std::cout << endl << endl;

    std::cout << "creating training data... " << endl;

    vector<vector<double>> inputData;
    vector<vector<double>> outputData;
    vector<double> tempIn;
    vector<double> tempOut;

    ifstream inputTrainingDataFile;

    inputTrainingDataFile.open("inputtrainingdata.txt");

    ifstream outputTrainingDataFile;

    outputTrainingDataFile.open("outputtrainingdata.txt");

    if (inputTrainingDataFile)
    {
        string temp;
        vector<double> tempList;
        stringstream line;

        while (getline(inputTrainingDataFile, temp))
        {
            temp.pop_back();
            line << temp;
            while (getline(line, temp, ',' ))
            {
                tempList.push_back(stod(temp));
            }
            inputData.push_back(tempList);
            tempList.clear();
            line.clear();
        }

        while (getline(outputTrainingDataFile, temp))
        {
            temp.pop_back();
            line << temp;
            while (getline(line, temp, ','))
            {
                tempList.push_back(stod(temp));
            }
            outputData.push_back(tempList);
            tempList.clear();
            line.clear();
        }
    }
    else
    {
        // get training data
        for (int i = 0; i < 95; i++)
        {
            // get raw input and output data
            //inputData.push_back(shapleyValueGenerator.CreateRandomCharacterTable(re));
            //outputData.push_back(shapleyValueGenerator.CalculateShapleyValues(inputData[i]));
            tempIn = shapleyValueGenerator.CreateRandomCharacterTable(re);
            tempOut = shapleyValueGenerator.CalculateShapleyValues(tempIn);

            // parse them for the network
            for (int x = tempIn.size(); x < 64; x++)
            {
                tempIn.push_back(0.0);
            }
            tempIn.push_back(shapleyValueGenerator.numOfAgents);
            inputData.push_back(tempIn);
            tempIn.clear();

            for (int x = shapleyValueGenerator.numOfAgents - 1; x < 6; x++)
            {
                tempOut.push_back(0.0);
            }
            tempOut.push_back(shapleyValueGenerator.numOfAgents);
            outputData.push_back(tempOut);
            tempOut.clear();
        }

        ofstream newInputTrainingDataFile("inputtrainingdata.txt");
        for (int i = 0; i < inputData.size(); i++)
        {
            for (int x = 0; x < inputData[i].size(); x++)
            {
                newInputTrainingDataFile << to_string(inputData[i][x]) << ",";
            }
            newInputTrainingDataFile << endl;
        }

        ofstream newOutputTrainingDataFile("outputtrainingdata.txt");
        for (int i = 0; i < outputData.size(); i++)
        {
            for (int x = 0; x < outputData[i].size(); x++)
            {
                newOutputTrainingDataFile << to_string(outputData[i][x]) << ",";
            }
            newOutputTrainingDataFile << endl;
        }
    }

    // make the network train on the training data
    std::cout << "training... " << endl;
    BackpropagationTrainer trainer;
    trainer.TrainBackpropagation(network, inputData, outputData);

    std::cout << "done training" << endl << endl;

    characterTable = shapleyValueGenerator.CreateRandomCharacterTable(re);

    std::cout << "characteristic function: " << endl;
    for (int i = 0; i < characterTable.size(); i++)
    {
        std::cout << characterTable[i] << " ";
    }
    std::cout << endl << endl;

    shapleyValues = shapleyValueGenerator.CalculateShapleyValues(characterTable);

    std::cout << "correct shapley values: " << endl;
    for (int i = 0; i < shapleyValues.size(); i++)
    {
        std::cout << shapleyValues[i] << " ";
    }
    std::cout << endl;

    testInputs = characterTable;
    for (int i = testInputs.size(); i < 64; i++)
    {
        testInputs.push_back(0.0);
    }
    testInputs.push_back(shapleyValueGenerator.numOfAgents);

    results = network.RunNetwork(testInputs);

    std::cout << "network's shapley values: " << endl;
    for (int i = 0; i < results.size(); i++)
    {
        std::cout << results[i] << " ";
    }

    //vector<vector<vector<vector<int>>>> test;
    //vector<vector<vector<int>>> addedTest;
    //vector<vector<vector<int>>> test3;
    //vector<vector<int>> test2;
    //vector<int> test1;
    //int num = 0;
    //for (int w = 0; w < 2; w++)
    //{
    //    for (int x = 0; x < 3; x++)
    //    {
    //        for (int y = 0; y < 4; y++)
    //        {
    //            for (int z = 0; z < 5; z++)
    //            {
    //                test1.push_back(num);
    //                num++;
    //            }
    //            test2.push_back(test1);
    //            test1.clear();
    //        }
    //        test3.push_back(test2);
    //        test2.clear();
    //    }
    //    test.push_back(test3);
    //    test3.clear();
    //}

    //for (int w = 0; w < test.size(); w++)
    //{
    //    cout << "network " << w << endl;
    //    for (int x = 0; x < test[w].size(); x++)
    //    {
    //        for (int y = 0; y < test[w][x].size(); y++)
    //        {
    //            for (int z = 0; z < test[w][x][y].size(); z++)
    //            {
    //                cout << test[w][x][y][z] << endl;
    //            }
    //            cout << endl << endl;
    //        }
    //        cout << endl << endl << endl;
    //    }
    //    cout << endl << endl << endl << endl;
    //}
    //cout << endl << endl << endl << endl;
    //int tempNum = 0;
    //for (int x = 0; x < test[0].size(); x++)
    //{
    //    for (int y = 0; y < test[0][x].size(); y++)
    //    {
    //        for (int z = 0; z < test[0][x][y].size(); z++)
    //        {
    //            for (int a = 0; a < test.size(); a++)
    //            {
    //                tempNum += test[a][x][y][z];
    //            }
    //            test1.push_back(tempNum);
    //            tempNum = 0;
    //        }
    //        test2.push_back(test1);
    //        test1.clear();
    //    }
    //    addedTest.push_back(test2);
    //    test2.clear();
    //}

    //for (int x = 0; x < addedTest.size(); x++)
    //{
    //    for (int y = 0; y < addedTest[x].size(); y++)
    //    {
    //        for (int z = 0; z < addedTest[x][y].size(); z++)
    //        {
    //            cout << addedTest[x][y][z] << endl;
    //        }
    //        cout << endl << endl;
    //    }
    //    cout << endl << endl << endl;
    //}

    
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

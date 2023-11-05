#include <iostream>
#include <vector>
#include <time.h>
#include <random>
#include <algorithm>
#include "ShapleyValueGenerator.h"

using namespace std;

//int ShapleyValueGenerator::Factorial(int num)
//{
//    int factorial = 1;
//    for (int i = 1; i <= num; i++)
//    {
//        factorial *= i;
//    }
//    return factorial;
//}

int numOfAgents;

vector<double> ShapleyValueGenerator::CreateRandomCharacterTable(default_random_engine& re)
{
    //srand(time(NULL));
    uniform_int_distribution<int> rangeOfAgents(2, 6);
    uniform_int_distribution<int> rangeOfNums(-100, 101);
    //default_random_engine re{};
    //re.seed(time(NULL));
    numOfAgents = rangeOfAgents(re);

    //numOfAgents = (rand() % 6) + 2;
    //numOfAgents = 4;
    //cin >> numOfAgents;
    vector<double> characterTable(pow(2, numOfAgents));

    for (int i = 0; i < pow(2, numOfAgents); i++)
    {
        //characterTable[i] = (rand() % 200) - 100;
        characterTable[i] = rangeOfNums(re);
    }

    //for (int i = 0; i < pow(2, numOfAgents); i++)
    //{
    //    cin >> characterTable[i];
    //}

    return characterTable;
}

vector<vector<int>> permutations;

void ShapleyValueGenerator::GenerateAllPermutations(vector<int>& v, int size)
{
    // if size becomes 1 then adds on the obtained permutation
    if (size == 1) {
        permutations.push_back(v);
        return;
    }

    for (int i = 0; i < size; i++) {
        GenerateAllPermutations(v, size - 1);
        if (i < size - 1)
        {
            // if size is odd, swap first and last element
            if (size % 2 == 1)
            {
                iter_swap(v.begin(), v.begin() + size - 1);
            }
            // If size is even, swap ith and last element
            else
            {
                iter_swap(v.begin() + i, v.begin() + size - 1);
            }
        }
    }
}

vector<vector<int>> ShapleyValueGenerator::GenerateAllCombinations()
{
    vector<vector<int>> combinations = { {}, {0} };
    // this size value is necessary as otherwise when looping through the size of combinations, as i add more the size just increases. infinite loop. doesnt work
    int size = combinations.size();
    vector<int> temp;
    for (int i = 1; i < numOfAgents; i++)
    {
        for (int x = 0; x < size; x++)
        {
            for (int y = 0; y < combinations[x].size(); y++)
            {
                temp.push_back(combinations[x][y]);
            }
            temp.push_back(i);

            combinations.push_back(temp);

            temp.clear();
        }
        size = combinations.size();
    }

    return combinations;
}

// hey dont forget that v1 isnt sorted here, so if you use this for something where v1 has to be an unsorted list, uhh, just remember that this is why its not working. also v1 is a reference, not a copy.
bool ShapleyValueGenerator::DoVectorsContainSameElements(vector<int>& v1, vector<int> v2)
{

    if (v1.size() != v2.size())
    {
        return false;
    }

    //sort(v1.begin(), v1.end());
    sort(v2.begin(), v2.end());

    return v1 == v2;
}

vector<double> ShapleyValueGenerator::CalculateShapleyValues(vector<double>& characterTable)
{
    permutations.clear();
    vector<int> temp;
    for (int i = 0; i < numOfAgents; i++)
    {
        temp.push_back(i);
    }
    GenerateAllPermutations(temp, numOfAgents);

    temp.clear();

    vector<vector<int>> combinations = GenerateAllCombinations();

    vector<int> currentCombination;
    vector<double> shapleyValues(numOfAgents);
    double charTableSize = characterTable.size();
    int permSize = permutations.size();
    int combSize = combinations.size();

    for (int i = 0; i < numOfAgents; i++)
    {
        for (int permutation_index = 0; permutation_index < permSize; permutation_index++)
        {
            for (int y = 0; y < numOfAgents; y++)
            {
                currentCombination.push_back(permutations[permutation_index][y]);
                if (permutations[permutation_index][y] == i)
                {
                    break;
                }
            }
            for (int y = 0; y < combSize; y++)
            {
                if (DoVectorsContainSameElements(combinations[y], currentCombination))
                {
                    currentCombination.pop_back();
                    for (int z = 0; z < combSize; z++)
                    {
                        if (DoVectorsContainSameElements(combinations[z], currentCombination))
                        {
                            shapleyValues[i] = shapleyValues[i] + (characterTable[y] - characterTable[z]);
                            break;
                        }
                    }
                    break;
                }
            }
            currentCombination.clear();
        }
        shapleyValues[i] = shapleyValues[i] / charTableSize;
    }
    return shapleyValues;
}
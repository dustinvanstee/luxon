#ifndef LUXON_DATA_SOURCE_RANDOM_CUH
#define LUXON_DATA_SOURCE_RANDOM_CUH

#include <vector>
#include <iostream>

#include "../common.cuh"
#include "../data/idataSource.cuh"

#define RAND_BLOCK_SIZE 9000     //Size of the Data in a random block
#define RAND_BLOCK_COUNT 1024    //Number of Messages in the Flow

struct randomBlock {
    char buffer[RAND_BLOCK_SIZE];
    int  bufferSize;
} typedef randomBlock_t;

class RandomData : public IDataSource {
public:
    RandomData() {

        this->dataSourceType = eDataSourceType::RANDOM;

        for (int i = 0; i < RAND_BLOCK_COUNT; ++i) {
            randomBlock_t b;
            b.bufferSize = RAND_BLOCK_SIZE;
            for (int j = 0; j < RAND_BLOCK_SIZE; ++j) {
                int r = (uint8_t)((rand() % 256) + 1);
                b.buffer[j] = r;
            }
            randomBlocks.push_back(b);
        }
    }// Constructor declaration

    //Creates a random set of updates for the market data instruments, and puts them in a flow for sending.
    std::vector<randomBlock_t> createRandomUpdate(int numMsg)
    {
        std::vector<randomBlock_t> update;
        for(int j = 0; j < numMsg; j++)
        {
            randomBlock_t b = randomBlocks[(uint8_t)(rand()%RAND_BLOCK_COUNT)];
            update.push_back(b);
        }
        return update;
    }

private:
    std::vector<randomBlock_t> randomBlocks;

};


#endif //LUXON_DATA_SOURCE_RANDOM_CUH

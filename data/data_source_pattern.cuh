#ifndef LUXON_DATA_SOURCE_PATTERN_CUH
#define LUXON_DATA_SOURCE_PATTERN_CUH

#include <vector>
#include <iostream>

#include "../common.cuh"
#include "../data/idataSource.cuh"
#include "../transport/itransport.cuh"

#define PATTERN_CODE 0xAB              //Byte Pattern to Write
#define PATTERN_BLOCK_SIZE 9000     //Size of the Data in a block

struct patternBlock {
    char buffer[PATTERN_BLOCK_SIZE];
    int  bufferSize;
} typedef patternBlock_t;

class PatternData : public IDataSource {
public:
    PatternData() {
        this->dataSourceType = eDataSourceType::PAT;
        patternBlock.bufferSize = PATTERN_BLOCK_SIZE;
        for (int j = 0; j < PATTERN_BLOCK_SIZE; ++j) {
            patternBlock.buffer[j] = PATTERN_CODE;
        }
    }// Constructor declaration

    std::vector<patternBlock_t> createPatternUpdate(int numMsg) {
        std::vector <patternBlock_t> update;
        for (int j = 0; j < numMsg; j++) {
            update.push_back(patternBlock);
        }
        return update;
    }

    void summarizeMessage(Message* m)
    {
        ITransport::printMessage(m, 25);
    }

private:
    patternBlock_t patternBlock;

};


#endif //LUXON_DATA_SOURCE_PATTERN_CUH

//
// Created by alex on 7/15/20.
//

#ifndef SENSORSIM_PROCESSOR_CUH
#define SENSORSIM_PROCESSOR_CUH

#include <unistd.h>
#include <assert.h>

#include "../common.cuh"
#include "../transport/itransport.cuh"
#include "../data/idataSource.cuh"

__global__ void gpu_count_zeros(Message* flow, int* sum, int flowLength);
void cpu_count_zeros(Message* flow, int* sum, int flowLength);

class Processor {
public:

    // Constructor declaration
    explicit Processor(ITransport *t, eDataSourceType dataSource);

    //pop a message and process if you get one.
    int procPrintMessages(int minMsg);
    int procCountZerosCPU(int minMsg);
    int procCountZerosGPU(int minMsg);
    void procDropMsg(int i);
    // Allocate memory based on device type
    void initializeMsgBlk();
    void summarizeBuffer();
    int freeMemory();

private:
    ITransport*  transport;
    MessageBlk*  msgBlkPtr;
    eDataSourceType dataSourceType; 

};


#endif //SENSORSIM_PROCESSOR_CUH

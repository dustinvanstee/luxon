#ifndef LUXON_PROCESSOR_CUH
#define LUXON_PROCESSOR_CUH

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
    explicit Processor(ITransport *t, eDataSourceType dataSource, eMsgBlkLocation msgBlockLocation );
    ~Processor();

    //pop a message and process if you get one.
    int procPrintMessages(int minMsg);
    int procCountZerosCPU(int minMsg);
    int procCountZerosGPU(int minMsg);
    void procDropMsg(int i);

private:
    ITransport*  transport;
    MessageBlk  msgBlk;
    eDataSourceType dataSourceType;
    IDataSource* dataSource;
    eMsgBlkLocation msgBlockLocation;

};


#endif //LUXON_PROCESSOR_CUH

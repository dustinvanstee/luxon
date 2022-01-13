#ifndef LUXON_SENSOR_CUH
#define LUXON_SENSOR_CUH

#include <vector>
#include <iostream>
#include <pcap.h>

#include "../common.cuh"
#include "../data/idataSource.cuh"
#include "../data/data_sample_finance.cuh"
#include "../data/data_source_random.cuh"
#include "../data/data_source_pattern.cuh"
#include "../transport/itransport.cuh"
#include "../transport/print_transport.cuh"
#include "../transport/none_transport.cuh"
#include "../transport/udp_transport.cuh"
#include "../transport/rdma_ud_transport.cuh"

class Sensor {
public:
    Sensor(ITransport*, eDataSourceType); // Constructor declaration

    //Flow Creation Functions
    int createRandomFlow(MessageBlk &mb);
    int createPatternFlow(MessageBlk &mb);
    int createFinanceFlow(MessageBlk &mb);
    int createPCAPFlow(MessageBlk &mb, std::string fileName); //TODO: Need to create a data source class for pcap, this can solve overrun issue.

    int getFlowByteLength(MessageBlk &mb);                    //Length of the flow in Bytes
    int getFlowMsgCount(MessageBlk &mb);                      //Number of Messages in the Flow
    int getFlowMsgAvgSize(MessageBlk &mb);                    //Average Size of a Message in the flow

    //Flow display
    void printFlow(MessageBlk &mb);
    int sendFlow(MessageBlk &mb);

private:
    IDataSource*        dataSource;
    ITransport*         transport;

};


#endif //LUXON_SENSOR_CUH

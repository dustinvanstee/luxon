#include "Sensor.cuh"

#include <iostream>
#include <unistd.h>
#include <chrono>

#include "../common.cuh"
#include "../data/idataSource.cuh"
#include "../data/data_sample_finance.cuh"
#include "../transport/itransport.cuh"
#include "../transport/print_transport.cuh"
#include "../transport/none_transport.cuh"
#include "../transport/udp_transport.cuh"
#include "../transport/rdma_ud_transport.cuh"

void PrintUsage()
{
    cout << "usage: sensorSim [ -s pcap ] [-t mode] [-l local-addr] [-d duration] mcast-addr" << endl;
    cout << "\t multicast group where sensor publishes data" << endl;
    cout << "\t[-s source] - a data source, can be PCAP, RANDOM, FINANCE (default: random byte pattern)" << endl;
    cout << "\t[-t transport mode] - transport mode, PRINT, UDP, RDMA-UD, NONE (default: PRINT)" << endl;
    cout << "\t[-i iterations] - iterations to send the flow. (default: 1 sec)" << endl;
    cout << "\t[-l ip-addr] - local ip addresss to bind. (default: bind to first address)" << endl;
}

int main(int argc,char *argv[], char *envp[]) {
     /*
     * Parsing the command line and validating the input
     */
    int op;
    string fileName;
    eTransportType transportType;
    eDataSourceType dataSourceType;
    string mcastAddr;
    string srcAddr;
    int numIter = 1;
    char hostBuffer[256];
    ITransport* transport;
    IDataSource* dataSource;

    while ((op = getopt(argc, argv, "d:s:l:t:")) != -1) {
        switch (op) {
            case 'l':
                srcAddr = optarg;
                break;
            case 's':
                dataSourceType = dataSource->strToDataSourceType(optarg);
                break;
            case 't':
                transportType = transport->strToTransportType(optarg);
                break;
            case 'd':
                numIter = atoi(optarg);
                break;
            default:
                PrintUsage();
                return -1;
        }
    }

    if(argc <= optind)
    {
        PrintUsage();
        return -1;
    }
    else
    {
        mcastAddr = argv[optind++];
    }
    gethostname(hostBuffer, sizeof(hostBuffer));
    cout << "********  ********  ********  ********  ********  ********" << endl;
    cout << "Sensor Simulator - Read Data Source and sends data buffers to target" << endl;
    cout << "********  ********  ********  ********  ********  ********" << endl;
    cout << "Running on " << hostBuffer <<endl;
    cout << "Local Address: " << (srcAddr.empty() ? "Default" : srcAddr) << endl;
    cout << "Mcast Group Address: " << mcastAddr << endl;
    cout << "Source: " << dataSource->DataSourceTypeToStr(dataSourceType) << endl;
    cout << "Transport Mode: " << transport->TransportTypeToStr(transportType) << endl <<endl;

    //Create the Transport
    switch(transportType) {
        case eTransportType::UDP :
            transport = new UdpTransport(srcAddr, mcastAddr, eTransportRole::SENSOR);
            break;
        case eTransportType::RDMA_UD :
            transport = new RdmaUdTransport(srcAddr, mcastAddr, eTransportRole::SENSOR);
            break;
        case eTransportType::PRINT :
            transport = new PrintTransport();
            break;
        case eTransportType::NONE :
            transport = new NoneTransport();
            break;
        default :
            cout << "No or Invalid Transport Specified" << endl;
            PrintUsage();
            return -1;
    }

    //Create the Sensor
    Sensor s = Sensor(transport, dataSourceType);

    //Create the update flow based on the data source.
    switch(dataSourceType) {
        case eDataSourceType::PCAP:
            s.createPCAPFlow(fileName);
            break;
        case eDataSourceType::RANDOM:
            s.createRandomFlow(100);
            break;
        case eDataSourceType::FINANCE:
            s.createFinanceFlow(100);
            break;
        default :
            cout << "No valid data source" << endl;
            PrintUsage();
            return -1;
    }

    cout << "Sensor Flow has " << s.getFlowMsgCount() << " messages w/ avg size " << s.getFlowMsgAvgSize() << endl;
    cout << "Sensor Flow total size is " << s.getFlowByteLength() << " bytes " << endl;
    cout << "sending flow for " << numIter << " iterations" << endl;

    long long sentMessages = 0;
    long long messageRate = 0;
    int flowLength = s.getFlowMsgCount();

    timer t_runTime;

    int i = 0;
    t_runTime.start();
    do {
        if (0 != s.sendFlow())
        {
            cout << "Transport Error Sending sensor Flow - Exiting" << endl;
            return -1;
        }
        sentMessages += flowLength;
    } while (i++ <= numIter);
    t_runTime.stop();
    cerr << "\rSent " << sentMessages << " messages\t Time: " << t_runTime.seconds_elapsed() << "/" << t_runTime.usec_elapsed() << "msec"  << endl;

    cerr << "Rate " << (sentMessages/t_runTime.usec_elapsed()) * 1000 << "Messages Per Second" << endl;


    return 0;
}
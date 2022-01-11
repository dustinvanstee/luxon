#include <iostream>
#include <unistd.h>
#include <chrono>

#include "Sensor.cuh"

void PrintUsage()
{
    cout << "usage: sensorSim [ -s pcap ] [-t mode] [-l local-addr] [-d duration] mcast-addr" << endl;
    cout << "\t multicast group where sensor publishes data" << endl;
    cout << "\t[-s source] - a data source, can be PCAP, RANDOM, FINANCE (default: random byte pattern)" << endl;
    cout << "\t[-t transport mode] - transport mode, PRINT, UDP, RDMA_UD, NONE (default: PRINT)" << endl;
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
    
    npt("test %d",0);
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

    MessageBlk msgBlk;

    //Create the update flow based on the data source.
    switch(dataSourceType) {
        case eDataSourceType::PCAP:
            s.createPCAPFlow(msgBlk, fileName);
            break;
        case eDataSourceType::RANDOM:
            s.createRandomFlow(msgBlk, MSG_BLOCK_SIZE);
            break;
        case eDataSourceType::PAT:
            s.createPatternFlow(msgBlk, MSG_BLOCK_SIZE);
            break;
        case eDataSourceType::FINANCE:
            s.createFinanceFlow(msgBlk, MSG_BLOCK_SIZE);
            break;
        default :
            cout << "No valid data source" << endl;
            PrintUsage();
            return -1;
    }
    npt("Sensor Statistics %c", ':');
    cout << "Sensor Flow has " << s.getFlowMsgCount(msgBlk) << " messages w/ avg size " << s.getFlowMsgAvgSize(msgBlk) << endl;
    cout << "Sensor Flow total size is " << s.getFlowByteLength(msgBlk) << " bytes " << endl;
    cout << "sending flow for " << numIter << " iterations" << endl;
    cout << "sending " << numIter * s.getFlowMsgCount(msgBlk) << " messages" << endl;

    long long sentMessages = 0;
    int flowLength = s.getFlowMsgCount(msgBlk);

    timer t_runTime;

    int i = 1; //First Iteration
    t_runTime.start();
    do {
        if (0 != s.sendFlow(msgBlk))
        {
            cout << "Transport Error Sending sensor Flow - Exiting" << endl;
            return -1;
        }
        sentMessages += flowLength;
    } while (i++ < numIter);
    t_runTime.stop();
    cerr << "\rSent " << sentMessages << " messages\t Time: " << t_runTime.usec_elapsed() << "usec"  << endl;

    cerr << "Rate " << (sentMessages/t_runTime.usec_elapsed()) * 1000 << " Messages Per Second" << endl;


    return 0;
}

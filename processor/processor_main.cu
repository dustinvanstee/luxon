#include "Processor.cuh"

#include <iostream>
#include <unistd.h>

#include "../transport/itransport.cuh"
#include "../transport/udp_transport.cuh"
#include "../transport/rdma_ud_transport.cuh"


#define MIN_MSG_TO_PRINT    100
#define MIN_MSG_TO_PROCESS  10'000'000  //CPU count our GPU count


void PrintUsage()
{
    cout << "usage: processorSim [-l local-addr] [-t trans] [-m mode] mcast-addr" << endl;
    cout << "\t mcast-addr - multicast group where sensor publishes data" << endl;
    cout << "\t[-l local-addr] - local ipv4 addresss to bind. (default: bind to first address)" << endl;
    cout << "\t[-t trans] - transport to use: UDP, RDMA-UD, UCX (default: UDP)" << endl;
    cout << "\t[-m mode] - run mode: PRINT, NO-PROC, CPU-COUNT, GPU-COUNT (default: PRINT)" << endl;
}

int main(int argc,char *argv[], char *envp[]) {
    /*
    * Parsing the command line and validating the input
    */
    int op;
    string fileName;
    string mode = "PRINT";
    string mcastAddr;
    string localAddr;
    char hostBuffer[256];
    string tmode = "UDP";

    while ((op = getopt(argc, argv, "m:s:l:t:")) != -1) {
        switch (op) {
            case 'l':
                localAddr = optarg;
                break;
            case 'm':
                mode = optarg;
                if (mode != "PRINT" && mode != "NO-PROC" && mode != "CPU-COUNT" && mode != "GPU-COUNT")
                {
                    PrintUsage();
                    return -1;
                }
                break;
            case 't':
                tmode = optarg;
                if (tmode != "UDP" && tmode != "RDMA-UD")
                {
                    PrintUsage();
                    return -1;
                }
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
    cout << "Processor Simulator - Receive Messages from a sensor and process them" << endl;
    cout << "********  ********  ********  ********  ********  ********" << endl;
    cout << "Running on " << hostBuffer <<endl;
    cout << "Local Address: " << (localAddr.empty() ? "Default" : localAddr) << endl;
    cout << "Mcast Group Address: " << mcastAddr << endl;
    cout << "Processor Mode: " << mode << endl;
    cout << "Transport Mode: " << tmode << endl;


    //Create the Transport
    ITransport* t;
    if(tmode == "UDP")
        t = new UdpTransport(localAddr, mcastAddr, eTransportRole::PROCESSOR);
    else if(tmode == "RDMA-UD")
        t = new RdmaUdTransport(localAddr , mcastAddr, eTransportRole::PROCESSOR);

    Processor p = Processor(t);

    if(mode == "PRINT")
    {
        cout << "This processor will print " << MIN_MSG_TO_PRINT << " msg then exit" << endl;
        p.procPrintMessages(MIN_MSG_TO_PRINT);
    }
    else if(mode == "NO-PROC")
    {
        cout << "This processor will receive " << MIN_MSG_TO_PROCESS << " msg it does no processing" << endl;
        p.procDropMsg(MIN_MSG_TO_PROCESS);
    }
    else if(mode == "CPU-COUNT")
    {
        cout << "This processor will count zeros in " << MIN_MSG_TO_PROCESS << " msg using the CPU" << endl;
        p.procCountZerosCPU(MIN_MSG_TO_PROCESS);
    }
    else if(mode == "GPU-COUNT")
    {
        cout << "This processor will count zeros in " << MIN_MSG_TO_PROCESS << " msg using the GPU" << endl;
        p.procCountZerosGPU(MIN_MSG_TO_PROCESS);
    }

    return 0;
}

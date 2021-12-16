#ifndef LUXON_PRINT_TRANSPORT_CUH
#define LUXON_PRINT_TRANSPORT_CUH

#include <netinet/in.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "../common.cuh"
#include "itransport.cuh"

using namespace std;

class PrintTransport: public ITransport {

public:
    PrintTransport();

private:
    int push(Message* msgBlk);
    int pop(Message* msgBlk, int numReqMsg, int& numRetMsg);
    int freeMessage(Message* msgBlk);
};


#endif //LUXON_PRINT_TRANSPORT_CUH

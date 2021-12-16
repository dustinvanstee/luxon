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

    int push(Message* msgBlk) override;
    int pop(Message* msgBlk, int numReqMsg, int& numRetMsg);
    int freeMessage(Message* msgBlk);

private:

};


#endif //LUXON_PRINT_TRANSPORT_CUH

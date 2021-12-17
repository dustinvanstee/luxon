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
    int push(Message* msg);
    int pop(Message** msg, int numReqMsg, int& numRetMsg, eTransportDest dest);
    int pop(Message* msg, int numReqMsg, int& numRetMsg, eTransportDest dest);
    int pop(int a);
    Message* createMessage();
    int freeMessage(Message* msg);
};


#endif //LUXON_PRINT_TRANSPORT_CUH

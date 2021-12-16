#ifndef LUXON_NULL_TRANSPORT_CUH
#define LUXON_NULL_TRANSPORT_CUH

#include <netinet/in.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "../common.cuh"
#include "itransport.cuh"

using namespace std;

class NoneTransport: public ITransport {

public:
    NoneTransport();

private:
    int push(Message* msgBlk);
    int pop(Message* msgBlk, int numReqMsg, int& numRetMsg);
    int freeMessage(Message* msgBlk);

};


#endif //LUXON_NULL_TRANSPORT_CUH

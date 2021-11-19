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
    int push(Message** msg, int count);
    int pop(Message** msg, int numReqMsg, int& numRetMsg, eTransportDest dest);
    Message* createMessage();
    int freeMessage(Message* msg);
};


#endif //LUXON_NULL_TRANSPORT_CUH

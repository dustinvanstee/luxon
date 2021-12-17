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

    int push(Message* msgBlk);
    int pop(Message* msgBlk, int numReqMsg, int& numRetMsg);
    int createMessageBlock(Message* &msgBlk, eMsgBlkLocation dest);
    int freeMessageBlock(Message* msgBlk, eMsgBlkLocation dest);

private:


};


#endif //LUXON_NULL_TRANSPORT_CUH

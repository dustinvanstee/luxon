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

    int push(Message* msgBlk, int numMsg) override;
    int pop(Message* msgBlk, int numReqMsg, int& numRetMsg) override;
    int createMessageBlock(Message* &msgBlk, eMsgBlkLocation dest) override;
    int freeMessageBlock(Message* msgBlk, eMsgBlkLocation dest) override;

private:


};


#endif //LUXON_NULL_TRANSPORT_CUH

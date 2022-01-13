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

    int push(MessageBlk* msgBlk) override;
    int pop(MessageBlk* msgBlk, int numReqMsg, int& numRetMsg) override;
    int createMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest) override;
    int freeMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest) override;

private:


};


#endif //LUXON_NULL_TRANSPORT_CUH

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

    int push(MessageBlk* msgBlk, int numMsg) override;
    int pop(MessageBlk* msgBlk, int numReqMsg, int& numRetMsg) override;
    int createMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest) override;
    int freeMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest) override;

private:
    int push_counter = 0;
    int pop_counter = 0;

};


#endif //LUXON_NULL_TRANSPORT_CUH

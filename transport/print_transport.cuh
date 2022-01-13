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

    int push(MessageBlk* msgBlk) override;
    int pop(MessageBlk* msgBlk, int numReqMsg, int& numRetMsg) override;
    int createMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest) override;
    int freeMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest) override;

private:

};


#endif //LUXON_PRINT_TRANSPORT_CUH

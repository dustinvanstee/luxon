#ifndef LUXON_UDP_TRANSPORT_CUH
#define LUXON_UDP_TRANSPORT_CUH

#include <netinet/in.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "../common.cuh"
#include "itransport.cuh"

using namespace std;

class UdpTransport: public ITransport {

public:
    UdpTransport(string localAddr, string mcastAddr, eTransportRole role);

    int push(Message* msgBlk, int numMsg) override;
    int pop(Message* msgBlk, int numReqMsg, int& numRetMsg) override;
    int createMessageBlock(Message* &msgBlk, eMsgBlkLocation dest) override;
    int freeMessageBlock(Message* msgBlk, eMsgBlkLocation dest)override;

private:
     struct ip_mreq        mcastGroup;

};


#endif //LUXON_UDP_TRANSPORT_CUH

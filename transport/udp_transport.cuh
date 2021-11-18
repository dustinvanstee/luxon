//
// Created by alex on 7/16/20.
//

#ifndef SENSORSIM_UDP_TRANSPORT_CUH
#define SENSORSIM_UDP_TRANSPORT_CUH

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

private:
    int push(Message* msg);
    int pop(Message** msg, int numReqMsg, int& numRetMsg, eTransportDest dest);
    Message* createMessage();
    int freeMessage(Message* msg);

    struct ip_mreq        mcastGroup;

};


#endif //SENSORSIM_UDP_TRANSPORT_CUH

#ifndef LUXON_RDMA_UD_TRANSPORT_CUH
#define LUXON_RDMA_UD_TRANSPORT_CUH

#include <netinet/in.h>
#include <unistd.h>
#include <string>
#include <vector>

#include </usr/include/infiniband/verbs.h>
#include </usr/include/rdma/rdma_cma.h>

#include "../common.cuh"

#include "itransport.cuh"

class RdmaUdTransport: public ITransport {

public:
    RdmaUdTransport(std::string localAddr, std::string mcastAddr, eTransportRole role);
    ~RdmaUdTransport();

    int         push(MessageBlk* msgBlk) override;
    int         pop(MessageBlk* msgBlk, int numReqMsg, int& numRetMsg) override;
    int         createMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest) override;
    int         freeMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest) override;

private:

    struct rdma_event_channel*  g_CMEventChannel;
    struct rdma_cm_id*			g_CMId;

    struct sockaddr_in6			localAddr_in;
    struct sockaddr_in6			mcastAddr_in;

    //Shared Memory Context
    struct ibv_pd*              g_pd;                            /* Protection Domain Handle */
    struct ibv_cq*              g_cq;                            /* Completion Queue Handle */

    // Address Info for the Mutlicast Group
    struct 	ibv_ah*             AddressHandle;
    uint32_t 				    RemoteQpn;
    uint32_t 				    RemoteQkey;

    ibv_mr*                     mrMsgBlk;
    ibv_send_wr                 sendWqe[MSG_BLOCK_SIZE_STATIC_OVERALLOC]; // TODO : Fix static overallocation later !! 
    ibv_recv_wr                 rcvWqe[MSG_BLOCK_SIZE_STATIC_OVERALLOC];
    ibv_wc                      cqe[MSG_BLOCK_SIZE_STATIC_OVERALLOC];


    int         initSendWqe(ibv_send_wr*, int);
    int         updateSendWqe(ibv_send_wr* wqe, void *buffer, size_t bufferlen, ibv_mr *bufferMemoryRegion);

    int         initRecvWqe(ibv_recv_wr *wqe, int);
    int         updateRecvWqe(ibv_recv_wr* wqe, void *buffer, size_t bufferlen, ibv_mr *bufferMemoryRegion);

    int         post_SEND_WQE(ibv_send_wr*);
    int         post_RECEIVE_WQE(ibv_recv_wr*);

    ibv_mr*     create_MEMORY_REGION(void* , size_t);

    int         RDMACreateQP();
    int         RDMACreateContext();

    int         RdmaMcastConnect();

    void        DestroyContext();
    void        DestroyQP();

    int         GetCMEvent(rdma_cm_event_type *EventType);
};

#endif //LUXON_RDMA_UD_TRANSPORT_CUH

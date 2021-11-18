//
// Created by alex on 8/7/20.
//

#ifndef SENSORSIM_RDMA_UD_TRANSPORT_CUH
#define SENSORSIM_RDMA_UD_TRANSPORT_CUH

#include <netinet/in.h>
#include <unistd.h>
#include <string>
#include <vector>

#include </usr/include/infiniband/verbs.h>
#include </usr/include/rdma/rdma_cma.h>

#include "../common.cuh"

#include "itransport.cuh"

#define NUM_OPERATIONS  MSG_BLOCK_SIZE //Set this to same as the block size for other transports.

class RdmaUdTransport: public ITransport {

public:
    RdmaUdTransport(std::string localAddr, std::string mcastAddr, eTransportRole role);
    ~RdmaUdTransport();

private:
    int         push(Message* msg);
    int         pop(Message** msg, int numReqMsg, int& numRetMsg, eTransportDest dest);
    Message*    createMessage();
    int         freeMessage(Message* msg);
    int         freeMsgBlock();

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

    uint8_t                     messagePool[MSG_BLOCK_SIZE * sizeof(Message)];
    bool                        messagePoolSlotFree[MSG_BLOCK_SIZE];  //Track which slots in the message pool is available.
    ibv_mr*                     mr_messagePool;

    ibv_send_wr                 dataSendWqe;
    ibv_recv_wr                 dataRcvWqe;
    ibv_wc                      dataWc;


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

#endif //SENSORSIM_RDMA_UD_TRANSPORT_CUH

#include "rdma_ud_transport.cuh"

#include <assert.h>
#include <cstdio>
#include <algorithm>
#include <arpa/inet.h>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>


int get_addr(const char *dst, struct sockaddr *addr)
{
    struct addrinfo *res;
    int ret;
    ret = getaddrinfo(dst, NULL, NULL, &res);
    if (ret)
    {
        fprintf(stderr,"ERROR: getaddrinfo failed - invalid hostname or IP address\n");
        return -1;
    }
    memcpy(addr, res->ai_addr, res->ai_addrlen);
    freeaddrinfo(res);
    return ret;
}

int RdmaUdTransport::createMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest)
{
    npt("%s:", "TRACE\n");

    if (dest == eMsgBlkLocation::HOST) {
        if(0 != createMessageBlockHelper(msgBlk, dest))
        {
            exit(EXIT_FAILURE);
        }

        //Register this new message block for RDMA access
        mrMsgBlk = create_MEMORY_REGION(msgBlk->messages, (sizeof(Message) * MSG_BLOCK_SIZE));

        //Create Match Set of QP work elements.
        for(int i = 0; i < MSG_BLOCK_SIZE; i++)
        {
            initSendWqe(&sendWqe[i], i);
            if(i == MSG_BLOCK_SIZE-1) { //There is no next block set to NULL
                sendWqe[i].next = NULL;
            } else {
                sendWqe[i].next = &sendWqe[i + 1];
            }
            updateSendWqe(&sendWqe[i], &msgBlk->messages[0], MSG_MAX_SIZE, mrMsgBlk);
            initRecvWqe(&rcvWqe[i], i);
            updateRecvWqe(&rcvWqe[i], &msgBlk->messages[0], MSG_MAX_SIZE, mrMsgBlk);
        }

    } else {
        //TODO: Need to add ability to rdma to gpu memory.
        fprintf(stderr, "ERROR: GPU Direct RDMA not supported yet\n");
        return -1;
    }

    return 0;
}

int RdmaUdTransport::freeMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest)
{
    npt("%s:", "TRACE\n");
    return freeMessageBlockHelper(msgBlk, dest);
}


RdmaUdTransport::RdmaUdTransport(std::string localAddr, std::string mcastAddr, eTransportRole role) {
    this->transportType = eTransportType::RDMA_UD;
    npt("%s:", "TRACE\n");

    s_localAddr = localAddr;
    s_mcastAddr = mcastAddr;

    // Creating socket file descriptor
    if(RDMACreateContext() != 0)
    {
        std::cerr << "Failed Create the RDMA Channel." << std::endl;
        exit(EXIT_FAILURE);
    }

    if(role == eTransportRole::SENSOR) { //Sensor

        if(RDMACreateQP() != 0)
        {
            fprintf(stdout, "Exiting - Failed to Create Queue Pair, make sure processor is running\n");
            exit(EXIT_FAILURE);
        }

    } else { //Processor

        if(RDMACreateQP() != 0)
        {
            fprintf(stderr, "Exiting - Failed to establish connection with the client\n");
            exit(EXIT_FAILURE);
        }

    }

    if(RdmaMcastConnect() != 0)
    {
        fprintf(stdout, "Exiting - Failed to establish connection to MultiCast Group\n");
        exit(EXIT_FAILURE);
    }

}

RdmaUdTransport::~RdmaUdTransport() {
    npt("%s:", "TRACE\n");
    //Clean the RDMA Contexts
    DestroyContext();
    DestroyQP();

    ibv_dereg_mr(mrMsgBlk);
}

int RdmaUdTransport::push(MessageBlk* m, int numMsg)
{
    npt("%s:", "TRACE\n");
    int err;
    int ret = 0;
    struct ibv_send_wr *bad_wqe = NULL;

    if(numMsg < MSG_BLOCK_SIZE)
    {
        sendWqe[numMsg].next = NULL; //Stop sending after numMsgs.
    }

    do {
        err = ibv_post_send(g_CMId->qp, sendWqe, &bad_wqe);

        //Error Handling BLock for Post Send
        if (err != 0)
        {
            fprintf(stderr, "ERROR: post_SEND_WQE Error %u\n", err);
            if (err == ENOMEM) //Queue Full Wait for CQ Polling Thread to Clear
            {
                fprintf(stderr, "ERROR: Send Queue Full Retry %u of 10\n", ret);
                usleep(100); //Wait 100 Microseconds, max of 1 msec
            } else {
                fprintf(stderr, "ERROR: Unrecoverable Send Queue, aborting\n");
                return -1;
            }
        }

    } while(err != 0);

    for(int i = 0; i < numMsg; i++) {
        //Wait For Completion
        int ret;

        DEBUG("DEBUG: Waiting for CQE\n");
        do {
            ret = ibv_poll_cq(g_cq, 1, &cqe[0]);
        } while (ret == 0);
        DEBUG("DEBUG: Received " << ret << " CQE Elements\n");
        //DEBUG("DEBUG: WRID(" << cqe.wr_id << ")\tStatus(" << cqe.status << ") length( " << cqe.byte_len
        //                     << ")\n");

        if (cqe[0].status == IBV_WC_RNR_RETRY_EXC_ERR) {
            usleep(50); //wait 50 us and we will try again.
            //std::cerr << "DEBUG: WRID(" << cqe.wr_id << ")\tStatus(IBV_WC_RNR_RETRY_EXC_ERR)" << std::endl;
            return -1;
        }
        if (cqe[0].status != IBV_WC_SUCCESS) {
            //std::cerr << "DEBUG: WRID(" << cqe.wr_id << ")\tStatus(" <<  cqe.status << ")" << std::endl;
            return -1;
        }

    }

    return 0;
}

/*
*  Pulls messages from the transport and places it in the message block  buffer
*/

// TODO : 121621 broke this due to code refactor for gpudirect
int RdmaUdTransport::pop(MessageBlk* msgBlk, int numReqMsg, int& numRetMsg)
{
    numRetMsg = 0;
    Message* msg = NULL;
    npt("%s:", "TRACE\n");

    do {
        //Post the RcvWQE
        msg = reinterpret_cast<Message *>(&msgBlk[numRetMsg * sizeof(Message)]);
        updateRecvWqe(&rcvWqe[numRetMsg], msgBlk->messages[numRetMsg].buffer, MSG_MAX_SIZE, mrMsgBlk);
        post_RECEIVE_WQE(&rcvWqe[numRetMsg]);

        int r;
        DEBUG("DEBUG: Waiting for CQE\n");
        do {
            //TODO: We should be pulling block size (1k) messages at a time
            r = ibv_poll_cq(g_cq, 1, &cqe[0]);
        } while (r == 0);
        DEBUG("DEBUG: Received " << r << " CQE Elements\n");

        numRetMsg += r;

        for (int j = 0; j < r; j++) {
            DEBUG ("test");
            //DEBUG("DEBUG: WRID(" << cqe.wr_id <<
            //                     ")\tStatus(" << cqe.status << ")" <<
            //                     ")\tSize(" << cqe.byte_len << ")\n");
        }

        *msg->buffer += 40;
        msg->bufferSize = cqe[0].byte_len - 40;
        msg->seqNumber = numRetMsg-1;
        msg->interval = 0;
        //m[numRetMsg-1] = msg;

        DEBUG ("DEBUG: Received Message:\n");
        #ifdef DEBUG_BUILD
        std::cerr<<"here";
        ITransport::printMessage(msg, 48);
        #endif

    } while(numRetMsg < numReqMsg);

    return 0;
}

int RdmaUdTransport::initSendWqe(ibv_send_wr* wqe, int i)
{
    struct ibv_sge *sge;
    npt("%s:", "TRACE\n");

    //wqe = (ibv_send_wr *)malloc(sizeof(ibv_send_wr));
    sge = (ibv_sge *)malloc(sizeof(ibv_sge));

    //memset(wqe, 0, sizeof(ibv_send_wr));
    memset(sge, 0, sizeof(ibv_sge));

    wqe->wr_id = i;
    wqe->opcode = IBV_WR_SEND;
    wqe->sg_list = sge;
    wqe->num_sge = 1;
    wqe->send_flags = IBV_SEND_SIGNALED;

    wqe->wr.ud.ah = AddressHandle;
    wqe->wr.ud.remote_qpn = RemoteQpn;
    wqe->wr.ud.remote_qkey = RemoteQkey;

    return 0;
}

int RdmaUdTransport::updateSendWqe(ibv_send_wr* wqe, void* buffer, size_t bufferlen, ibv_mr* bufferMemoryRegion)
{
    npt("%s wr_id :%lu \n", "TRACE", wqe->wr_id);
    wqe->sg_list->addr = (uintptr_t)buffer;
    wqe->sg_list->length = bufferlen;
    wqe->sg_list->lkey = bufferMemoryRegion->lkey;
    return 0;
}

int RdmaUdTransport::initRecvWqe(ibv_recv_wr* wqe, int id)
{
    struct ibv_sge *sge;
    npt("%s:", "TRACE\n");

    sge = (ibv_sge *)malloc(sizeof(ibv_sge));

    memset(sge, 0, sizeof(ibv_sge));

    wqe->wr_id = id;
    wqe->next = NULL;
    wqe->sg_list = sge;
    wqe->num_sge = 1;

    return 0;
}

int RdmaUdTransport::updateRecvWqe(ibv_recv_wr *wqe, void *buffer, size_t bufferlen, ibv_mr *bufferMemoryRegion) {
    npt("%s:", "TRACE\n");
    wqe->sg_list->addr = (uintptr_t)buffer;
    wqe->sg_list->length = bufferlen;
    wqe->sg_list->lkey = bufferMemoryRegion->lkey;
    return 0;
}

int RdmaUdTransport::post_RECEIVE_WQE(ibv_recv_wr* ll_wqe)
{
    npt("%s:", "TRACE\n");
    DEBUG("DEBUG: Enter post_RECEIVE_WQE\n");
    int ret;
    struct ibv_recv_wr *bad_wqe = NULL;

    ret = ibv_post_recv(g_CMId->qp, ll_wqe, &bad_wqe);
    if(ret != 0)
    {
        fprintf(stderr, "ERROR: post_RECEIVE_WQE - Couldn't Post Receive WQE\n");
        return -1;
    }

    DEBUG("DEBUG: Exit post_RECEIVE_WQE\n");
    return 0;
}

ibv_mr* RdmaUdTransport::create_MEMORY_REGION(void* buffer, size_t bufferlen)
{
    npt("%s:", "TRACE\n");
    ibv_mr* tmpmr = (ibv_mr*)malloc(sizeof(ibv_mr));
    int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    tmpmr = ibv_reg_mr(g_pd, buffer, bufferlen, mr_flags);
    if(!tmpmr)
    {
        fprintf(stderr, "ERROR: create_MEMORY_REGION: Couldn't Register memory region\n");
        return NULL;
    }

    npt("DEBUG: Memory Region was registered with addr=%p, lkey=0x%x, rkey=0x%x, flags=0x%x\n",
            buffer, tmpmr->lkey, tmpmr->rkey, mr_flags);

    return tmpmr;
}

/*
 * Create the CM Event Channel, the Connection Identifier, Bind the application to a local address
 */
int RdmaUdTransport::RDMACreateContext()
{
    int ret;
    struct rdma_cm_event *CMEvent;
    npt("%s:", "TRACE\n");

    // Open a Channel to the Communication Manager used to receive async events from the CM.
    g_CMEventChannel = rdma_create_event_channel();
    if(!g_CMEventChannel)
    {
        fprintf(stderr,"ERROR - RDMACreateContext: Failed to Create CM Event Channel");
        DestroyContext();
        return -1;
    }

    ret = rdma_create_id(g_CMEventChannel, &g_CMId, NULL, RDMA_PS_UDP);
    if(ret != 0)
    {
        fprintf(stderr,"ERROR - RDMACreateContext: Failed to Create CM ID");
        DestroyContext();
        return -1;
    }

    if(get_addr(s_localAddr.c_str(), (struct sockaddr*)&localAddr_in) != 0)
    {
        fprintf(stderr, "ERROR - RDMACreateContext: Failed to Resolve Local Address\n");
        DestroyContext();
        return -1;
    }

    if(get_addr(s_mcastAddr.c_str(), (struct sockaddr*)&mcastAddr_in) != 0)
    {
        fprintf(stderr, "ERROR - RDMACreateContext: Failed to Resolve Multicast Address Address\n");
        DestroyContext();
        return -1;
    }

    ret = rdma_bind_addr(g_CMId, (struct sockaddr*)&localAddr_in);
    if(ret != 0 )
    {
        fprintf(stderr, "ERROR - RDMACreateContext: Couldn't bind to local address\n");
        fprintf(stderr, "ERROR - errno %s\n", strerror(errno));
        return -1;
    }

    ret = rdma_resolve_addr(g_CMId,
                            (struct sockaddr*)&localAddr_in,
                            (struct sockaddr*)&mcastAddr_in,
                            2000);
    if(ret != 0 )
    {
        fprintf(stderr, "ERROR - RDMACreateContext: Couldn't resolve local address and or mcast address.\n");
        fprintf(stderr, "ERROR - errno %s\n", strerror(errno));
        return -1;
    }

    ret = rdma_get_cm_event(g_CMEventChannel, &CMEvent);
    if(ret != 0)
    {
        fprintf(stderr, "ERROR - RDMACreateContext: No Event Received Time Out\n");
        return -1;
    }
    if(CMEvent->event != RDMA_CM_EVENT_ADDR_RESOLVED)
    {
        fprintf(stderr, "ERROR - RDMACreateContext: Expected Multicast Joint Event\n");
        return -1;
    }


    return 0;
}

int RdmaUdTransport::RDMACreateQP()
{
    npt("%s:", "TRACE\n");
    int ret;
    struct ibv_qp_init_attr qp_init_attr;

    //g_CMId->qp_type = IBV_QPT_UD;
    //g_CMId->ps = RDMA_PS_UDP;

    //Create a Protection Domain
    g_pd = ibv_alloc_pd(g_CMId->verbs);
    if(!g_pd)
    {
        fprintf(stderr,"ERROR: - RDMACreateQP: Couldn't allocate protection domain\n");
        fprintf(stderr, "ERROR - errno %s\n", strerror(errno));
        return -1;
    }

    /*Create a completion Queue */
    //g_cq = ibv_create_cq(g_CMId->verbs, NUM_OPERATIONS, NULL, NULL, 0);
    g_cq = ibv_create_cq(g_CMId->verbs, MSG_BLOCK_SIZE, NULL, NULL, 1);
    if(!g_cq)
    {
        fprintf(stderr, "ERROR: RDMACreateQP - Couldn't create completion queue\n");
        fprintf(stderr, "ERROR - errno %s\n", strerror(errno));
        return -1;
    }

    /* create the Queue Pair */
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));

    qp_init_attr.qp_type = IBV_QPT_UD;
    //qp_init_attr.sq_sig_all = 0;
    qp_init_attr.send_cq = g_cq;
    qp_init_attr.recv_cq = g_cq;
    qp_init_attr.cap.max_send_wr = MSG_BLOCK_SIZE;
    qp_init_attr.cap.max_recv_wr = MSG_BLOCK_SIZE;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;

    ret = rdma_create_qp(g_CMId, g_pd, &qp_init_attr);
    if(ret != 0)
    {
        fprintf(stderr, "ERROR: RDMACreateQP: Couldn't Create Queue Pair Error\n");
        fprintf(stderr, "ERROR - errno %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

int RdmaUdTransport::RdmaMcastConnect()
{
    npt("%s:", "TRACE\n");
    int ret;
    struct rdma_cm_event *CMEvent;

    ret = rdma_join_multicast(g_CMId, (struct sockaddr*)&mcastAddr_in, NULL);
    if(ret)
    {
        fprintf(stderr, "RDMA multicast join Failed\n");
        fprintf(stderr, "ERROR - errno %s\n", strerror(errno));
        return -1;
    }

    ret = rdma_get_cm_event(g_CMEventChannel, &CMEvent);
    if(ret != 0)
    {
        fprintf(stderr, "ERROR: No Event Received Time Out\n");
        fprintf(stderr, "ERROR - errno %s\n", strerror(errno));
        return -1;
    }
    if(CMEvent->event == RDMA_CM_EVENT_MULTICAST_JOIN)
    {
        rdma_ud_param *param;
        param = &CMEvent->param.ud;

        RemoteQpn = param->qp_num;
        RemoteQkey = param->qkey;
        AddressHandle = ibv_create_ah(g_pd, &param->ah_attr);
        if (!AddressHandle)
        {
            fprintf(stderr, "ERROR OnMulticastJoin - Failed to create the Address Handle\n");
            return -1;
        }
        fprintf(stderr, "Joined Multicast Group QPN(%d) QKey(%d)\n", RemoteQpn, RemoteQkey);
    } else {

        fprintf(stderr, "Expected Multicast Joint Event\n");
        return -1;
    }



    return 0;
}

void RdmaUdTransport::DestroyContext()
{
    npt("%s:", "TRACE\n");
    if(g_CMEventChannel != NULL)
    {
        rdma_destroy_event_channel(g_CMEventChannel);
    }

    if(g_CMId != NULL)
    {
        if(rdma_destroy_id(g_CMId) != 0)
        {
            fprintf(stderr, "ERROR: DestroyContext - Failed to destroy Connection Manager Id\n");
        }
    }
}

void RdmaUdTransport::DestroyQP()
{
    npt("%s:", "TRACE\n");
    if(g_pd != NULL)
    {
        if(ibv_dealloc_pd(g_pd) != 0)
        {
            fprintf(stderr, "ERROR: DestroyQP - Failed to destroy Protection Domain\n");
        }
    }

    if(g_cq != NULL)
    {
        ibv_destroy_cq(g_cq);
        {
            fprintf(stderr, "ERROR: DestroyQP - Failed to destroy Completion Queue\n");
        }
    }

    rdma_destroy_qp(g_CMId);

}






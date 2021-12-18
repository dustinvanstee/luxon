#include "Processor.cuh"

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void gpu_count_zeros(Message* flow, int* sum, int flowLength)
{
    int indx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = indx; i < flowLength; i += stride)
    {
        for(int j = 0; j < flow[i].bufferSize; j++)
        {
            if(flow[i].buffer[j] == 0)
            {
                sum[i] += 1;
                //cout << "found a zero at msg[" << i << "] byte[" << j << "]" << endl;
            }
        }
    }
}


void cpu_count_zeros(Message* flow, int& sum, int flowLength)
{
    for(int i = 0; i < flowLength; i++)
    {
        for(int j = 0; j < flow[i].bufferSize; j++)
        {
            if(flow[i].buffer[j] == 0)
            {
                sum += 1;
                //cout << "found a zero at msg[" << i << "] byte[" << j << "]" << endl;
            }
        }
    }
}


Processor::Processor(ITransport* t) {
    transport = t;
}

void Processor::procCountZerosGPU(int minMessageToProcess) {
    timer t;

    int deviceId;
    int numberOfSMs;

    CUDA_CHECK( cudaGetDevice(&deviceId));
    CUDA_CHECK( cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));

    size_t threadsPerBlock;
    size_t numberOfBlocks;

    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;

    int msgCountReturned = 0;
    int processedMessages = 0;
    int sum =0;

    Message* msgBlk = NULL;//Create array that is max message block size

    if(!transport->createMessageBlock(msgBlk, eMsgBlkLocation::DEVICE)){
        // print error messge
    }

    int* blockSum;   //Array with sum of zeros for this message
    size_t sumArraySize = MSG_BLOCK_SIZE * sizeof(int);
    checkCuda( cudaMallocManaged(&blockSum, sumArraySize));
   // cout << "Processing on GPU using " <<  numberOfBlocks << " blocks with " << threadsPerBlock << " threads per block" << endl;

    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(msgBlk, MSG_BLOCK_SIZE, msgCountReturned)) {
            exit(EXIT_FAILURE);
        }

        cudaMemPrefetchAsync(msgBlk, MSG_BLOCK_SIZE*sizeof(Message), deviceId);

        if(msgCountReturned > 0) //If there are new messages process them
        {
            gpu_count_zeros <<< numberOfBlocks, threadsPerBlock >>>(msgBlk, blockSum, msgCountReturned);

            checkCuda( cudaGetLastError() );
            checkCuda( cudaDeviceSynchronize() ); //Wait for GPU threads to complete

            cudaMemPrefetchAsync(blockSum, sumArraySize, cudaCpuDeviceId);

            for(int k = 0; k < msgCountReturned; k++)
            {
                sum += blockSum[k]; //Add all the counts to the accumulator
                blockSum[k] = 0;
            }

            processedMessages += msgCountReturned;
            pt("GPU Loop processing complete.  Processed %d messages.\n", processedMessages );

        }
        //m.clear();
        msgCountReturned=0;

    }

    checkCuda( cudaFree(msgBlk));
    checkCuda( cudaFree(blockSum));

    std::cout << "\n Processing Completed: " << std::endl;
    std::cout << "\t processed " << processedMessages << " in " << t.seconds_elapsed() << " sec" << std::endl;
    std::cout << "\t total zero's in messages = " << sum << std::endl;

    exit(EXIT_SUCCESS);
}

int Processor::procCountZerosCPU(int minMessageToProcess) {
    timer t;

    int msgCountReturned = 0;
    int sum = 0;
    int processedMessages = 0;

    //Intitialize the receive buffer
    Message* msgBlk;//Create array that is max message block size
    if(!transport->createMessageBlock(msgBlk, eMsgBlkLocation::HOST)){
        // print error messge
    }

    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(msgBlk, MSG_BLOCK_SIZE, msgCountReturned)) {
            exit(EXIT_FAILURE);
        }

        if(msgCountReturned > 0) //If there are new messages process them
        {
            std::cerr << "\rProcessed " << processedMessages << " messages";
            cpu_count_zeros(msgBlk, sum, msgCountReturned);
            processedMessages += msgCountReturned;
        }
        msgCountReturned=0;

    }

    //Free the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        transport->freeMessageBlock(&msgBlk[i],eMsgBlkLocation::HOST) ;
    }

    std::cout << "\nProcessing Completed: " << std::endl;
    std::cout << "\t processed " << processedMessages << " in " << t.seconds_elapsed() << " sec" << std::endl;
    std::cout << "\t total zero's in messages = " << sum << std::endl;
    exit(EXIT_SUCCESS);
}

void Processor::procDropMsg(int minMessageToProcess) {
    timer t;

    Message* msgBlk;
    if(!transport->createMessageBlock(msgBlk, eMsgBlkLocation::HOST)){
        exit(EXIT_FAILURE);
    }

    int msgCountReturned = 0;
    int processedMessages = 0;

    t.start();
    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(msgBlk, MSG_BLOCK_SIZE, msgCountReturned)) {
            exit(EXIT_FAILURE);
        }

        if(msgCountReturned > 0) //If there are new messages process them
        {
            std::cerr << "\rProcessed " << processedMessages << " messages";
            processedMessages += msgCountReturned;
        }
        msgCountReturned=0;

    }
    t.stop();

    //Free the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        transport->freeMessageBlock(&msgBlk[i],eMsgBlkLocation::DEVICE);
    }

    std::cout << "\nProcessing Completed: " << std::endl;
    std::cout << "\t processed " << processedMessages << " in " << t.seconds_elapsed() << " sec" << std::endl;
    exit(EXIT_SUCCESS);
}

int Processor::procPrintMessages(int minMessageToProcess) {
    Message* msgBlk;
    if(!transport->createMessageBlock(msgBlk, eMsgBlkLocation::HOST)){
        exit(EXIT_FAILURE);
    }
    
    int processedCount = 0;
    int messagesReturned = 0;

    do {

        if (0 != transport->pop(msgBlk, MSG_BLOCK_SIZE, messagesReturned)) {
            exit(EXIT_FAILURE);
        }

        processedCount += messagesReturned;

        std::cout << "Printing first bytes of " << min(messagesReturned,minMessageToProcess) << " messages" << std::endl;
        for(int i = 0; i<min(messagesReturned,minMessageToProcess); i++)
        {
            transport->printMessage(&msgBlk[i], 32);
            std::cout << std::endl;
        }
    } while (processedCount < minMessageToProcess);

    //Free the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        transport->freeMessageBlock(&msgBlk[i],eMsgBlkLocation::HOST);
    }

    //Simple process (i.e. print)
    std::cout << "Processing Completed: found " << processedCount << " messages" << std::endl;
    exit(EXIT_SUCCESS);
}

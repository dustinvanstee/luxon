//
// Created by alex on 7/15/20.
//

#include "Processor.cuh"
#include <cuda_runtime.h>
// #include <helper_cuda.h>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHECK_LINE(a,file,line) { cudaError_t __cuer = a; if (cudaSuccess != __cuer) { ::fprintf (stderr, "[CUDA-ERRROR] @ %s:%d -- %d : %s -- running %s\n", file,line, __cuer, ::cudaGetErrorString(__cuer),#a) ; ::exit(__cuer) ; } }
#define CUDA_CHECK(a) CUDA_CHECK_LINE(a,__FILE__,__LINE__)
#define CU_CHECK_LINE(a,file,line) { CUresult __cuer = a; if (CUDA_SUCCESS != __cuer) { const char* errstr; ::cuGetErrorString(__cuer, &errstr) ; ::fprintf (stderr, "[CU-ERRROR] @ %s:%d -- %d : %s -- running %s\n", file,line, __cuer, errstr,#a) ; ::exit(__cuer) ; } }
#define CU_CHECK(a) CU_CHECK_LINE(a,__FILE__,__LINE__)

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


void cpu_count_zeros(Message** flow, int& sum, int flowLength)
{
    for(int i = 0; i < flowLength; i++)
    {
        for(int j = 0; j < flow[i]->bufferSize; j++)
        {
            if(flow[i]->buffer[j] == 0)
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

    CUDA_CHECK( cudaGetDevice(&deviceId) );
    npt("Cuda device id:%d\n", deviceId);
    CUDA_CHECK( cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));
    
    npt("Cuda sm:%d\n", numberOfSMs);
    size_t threadsPerBlock;
    size_t numberOfBlocks;

    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;

    int msgCountReturned = 0;
    int processedMessages = 0;
    int sum =0;

    //void* m_ptr;  
    //Message* m[MSG_BLOCK_SIZE];//Create array that is max message block size
    uint8_t * d; //The data we will store.
    size_t msgBlockSize = MSG_BLOCK_SIZE * sizeof(Message);
    size_t msgDataSize = MSG_MAX_SIZE * MSG_BLOCK_SIZE;
    // of these examples, m0 increments properly via debuuger!
    Message *m0;
    CUDA_CHECK( cudaMallocManaged((void **)&m0, msgBlockSize));
    Message **m1;
    CUDA_CHECK( cudaMallocManaged((void **)&m1, msgBlockSize));
    Message *m2[MSG_BLOCK_SIZE];
    CUDA_CHECK( cudaMallocManaged((void **)&m2, msgBlockSize));
    //checkCuda( cudaMallocManaged(&d, msgDataSize));
    //checkCuda( cudaMallocManaged(&m_ptr, msgBlockSize));
    //Message **m = reinterpret_cast<Message**>(m_ptr);
    //Message *m2[MSG_BLOCK_SIZE];
    //Message *m2[MSG_BLOCK_SIZE] = reinterpret_cast<Message (*)[MSG_BLOCK_SIZE]>(m1);
    //Message temp = {.interval=10, .bufferSize=4};

    //Intitialize the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        m0[i].interval = 0;
        m0[i].bufferSize = 0;
        m0[i].seqNumber = i;
        m0[i].buffer[0] = 0x01;
        m0[i].buffer[1] = 0x02;
    }

    npt("ti:%d\n", m0[0].interval);
    //m[0]->interval = temp.interval;
    //m[]->interval = temp.interval;
    int* blockSum;   //Array with sum of zeros for this message
    size_t sumArraySize = MSG_BLOCK_SIZE * sizeof(int);
    CUDA_CHECK( cudaMallocManaged(&blockSum, sumArraySize));
   // cout << "Processing on GPU using " <<  numberOfBlocks << " blocks with " << threadsPerBlock << " threads per block" << endl;
    
    transport->pop(8);

    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(m0, MSG_BLOCK_SIZE, msgCountReturned, eTransportDest::DEVICE)) {
            exit(EXIT_FAILURE);
        }

        CUDA_CHECK( cudaMemPrefetchAsync(m0, msgBlockSize, deviceId) );

        if(msgCountReturned > 0) //If there are new messages process them
        {
            std::cerr << "\rProcessed " << processedMessages << " messages\n";
            gpu_count_zeros <<< numberOfBlocks, threadsPerBlock >>>(m0, blockSum, msgCountReturned);

            CUDA_CHECK( cudaGetLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() ); //Wait for GPU threads to complete

            CUDA_CHECK(cudaMemPrefetchAsync(blockSum, sumArraySize, cudaCpuDeviceId));

            for(int k = 0; k < msgCountReturned; k++)
            {
                sum += blockSum[k]; //Add all the counts to the accumulator
                blockSum[k] = 0;
            }

            processedMessages += msgCountReturned;
        }
        //m.clear();
        msgCountReturned=0;

    }

    checkCuda( cudaFree(m0));
    checkCuda( cudaFree(blockSum));

    std::cout << "\n Processing Completed: " << std::endl;
    std::cout << "\t processed " << processedMessages << " in " << t.seconds_elapsed() << " sec" << std::endl;
    std::cout << "\t total zero's in messages = " << sum << std::endl;

    exit(EXIT_SUCCESS);
}

////////////////////////////
// CPU
////////////////////////////
int Processor::procCountZerosCPU(int minMessageToProcess) {
    timer t;

    Message* m[MSG_BLOCK_SIZE];
    int msgCountReturned = 0;
    int sum = 0;
    int processedMessages = 0;

    //Intitialize the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        m[i] = transport->createMessage();
    }

    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(m, MSG_BLOCK_SIZE, msgCountReturned, eTransportDest::HOST)) {
            exit(EXIT_FAILURE);
        }

        if(msgCountReturned > 0) //If there are new messages process them
        {
            std::cerr << "\rProcessed " << processedMessages << " messages";
            cpu_count_zeros(m, sum, msgCountReturned);
            processedMessages += msgCountReturned;
        }
        msgCountReturned=0;

    }

    //Free the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        transport->freeMessage(m[i]);
    }

    std::cout << "\nProcessing Completed: " << std::endl;
    std::cout << "\t processed " << processedMessages << " in " << t.seconds_elapsed() << " sec" << std::endl;
    std::cout << "\t total zero's in messages = " << sum << std::endl;
    exit(EXIT_SUCCESS);
}

void Processor::procDropMsg(int minMessageToProcess) {
    timer t;

    Message* m[MSG_BLOCK_SIZE];
    int msgCountReturned = 0;
    int processedMessages = 0;

    //Intitialize the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        m[i] = transport->createMessage();
    }

    t.start();
    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(m, MSG_BLOCK_SIZE, msgCountReturned, eTransportDest::HOST)) {
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
        transport->freeMessage(m[i]);
    }

    std::cout << "\nProcessing Completed: " << std::endl;
    std::cout << "\t processed " << processedMessages << " in " << t.seconds_elapsed() << " sec" << std::endl;
    exit(EXIT_SUCCESS);
}

int Processor::procPrintMessages(int minMessageToProcess) {
    Message* m[MSG_BLOCK_SIZE];
    int processedCount = 0;
    int r = 0;

    //Intitialize the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        m[i] = transport->createMessage();
    }

    do {

        if (0 != transport->pop(m, MSG_BLOCK_SIZE, r, eTransportDest::HOST)) {
            exit(EXIT_FAILURE);
        }

        processedCount += r;

        std::cout << "Printing first bytes of " << min(r,minMessageToProcess) << " messages" << std::endl;
        for(int i = 0; i<min(r,minMessageToProcess); i++)
        {
            transport->printMessage(m[i], 32);
            std::cout << std::endl;
        }
    } while (processedCount < minMessageToProcess);

    //Free the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        transport->freeMessage(m[i]);
    }

    //Simple process (i.e. print)
    std::cout << "Processing Completed: found " << processedCount << " messages" << std::endl;
    exit(EXIT_SUCCESS);
}

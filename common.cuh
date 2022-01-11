// Change log
// 121821 DV added a pt function for formatted print during normal runtime fixed DEBUG_BUILD logic 
#ifndef LUXON_COMMON_CUH
#define LUXON_COMMON_CUH

#include <iostream>
#include <chrono>
#include <nvtx3/nvToolsExt.h>

#define DEBUG_BUILD 0 // just set to 0 or 1 

#if DEBUG_BUILD > 0
    #define DEBUG(x) std::cerr << x
    #define DEBUG_DETAIL(x) x
    #define npt(fmt, ...) \
        do { if (DEBUG_BUILD) fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, \
                                __LINE__, __func__, __VA_ARGS__); } while (0)
#else
    #define DEBUG(x) do {} while (0)
    #define DEBUG_DETAIL(x) do {} while (0)
    #define npt(fmt, ...) do {} while (0)
#endif
// Use npt for debug printing, use pt for nice formatted print with line numbers and file name
#define pt(fmt, ...) \
        do { fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, \
                                __LINE__, __func__, __VA_ARGS__); } while (0)

#define PRINT_UPDATE_DELAY 1    //Used with timer

#define MSG_MAX_SIZE 1500       //Max size of a message must be > RAND_FLOW_MSG_SIZE or max size message from pcap
#define MSG_BLOCK_SIZE 1024     //Number of messages to process in parallel

#define CUDA_CHECK_LINE(a,file,line) { cudaError_t __cuer = a; if (cudaSuccess != __cuer) { ::fprintf (stderr, "[CUDA-ERRROR] @ %s:%d -- %d : %s -- running %s\n", file,line, __cuer, ::cudaGetErrorString(__cuer),#a) ; ::exit(__cuer) ; } }
#define CUDA_CHECK(a) CUDA_CHECK_LINE(a,__FILE__,__LINE__)
#define CU_CHECK_LINE(a,file,line) { CUresult __cuer = a; if (CUDA_SUCCESS != __cuer) { const char* errstr; ::cuGetErrorString(__cuer, &errstr) ; ::fprintf (stderr, "[CU-ERRROR] @ %s:%d -- %d : %s -- running %s\n", file,line, __cuer, errstr,#a) ; ::exit(__cuer) ; } }
#define CU_CHECK(a) CU_CHECK_LINE(a,__FILE__,__LINE__)

class timer
{
    typedef std::chrono::steady_clock clock ;
    typedef std::chrono::seconds seconds ;

public:
    void reset() {
        start_time = clock::now() ;
    }
    void start() {
        start_time = clock::now() ;
    }
    void stop() {
        stop_time = clock::now() ;
    }

    unsigned long long seconds_elapsed() const
    {
        return std::chrono::duration_cast<seconds>( stop_time - start_time ).count() ;
    }

    unsigned long long usec_elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>( stop_time - start_time ).count() ;
    }

private:
    clock::time_point start_time = clock::now() ;
    clock::time_point stop_time = clock::now() ;
};


#endif //LUXON_COMMON_CUH

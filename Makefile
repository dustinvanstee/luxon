# Change Log
# 121821 DV added header files to target lines to trigger rebuild if header changes

CXXCUDA=/usr/local/cuda-11.4/bin/nvcc
LIBS = -libverbs -lrdmacm -lcuda -lpcap # -lcudart

OBJ=$(wildcard lib/*.o)
DHEADERS=$(shell echo */*cuh)


all: bin/sensor-debug bin/processor-debug
p: bin/processor-debug

# 121021 dv modified to get build caching effect : https://stackoverflow.com/questions/7815400/how-do-i-make-makefile-to-recompile-only-changed-files

.PHONY : clean all
lib/libtransport.a: lib/none_transport.o lib/print_transport.o lib/udp_transport.o lib/rdma_ud_transport.o
	ar r lib/libtransport.a lib/none_transport.o lib/print_transport.o lib/udp_transport.o lib/rdma_ud_transport.o

lib/none_transport.o: transport/udp_transport.cu transport/itransport.cuh common.cuh
	$(CXXCUDA) -c -g -o lib/none_transport.o transport/none_transport.cu

lib/print_transport.o: transport/udp_transport.cu transport/itransport.cuh common.cuh
	$(CXXCUDA) -c -g -o lib/print_transport.o transport/print_transport.cu

lib/udp_transport.o: transport/udp_transport.cu transport/itransport.cuh common.cuh
	$(CXXCUDA) -c -g -o lib/udp_transport.o transport/udp_transport.cu

lib/rdma_ud_transport.o: transport/rdma_ud_transport.cu transport/itransport.cuh common.cuh
	$(CXXCUDA) -c -g -o lib/rdma_ud_transport.o transport/rdma_ud_transport.cu

# $@ is the name of target:wildcard
# $^ filename of all prerequisites
bin/processor-debug: lib/libtransport.a processor/processor_main.cu processor/Processor.cu $(DHEADERS)
	$(CXXCUDA) -rdc=true -g -o bin/processor-debug processor/Processor.cu processor/processor_main.cu lib/libtransport.a  $(LIBS)

bin/sensor-debug: lib/libtransport.a sensor/sensor_main.cu sensor/Sensor.cu $(DHEADERS)
	$(CXXCUDA) -g -o bin/sensor-debug sensor/sensor_main.cu sensor/Sensor.cu lib/libtransport.a $(LIBS)

clean:
	rm bin/* lib/*
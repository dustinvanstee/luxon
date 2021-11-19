CXXCUDA=/usr/local/cuda-11.4/bin/nvcc
LIBS = -libverbs -lrdmacm -lcuda -lpcap

all: sensor-debug processor-debug

transport: none_transport print_transport udp_transport rdma_ud_transport
	ar ru lib/libtransport.a lib/none_transport.o lib/print_transport.o lib/udp_transport.o lib/rdma_ud_transport.o

none_transport: transport/udp_transport.cu
	$(CXXCUDA) -c -g -o lib/none_transport.o transport/none_transport.cu

print_transport: transport/udp_transport.cu
	$(CXXCUDA) -c -g -o lib/print_transport.o transport/print_transport.cu

udp_transport: transport/udp_transport.cu
	$(CXXCUDA) -c -g -o lib/udp_transport.o transport/udp_transport.cu

rdma_ud_transport: transport/rdma_ud_transport.cu
	$(CXXCUDA) -c -g -o lib/rdma_ud_transport.o transport/rdma_ud_transport.cu


processor-debug: transport processor/processor_main.cu processor/Processor.cu
	$(CXXCUDA) -g -o bin/processor-debug.out processor/Processor.cu processor/processor_main.cu lib/libtransport.a   $(LIBS)

sensor-debug: transport sensor/sensor_main.cu sensor/Sensor.cu
	$(CXXCUDA) -g -o bin/sensor-debug.out sensor/sensor_main.cu sensor/Sensor.cu lib/libtransport.a $(LIBS)

clean:
	rm -f bin/* lib/*

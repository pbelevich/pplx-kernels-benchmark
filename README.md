# Perplexity Kernels Benchmark
https://github.com/ppl-ai/pplx-kernels

## Building NCCL Tests Docker image

```bash
GDRCOPY_VERSION=v2.4.4
EFA_INSTALLER_VERSION=1.41.0
AWS_OFI_NCCL_VERSION=v1.14.2
NCCL_VERSION=v2.26.6-1
NCCL_TESTS_VERSION=v2.15.2
TAG="efa${EFA_INSTALLER_VERSION}-ofi${AWS_OFI_NCCL_VERSION}-nccl${NCCL_VERSION}-tests${NCCL_TESTS_VERSION}"
NCCL_CONTAINER_IMAGE_NAME_TAG="nccl-tests:${TAG}"
```

```bash
docker build --progress=plain -f ../nccl-tests/nccl-tests.Dockerfile \
       --build-arg="EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
       --build-arg="AWS_OFI_NCCL_VERSION=${AWS_OFI_NCCL_VERSION}" \
       --build-arg="NCCL_VERSION=${NCCL_VERSION}" \
       --build-arg="NCCL_TESTS_VERSION=${NCCL_TESTS_VERSION}" \
       -t ${NCCL_CONTAINER_IMAGE_NAME_TAG} \
       .
```

### Building NVSHMEM Docker image on top of NCCL Tests Docker base image
https://github.com/aws-samples/awsome-distributed-training/tree/main/micro-benchmarks/nvshmem

```bash
NVSHMEM_VERSION=3.2.5-1
TAG="efa${EFA_INSTALLER_VERSION}-ofi${AWS_OFI_NCCL_VERSION}-nccl${NCCL_VERSION}-tests${NCCL_TESTS_VERSION}-nvshmem${NVSHMEM_VERSION}"
NVSHMEM_CONTAINER_IMAGE_NAME_TAG="nvshmem:${TAG}"
```

```bash
docker build --progress=plain -f ../nvshmem/nvshmem.Dockerfile \
       --build-arg="EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
       --build-arg="AWS_OFI_NCCL_VERSION=${AWS_OFI_NCCL_VERSION}" \
       --build-arg="NCCL_VERSION=${NCCL_VERSION}" \
       --build-arg="NCCL_TESTS_VERSION=${NCCL_TESTS_VERSION}" \
       --build-arg="NVSHMEM_VERSION=${NVSHMEM_VERSION}" \
       -t ${NVSHMEM_CONTAINER_IMAGE_NAME_TAG} \
       .
```

## Building Perplexity Kernels Docker image on top of NVSHMEM Docker image

```bash
TAG="efa${EFA_INSTALLER_VERSION}-ofi${AWS_OFI_NCCL_VERSION}-nccl${NCCL_VERSION}-tests${NCCL_TESTS_VERSION}-nvshmem${NVSHMEM_VERSION}"
PPLX_CONTAINER_IMAGE_NAME_TAG="pplx-kernels:${TAG}"
```

```bash
docker build --progress=plain -f ./pplx-kernels.Dockerfile \
       --build-arg="EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
       --build-arg="AWS_OFI_NCCL_VERSION=${AWS_OFI_NCCL_VERSION}" \
       --build-arg="NCCL_VERSION=${NCCL_VERSION}" \
       --build-arg="NCCL_TESTS_VERSION=${NCCL_TESTS_VERSION}" \
       --build-arg="NVSHMEM_VERSION=${NVSHMEM_VERSION}" \
       -t ${PPLX_CONTAINER_IMAGE_NAME_TAG} \
       .
```

```bash
enroot import -o ./pplx-kernels.sqsh dockerd://${PPLX_CONTAINER_IMAGE_NAME_TAG}
```

## Running Perplexity Kernels Benchmark

```bash
sbatch pplx-kernels.sbatch
```

## Check the logs

```bash
tail -f -n +0 slurm-XXX.out
```

## Example of the output

```
EP=16 DP=16
E	E/tok	tok	dim	Dispatch_lat	       Dispatch_bw	Dispatch_bytes       Combine_lat	       Combine_bw	Combine_bytes	       Torch_lat	       Torch_bw	Torch_bytes	NVSHMEM_lat	       NVSHMEM_bw	NVSHMEM_bytes
64	6	1	2048	118.9μs ±  5.1μs	0.107GB/s	12672	              120.1μs ±  5.5μs	0.205GB/s	24576	              88.5μs ±  8.4μs	1.541GB/s	135168	       99.9μs ±  8.9μs	1.362GB/s	135168
64	6	4	2048	186.6μs ±  2.6μs	0.272GB/s	50688	              146.2μs ±  3.5μs	0.673GB/s	98304	              95.2μs ±  6.0μs	5.697GB/s	540672	       106.7μs ±  5.5μs	5.082GB/s	540672
64	6	8	2048	262.7μs ±  1.8μs	0.386GB/s	101376	              221.5μs ±  3.9μs	0.888GB/s	196608	              109.4μs ±  5.2μs	9.905GB/s	1081344	173.3μs ± 255.6μs	8.908GB/s	1081344
64	6	16	2048	396.9μs ±  4.0μs	0.511GB/s	202752	              738.7μs ±  7.4μs	0.532GB/s	393216	              146.7μs ±  9.0μs	14.796GB/s	2162688	185.0μs ±  5.7μs	11.702GB/s	2162688
64	6	32	2048	677.4μs ± 44.0μs	0.601GB/s	405504	              1442.7μs ± 12.8μs	0.545GB/s	786432	              141.3μs ± 12.0μs	30.777GB/s	4325376	306.3μs ±  8.0μs	14.130GB/s	4325376
64	6	64	2048	1528.9μs ± 46.8μs	0.531GB/s	811008	              1984.4μs ± 50.6μs	0.793GB/s	1572864	       296.1μs ± 26.3μs	29.434GB/s	8650752	412.9μs ± 17.0μs	20.984GB/s	8650752
64	6	128	2048	3116.0μs ± 36.6μs	0.521GB/s	1622016	       4070.8μs ± 110.7μs	0.773GB/s	3145728	       641.1μs ± 129.0μs	28.085GB/s	17301504	475.0μs ± 25.7μs	36.520GB/s	17301504
256	8	1	7168	297.0μs ±  4.9μs	0.199GB/s	59136	              233.4μs ± 10.5μs	0.492GB/s	114688	              149.4μs ±  7.8μs	12.702GB/s	1892352	133.1μs ±  3.6μs	14.232GB/s	1892352
256	8	4	7168	386.7μs ±  5.6μs	0.612GB/s	236544	              253.4μs ± 12.1μs	1.814GB/s	458752	              310.7μs ±  8.6μs	24.378GB/s	7569408	310.7μs ± 16.9μs	24.426GB/s	7569408
256	8	8	7168	525.6μs ±  3.2μs	0.900GB/s	473088	              296.5μs ±  6.2μs	3.096GB/s	917504	              614.3μs ± 14.7μs	24.656GB/s	15138816	577.8μs ± 18.4μs	26.226GB/s	15138816
256	8	16	7168	737.0μs ±  5.7μs	1.284GB/s	946176	              562.7μs ± 11.0μs	3.262GB/s	1835008	       599.0μs ± 21.3μs	50.607GB/s	30277632	653.3μs ± 26.6μs	46.414GB/s	30277632
256	8	32	7168	1140.4μs ± 37.1μs	1.661GB/s	1892352	       1920.7μs ± 80.2μs	1.914GB/s	3670016	       925.6μs ± 100.6μs	66.127GB/s	60555264	1404.5μs ± 113.2μs	43.390GB/s	60555264
256	8	64	7168	2580.6μs ± 39.8μs	1.467GB/s	3784704	       3565.1μs ± 198.0μs	2.065GB/s	7340032	       2325.2μs ± 221.4μs	52.543GB/s	121110528	2246.1μs ± 238.3μs	54.476GB/s	121110528
256	8	128	7168	5512.0μs ± 43.2μs	1.373GB/s	7569408	       7931.1μs ± 288.5μs	1.853GB/s	14680064	       3227.7μs ± 278.8μs	75.582GB/s	242221056	3456.7μs ± 89.8μs	70.118GB/s	242221056
```
# Perplexity Kernels Benchmark
https://github.com/ppl-ai/pplx-kernels

## Building NCCL Tests Docker image

```bash
GDRCOPY_VERSION=v2.4.4
EFA_INSTALLER_VERSION=1.42.0
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

## Building NVSHMEM Docker image on top of NCCL Tests Docker base image
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

## Core dump

1. run `ulimit -c unlimited` and check that `srun -N <num of nodes> bash -c "ulimit -c"` should print `unlimited` <num of nodes> times
2. run `srun -N <num of nodes> sudo bash -c "mkdir -p /tmp/coredump && echo '/tmp/coredump/core.%e.%p' > /proc/sys/kernel/core_pattern"`
3. run `sbatch pplx-kernels.sbatch`

## Example of the output

```
EP=16 DP=16
E	E/tok	tok	dim	Dispatch_lat	Dispatch_bw	Dispatch_bytes	Combine_lat	Combine_bw	Combine_bytes	Torch_lat	Torch_bw	Torch_bytes	NVSHMEM_lat	NVSHMEM_bw	NVSHMEM_bytes
64	6	1	2048	132.6μs ±  5.6μs	0.096GB/s	12672	132.1μs ±  6.6μs	0.187GB/s	24576	101.4μs ± 66.2μs	1.510GB/s	135168	109.5μs ±  4.9μs	1.237GB/s	135168
64	6	4	2048	195.2μs ±  6.2μs	0.260GB/s	50688	152.9μs ±  3.7μs	0.643GB/s	98304	92.4μs ±  4.9μs	5.866GB/s	540672	118.8μs ±  5.8μs	4.562GB/s	540672
64	6	8	2048	266.8μs ±  2.6μs	0.380GB/s	101376	255.3μs ± 31.0μs	0.777GB/s	196608	96.5μs ±  4.8μs	11.228GB/s	1081344	134.5μs ±  3.9μs	8.045GB/s	1081344
64	6	16	2048	411.1μs ±  8.6μs	0.493GB/s	202752	710.9μs ± 38.5μs	0.554GB/s	393216	151.3μs ± 32.3μs	14.723GB/s	2162688	163.8μs ±  8.2μs	13.237GB/s	2162688
64	6	32	2048	690.7μs ± 10.3μs	0.587GB/s	405504	1444.0μs ± 21.7μs	0.545GB/s	786432	136.2μs ±  3.7μs	31.780GB/s	4325376	319.8μs ±  5.2μs	13.530GB/s	4325376
64	6	64	2048	1758.1μs ± 33.1μs	0.461GB/s	811008	1879.4μs ± 90.1μs	0.839GB/s	1572864	464.6μs ± 53.9μs	18.859GB/s	8650752	399.3μs ± 19.0μs	21.715GB/s	8650752
64	6	128	2048	3686.1μs ± 44.8μs	0.440GB/s	1622016	3773.8μs ± 295.5μs	0.839GB/s	3145728	1692.6μs ± 403.7μs	10.630GB/s	17301504	489.0μs ± 25.4μs	35.471GB/s	17301504
256	8	1	7168	262.9μs ±  3.9μs	0.225GB/s	59136	255.7μs ±  5.9μs	0.449GB/s	114688	147.1μs ± 10.7μs	12.931GB/s	1892352	141.0μs ±  6.3μs	13.443GB/s	1892352
256	8	4	7168	353.6μs ±  7.2μs	0.669GB/s	236544	306.9μs ± 11.9μs	1.497GB/s	458752	306.5μs ±  8.3μs	24.715GB/s	7569408	309.2μs ±  5.2μs	24.485GB/s	7569408
256	8	8	7168	471.0μs ±  4.2μs	1.005GB/s	473088	416.2μs ± 27.4μs	2.212GB/s	917504	579.9μs ± 12.0μs	26.117GB/s	15138816	584.5μs ± 22.9μs	25.937GB/s	15138816
256	8	16	7168	802.7μs ± 13.8μs	1.179GB/s	946176	611.2μs ± 110.9μs	3.106GB/s	1835008	573.7μs ± 91.6μs	54.066GB/s	30277632	633.3μs ± 41.9μs	48.014GB/s	30277632
256	8	32	7168	1296.0μs ± 51.7μs	1.462GB/s	1892352	1508.4μs ± 148.8μs	2.455GB/s	3670016	1385.1μs ± 142.3μs	44.154GB/s	60555264	1204.3μs ± 101.6μs	50.648GB/s	60555264
256	8	64	7168	3139.9μs ± 63.7μs	1.206GB/s	3784704	3698.4μs ± 303.2μs	1.998GB/s	7340032	2469.2μs ± 236.3μs	49.471GB/s	121110528	2405.4μs ± 128.4μs	50.490GB/s	121110528
256	8	128	7168	6955.0μs ± 104.0μs	1.089GB/s	7569408	7750.2μs ± 423.0μs	1.900GB/s	14680064	4651.6μs ± 422.3μs	52.512GB/s	242221056	3771.7μs ± 102.4μs	64.266GB/s	242221056
```

```
EP=32 DP=32
E	E/tok	tok	dim	Dispatch_lat	Dispatch_bw	Dispatch_bytes	Combine_lat	Combine_bw	Combine_bytes	Torch_lat	Torch_bw	Torch_bytes	NVSHMEM_lat	NVSHMEM_bw	NVSHMEM_bytes
64	6	1	2048	162.6μs ±  7.6μs	0.078GB/s	12672	195.5μs ± 21.1μs	0.127GB/s	24576	187.4μs ± 52.6μs	0.751GB/s	135168	226.5μs ± 15.7μs	0.600GB/s	135168
64	6	4	2048	246.3μs ±  7.4μs	0.206GB/s	50688	245.3μs ± 10.0μs	0.401GB/s	98304	208.2μs ± 11.2μs	2.604GB/s	540672	218.7μs ±  9.2μs	2.477GB/s	540672
64	6	8	2048	369.0μs ±  5.4μs	0.275GB/s	101376	381.2μs ±  6.4μs	0.516GB/s	196608	195.4μs ± 22.7μs	5.591GB/s	1081344	222.6μs ± 18.2μs	4.888GB/s	1081344
64	6	16	2048	617.5μs ±  7.4μs	0.328GB/s	202752	567.9μs ± 19.1μs	0.693GB/s	393216	241.5μs ± 29.9μs	9.078GB/s	2162688	213.2μs ± 20.7μs	10.248GB/s	2162688
64	6	32	2048	1109.3μs ± 13.4μs	0.366GB/s	405504	1117.0μs ± 23.3μs	0.704GB/s	786432	472.5μs ± 26.2μs	9.182GB/s	4325376	304.4μs ± 17.8μs	14.254GB/s	4325376
64	6	64	2048	2672.8μs ± 28.2μs	0.303GB/s	811008	2802.0μs ± 127.1μs	0.562GB/s	1572864	433.4μs ± 90.1μs	20.882GB/s	8650752	500.9μs ± 24.6μs	17.311GB/s	8650752
64	6	128	2048	5788.9μs ± 45.4μs	0.280GB/s	1622016	6780.3μs ± 345.7μs	0.465GB/s	3145728	1298.0μs ± 305.6μs	13.944GB/s	17301504	785.8μs ± 54.2μs	22.120GB/s	17301504
256	8	1	7168	389.2μs ±  6.7μs	0.152GB/s	59136	469.7μs ±  7.7μs	0.244GB/s	114688	201.9μs ± 25.7μs	9.499GB/s	1892352	219.7μs ± 20.0μs	8.688GB/s	1892352
256	8	4	7168	502.5μs ± 17.3μs	0.471GB/s	236544	554.0μs ± 19.6μs	0.829GB/s	458752	342.6μs ± 13.4μs	22.125GB/s	7569408	320.3μs ±  9.6μs	23.657GB/s	7569408
256	8	8	7168	698.4μs ± 10.7μs	0.678GB/s	473088	646.5μs ±  9.5μs	1.419GB/s	917504	512.5μs ±  8.8μs	29.546GB/s	15138816	576.2μs ± 10.8μs	26.284GB/s	15138816
256	8	16	7168	1007.7μs ± 16.5μs	0.939GB/s	946176	1185.0μs ± 24.2μs	1.549GB/s	1835008	1224.2μs ± 48.2μs	24.770GB/s	30277632	1227.1μs ± 71.6μs	24.750GB/s	30277632
256	8	32	7168	1801.4μs ± 83.2μs	1.052GB/s	1892352	2138.5μs ± 132.0μs	1.722GB/s	3670016	1181.8μs ± 63.3μs	51.378GB/s	60555264	1393.2μs ± 58.2μs	43.536GB/s	60555264
256	8	64	7168	4361.0μs ± 45.7μs	0.868GB/s	3784704	5130.8μs ± 331.1μs	1.436GB/s	7340032	3459.8μs ± 312.3μs	35.260GB/s	121110528	2876.0μs ± 195.9μs	42.300GB/s	121110528
256	8	128	7168	10189.8μs ± 83.8μs	0.743GB/s	7569408	11986.4μs ± 678.8μs	1.228GB/s	14680064	7152.4μs ± 555.3μs	34.054GB/s	242221056	5691.6μs ± 340.4μs	42.704GB/s	242221056
```

```
EP=64 DP=64
E	E/tok	tok	dim	Dispatch_lat	Dispatch_bw	Dispatch_bytes	Combine_lat	Combine_bw	Combine_bytes	Torch_lat	Torch_bw	Torch_bytes	NVSHMEM_lat	NVSHMEM_bw	NVSHMEM_bytes
64	6	1	2048	262.4μs ± 289.4μs	0.064GB/s	12672	1046.4μs ± 715.0μs	0.036GB/s	24576	468.3μs ± 161.5μs	0.306GB/s	135168	444.3μs ± 47.5μs	0.308GB/s	135168
64	6	4	2048	412.9μs ± 277.7μs	0.141GB/s	50688	530.7μs ± 213.8μs	0.206GB/s	98304	438.3μs ± 75.9μs	1.257GB/s	540672	477.7μs ± 35.0μs	1.138GB/s	540672
64	6	8	2048	531.8μs ± 15.3μs	0.191GB/s	101376	651.3μs ± 38.3μs	0.303GB/s	196608	370.0μs ± 31.3μs	2.941GB/s	1081344	479.0μs ± 35.5μs	2.270GB/s	1081344
64	6	16	2048	998.0μs ± 55.6μs	0.204GB/s	202752	1163.9μs ± 44.7μs	0.338GB/s	393216	398.8μs ± 40.7μs	5.474GB/s	2162688	461.9μs ± 46.0μs	4.727GB/s	2162688
64	6	32	2048	1614.3μs ± 31.1μs	0.251GB/s	405504	1294.3μs ± 58.9μs	0.609GB/s	786432	688.8μs ± 78.4μs	6.355GB/s	4325376	471.3μs ± 36.1μs	9.228GB/s	4325376
64	6	64	2048	4199.1μs ± 127.6μs	0.193GB/s	811008	3318.5μs ± 147.2μs	0.475GB/s	1572864	687.4μs ± 184.8μs	13.344GB/s	8650752	597.2μs ± 49.8μs	14.576GB/s	8650752
64	6	128	2048	8943.1μs ± 224.8μs	0.181GB/s	1622016	6525.6μs ± 372.0μs	0.484GB/s	3145728	2961.6μs ± 418.6μs	5.956GB/s	17301504	920.0μs ± 44.3μs	18.847GB/s	17301504
256	8	1	7168	410.9μs ± 14.2μs	0.144GB/s	59136	610.1μs ± 23.3μs	0.188GB/s	114688	391.0μs ± 38.5μs	4.885GB/s	1892352	455.8μs ± 57.6μs	4.210GB/s	1892352
256	8	4	7168	504.3μs ± 18.3μs	0.470GB/s	236544	874.9μs ± 29.9μs	0.525GB/s	458752	385.1μs ± 20.5μs	19.706GB/s	7569408	514.6μs ± 27.3μs	14.750GB/s	7569408
256	8	8	7168	985.3μs ± 142.7μs	0.488GB/s	473088	901.1μs ± 72.5μs	1.023GB/s	917504	596.6μs ± 25.4μs	25.418GB/s	15138816	643.0μs ± 29.0μs	23.589GB/s	15138816
256	8	16	7168	1486.6μs ± 22.3μs	0.637GB/s	946176	1283.2μs ± 57.2μs	1.432GB/s	1835008	976.0μs ± 21.3μs	31.036GB/s	30277632	1157.1μs ± 58.1μs	26.227GB/s	30277632
256	8	32	7168	2604.0μs ± 74.1μs	0.727GB/s	1892352	2706.0μs ± 282.5μs	1.367GB/s	3670016	2652.8μs ± 133.1μs	22.881GB/s	60555264	2502.6μs ± 105.9μs	24.239GB/s	60555264
256	8	64	7168	6523.8μs ± 56.5μs	0.580GB/s	3784704	6349.0μs ± 527.6μs	1.164GB/s	7340032	4201.1μs ± 468.1μs	29.184GB/s	121110528	3307.3μs ± 158.3μs	36.701GB/s	121110528
256	8	128	7168	15363.3μs ± 94.8μs	0.493GB/s	7569408	16192.6μs ± 866.3μs	0.909GB/s	14680064	7556.4μs ± 565.3μs	32.217GB/s	242221056	6107.9μs ± 303.3μs	39.751GB/s	242221056
```
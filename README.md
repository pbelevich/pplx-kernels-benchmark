# Perplexity Kernels Benchmark
https://github.com/ppl-ai/pplx-kernels

Updated to [NVSHMEM 2d7d25f0816235e3c2b51779571ec032606ea0dd](https://github.com/NVIDIA/nvshmem/tree/2d7d25f0816235e3c2b51779571ec032606ea0dd) and PPLX-KERNELS [Sep 19, 2025](https://github.com/ppl-ai/pplx-kernels/commit/2bd6e30fab358829bd01d1c0907be8088ff9e5e1)

## Git clone NVSHMEM

[NVSHMEM 2d7d25f0816235e3c2b51779571ec032606ea0dd](https://github.com/NVIDIA/nvshmem/tree/2d7d25f0816235e3c2b51779571ec032606ea0dd):
```
git clone https://github.com/NVIDIA/nvshmem.git && cd ./nvshmem && git checkout 2d7d25f0816235e3c2b51779571ec032606ea0dd && cd ..
```

devel branch:
```
git clone https://github.com/NVIDIA/nvshmem.git && cd ./nvshmem && git checkout devel && cd ..
```

## Building Perplexity Kernels Docker image

```bash
GDRCOPY_VERSION=v2.5.1
EFA_INSTALLER_VERSION=1.43.3
NCCL_VERSION=v2.28.7-1
NCCL_TESTS_VERSION=v2.16.9
NVSHMEM_VERSION=2d7d25f0816235e3c2b51779571ec032606ea0dd
TAG="efa${EFA_INSTALLER_VERSION}-nccl${NCCL_VERSION}-tests${NCCL_TESTS_VERSION}-nvshmem${NVSHMEM_VERSION}"
PPLX_CONTAINER_IMAGE_NAME_TAG="pplx-kernels:${TAG}"
```

```bash
docker build --progress=plain -f ./pplx-kernels.Dockerfile \
       --build-arg="EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
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

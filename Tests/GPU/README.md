This test works on both GPUs and CPUs.

## Nvidia GPU

To compile,

```
   make -j8 USE_CUDA=TRUE CUDA_ARCH=NN
```

Here `NN` is something like `80`. You can run
`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
to find out the compute capability of your GPU.

## AMD GPU

To compile,

```
   make -j8 USE_HIP=TRUE AMD_ARCH=xxx
```

Here `xxx` is something like `gfx90a`.

## Intel GPU

To compile,

```
   make -j8 USE_SYCL=TRUE
```

## CPU

To compile,

```
   make -j8 USE_CPU=TRUE [COMP=gcc]
```

If `COMP` is not set, the default is `gcc`. Currently available options are
`gcc`, `clang`, and `intel`. If one needs to specify the path to the
compiler, below is an example.

```
   make -j8 USE_CPU=TRUE COMP=clang CXX=/usr/bin/clang++-17
```



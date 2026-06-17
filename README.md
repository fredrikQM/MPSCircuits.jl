# MPSCircuits.jl

MPSCircuits.jl is a Julia package for compiling matrix product states (MPS) into quantum state-preparation circuits.

The package is designed for workflows where a classical tensor-network state serves as an input for downstream quantum algorithms, e.g. preparing high-quality guide states for QPE or data-loading for quantum machine learning.

## Features

- Compile an MPS into a state-preparation circuit.
- Refine an existing circuit with environment-tensor sweeps.
- Evaluate compiled-circuit fidelity against a target MPS.
- Transpile compiled circuits to elementary gate-sets with a Cartan KAK decomposition.
- Optional GPU acceleration on NVIDIA GPUs.

## Installation

Until the package is registered, install directly from GitHub.

```julia
using Pkg
Pkg.add(url="https://github.com/Quantum-Motion/MPSCircuits.jl.git")
```

Then load the package:

```julia
using MPSCircuits
```

## Optional GPU Support (CUDA)

CUDA is optional. CPU workflows work without CUDA.jl.

If `CUDA.jl` is installed, the package's CUDA extension is automatically available.

```julia
using Pkg
Pkg.add("CUDA")
```

Recommended runtime pattern:

```julia
use_gpu = true

if use_gpu
    try
        @eval using CUDA
        if !CUDA.functional()
            @warn "CUDA loaded, but no functional device found. Falling back to CPU."
            use_gpu = false
        end
    catch err
        @warn "CUDA could not be loaded ($(typeof(err))). Falling back to CPU."
        use_gpu = false
    end
end

backend = use_gpu ? :gpu : :cpu
```

## Quickstart

```julia
using MPSCircuits
using ITensors
using ITensorMPS

# Build a small spin model and get an MPS with DMRG
N = 8
sites = siteinds("S=1/2", N)
os = OpSum()
for j in 1:(N - 1)
    os += -1.0, "Sz", j, "Sz", j + 1
end
for j in 1:N
    os += -0.5, "Sx", j
end
H = MPO(os, sites)
psi0 = MPS(sites, [isodd(n) ? "Up" : "Dn" for n in 1:N])
_, target_mps = dmrg(H, psi0; nsweeps=2, maxdim=[10], cutoff=[1e-10])

# Compile a preparation circuit
circuit = compile_mps_circuit(
    target_mps,
    MPSCircuits.DecomposeAllAnalytical();
    n_layers_max=5,
    tolerance=1e-6,
    backend=:cpu,
)

# Evaluate fidelity
fid = evaluate_circuit_fidelity(circuit, target_mps)
println("Compiled circuit fidelity: ", fid)

# Optional refinement
refined = refine_mps_circuit(
    target_mps,
    circuit;
    n_iterations=5,
    backend=:cpu,
)

# Export transpiled operations
write_circuit_json(refined, "compiled_circuit.json")
```

## Running Tests

From the package root:

```julia
using Pkg
Pkg.test()
```

## Project Status

This repository is under active development. APIs may evolve before a stable `1.0` release.

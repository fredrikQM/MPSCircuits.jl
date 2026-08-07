# Copyright 2026 Quantum Motion Technologies Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

@testset "Backend Selection API" begin
    N = 8
    sites = siteinds("S=1/2", N)
    os = OpSum()
    for j in 1:(N-1)
        os += -1.0, "Sz", j, "Sz", j + 1
    end
    for j in 1:N
        os += -0.5, "Sx", j
    end
    H = MPO(os, sites)
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi_init = MPS(sites, state)
    _, mps = dmrg(H, psi_init; nsweeps=2, maxdim=[10], cutoff=[1e-10])

    cpu_circuit = MPSCircuits.compile_mps_circuit(mps, MPSCircuits.DecomposeAllAnalytical(); n_layers_max=2, tolerance=1e-12, backend=:cpu)
    auto_circuit = MPSCircuits.compile_mps_circuit(mps, MPSCircuits.DecomposeAllAnalytical(); n_layers_max=2, tolerance=1e-12, backend=:auto)
    gpu_fallback_circuit = MPSCircuits.compile_mps_circuit(mps, MPSCircuits.DecomposeAllAnalytical(); n_layers_max=2, tolerance=1e-12, backend=:gpu, gpu_fallback=true)

    @test cpu_circuit isa Vector
    @test auto_circuit isa Vector
    @test gpu_fallback_circuit isa Vector

    @test MPSCircuits.backend_from_symbol(:cpu) == MPSCircuits.BackendCPU
    @test MPSCircuits.backend_from_symbol(:gpu) == MPSCircuits.BackendGPU
    @test MPSCircuits.backend_from_symbol(:auto) == MPSCircuits.BackendAuto
    @test_throws ArgumentError MPSCircuits.backend_from_symbol(:invalid_backend)
    @test MPSCircuits.transfer_policy_from_symbol(:prefer_gpu_residency) == MPSCircuits.PreferGPUResidency
    @test MPSCircuits.transfer_policy_from_symbol(:strict_single_backend) == MPSCircuits.StrictSingleBackend
    @test_throws ArgumentError MPSCircuits.transfer_policy_from_symbol(:invalid_policy)

    strict_policy_circuit = MPSCircuits.compile_mps_circuit(
        mps,
        MPSCircuits.DecomposeAllAnalytical();
        n_layers_max=2,
        tolerance=1e-12,
        backend=:cpu,
        transfer_policy=:strict_single_backend,
    )
    @test strict_policy_circuit isa Vector

    has_functional_gpu = MPSCircuits.resolve_backend(MPSCircuits.BackendConfig(backend=MPSCircuits.BackendAuto)) == MPSCircuits.BackendGPU
    if has_functional_gpu
        strict_gpu_circuit = MPSCircuits.compile_mps_circuit(mps, MPSCircuits.DecomposeAllAnalytical(); n_layers_max=2, tolerance=1e-12, backend=:gpu, gpu_fallback=false)
        @test strict_gpu_circuit isa Vector
    else
        @test_throws ErrorException MPSCircuits.compile_mps_circuit(mps, MPSCircuits.DecomposeAllAnalytical(); n_layers_max=2, tolerance=1e-12, backend=:gpu, gpu_fallback=false)
    end
end

@testset "Backend Conversion API" begin
    sites = siteinds("S=1/2", 2)
    mps = MPS(sites, ["Up", "Dn"])
    mps_dense = ITensorMPS.dense(mps)

    gate_tensor = ITensors.itensor(ComplexF64[1 0; 0 1], ITensors.prime(sites[1]), sites[1])
    gate = MPSCircuits.UnitaryGate(gate_tensor, [1], [sites[1]])
    circuit = MPSCircuits.AbstractGate[gate]

    converted_mps_cpu = MPSCircuits.to_backend(mps_dense, MPSCircuits.BackendCPU; precision=:fp64)
    converted_gate_cpu = MPSCircuits.to_backend(gate, MPSCircuits.BackendCPU; precision=:preserve)
    converted_circuit_cpu = MPSCircuits.to_backend(circuit, MPSCircuits.BackendCPU; precision=:fp32)

    @test converted_mps_cpu isa ITensorMPS.MPS
    @test converted_gate_cpu isa MPSCircuits.UnitaryGate
    @test converted_circuit_cpu isa Vector{MPSCircuits.AbstractGate}
    @test MPSCircuits.backend_of(converted_mps_cpu) == MPSCircuits.BackendCPU
    @test MPSCircuits.backend_of(converted_gate_cpu) == MPSCircuits.BackendCPU
    @test MPSCircuits.backend_of(converted_circuit_cpu) == MPSCircuits.BackendCPU
    @test_throws ArgumentError MPSCircuits.to_backend(gate, MPSCircuits.BackendCPU; precision=:bad_precision)
    @test_throws ArgumentError MPSCircuits.apply_gate(gate, mps_dense; mixed_device=:invalid_policy)

    has_functional_gpu = MPSCircuits.resolve_backend(MPSCircuits.BackendConfig(backend=MPSCircuits.BackendAuto)) == MPSCircuits.BackendGPU
    if has_functional_gpu
        import CUDA

        gate_gpu_fp32 = MPSCircuits.to_backend(gate, MPSCircuits.BackendGPU; precision=:fp32)
        gate_gpu_fp64 = MPSCircuits.to_backend(gate, MPSCircuits.BackendGPU; precision=:fp64)
        mps_gpu_fp32 = MPSCircuits.to_backend(mps_dense, MPSCircuits.BackendGPU; precision=:fp32)
        mps_cpu_back = MPSCircuits.to_backend(mps_gpu_fp32, MPSCircuits.BackendCPU; precision=:preserve)

        gate_arr_fp32 = ITensors.array(gate_gpu_fp32.tensor, ITensors.prime(sites[1]), sites[1])
        gate_arr_fp64 = ITensors.array(gate_gpu_fp64.tensor, ITensors.prime(sites[1]), sites[1])
        mps_arr_fp32 = begin
            inds = ITensors.inds(mps_gpu_fp32[1])
            ITensors.array(mps_gpu_fp32[1], inds...)
        end
        mps_arr_cpu = begin
            inds = ITensors.inds(mps_cpu_back[1])
            ITensors.array(mps_cpu_back[1], inds...)
        end

        @test gate_arr_fp32 isa CUDA.CuArray
        @test gate_arr_fp64 isa CUDA.CuArray
        @test mps_arr_fp32 isa CUDA.CuArray
        @test mps_arr_cpu isa Array
        @test eltype(gate_arr_fp32) == ComplexF32
        @test eltype(gate_arr_fp64) == ComplexF64

        @test MPSCircuits.backend_of(gate_gpu_fp32) == MPSCircuits.BackendGPU
        @test MPSCircuits.backend_of(mps_gpu_fp32) == MPSCircuits.BackendGPU

        @test_throws ArgumentError MPSCircuits.apply_gate(gate_gpu_fp32, mps_dense; mixed_device=:error)
        @test MPSCircuits.apply_gate(gate_gpu_fp32, mps_dense; mixed_device=:coerce) isa ITensorMPS.MPS
        @test_throws ArgumentError MPSCircuits.apply_circuit([gate_gpu_fp32], mps_dense; mixed_device=:error)
        @test MPSCircuits.apply_circuit([gate_gpu_fp32], mps_dense; mixed_device=:coerce) isa ITensorMPS.MPS

        mixed_backend_circuit = MPSCircuits.AbstractGate[gate_gpu_fp32, gate]
        @test_throws ArgumentError MPSCircuits.apply_circuit(mixed_backend_circuit, mps_dense; mixed_device=:error)
        @test MPSCircuits.apply_circuit(mixed_backend_circuit, mps_dense; mixed_device=:coerce) isa ITensorMPS.MPS
    end
end

@testset "Stage 5 GPU Procrustes Consistency" begin
    has_functional_gpu = MPSCircuits.resolve_backend(MPSCircuits.BackendConfig(backend=MPSCircuits.BackendAuto)) == MPSCircuits.BackendGPU
    if has_functional_gpu
        N = 6
        sites = siteinds("S=1/2", N)
        os = OpSum()
        for j in 1:(N-1)
            os += -1.0, "Sz", j, "Sz", j + 1
        end
        for j in 1:N
            os += -0.5, "Sx", j
        end
        H = MPO(os, sites)
        state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
        psi_init = MPS(sites, state)
        _, mps = dmrg(H, psi_init; nsweeps=2, maxdim=[10], cutoff=[1e-10])

        mps_gpu = MPSCircuits.to_backend(ITensorMPS.dense(mps), MPSCircuits.BackendGPU; precision=:fp32)
        circuit_gpu = MPSCircuits.compile_mps_circuit(
            mps,
            MPSCircuits.DecomposeAllAnalytical();
            n_layers_max=2,
            tolerance=1e-8,
            backend=:gpu,
            gpu_fallback=false,
            precision=:fp32,
            transfer_policy=:strict_single_backend,
        )

        @test !isempty(circuit_gpu)
        gate_count = min(2, length(circuit_gpu))
        gate_indices = collect(1:gate_count)

        MPSCircuits.replace_gates!(
            mps_gpu,
            circuit_gpu,
            MPSCircuits.RollingEnvironment(),
            gate_indices;
            max_bond_dim=max(4, 2 * ITensorMPS.maxlinkdim(mps_gpu)),
            working_cutoff=1e-10,
            backend=MPSCircuits.BackendGPU,
        )

        for gate_index in gate_indices
            @test MPSCircuits.backend_of(circuit_gpu[gate_index]) == MPSCircuits.BackendGPU
        end
    else
        @test true
    end
end

@testset "Step 6 Tiny Kernel CPU Policy" begin
    has_functional_gpu = MPSCircuits.resolve_backend(MPSCircuits.BackendConfig(backend=MPSCircuits.BackendAuto)) == MPSCircuits.BackendGPU
    if has_functional_gpu
        import CUDA

        mat_gpu = CUDA.cu(ComplexF32[1 0; 0 1])
        q_full = MPSCircuits.full_positive_qr(mat_gpu)
        @test q_full isa Matrix

        sites = siteinds("S=1/2", 2)
        env_data_gpu = CUDA.cu(rand(ComplexF32, 2, 2, 2, 2))
        env = ITensors.itensor(env_data_gpu, ITensors.prime(sites[1]), ITensors.prime(sites[2]), sites[1], sites[2])
        gate = MPSCircuits.new_optimal_gate(env, [sites[1], sites[2]])
        @test MPSCircuits.backend_of(gate) == MPSCircuits.BackendCPU
    else
        @test true
    end
end

@testset "Step 7 Backend Parity and Precision" begin
    N = 8
    sites = siteinds("S=1/2", N)
    os = OpSum()
    for j in 1:(N-1)
        os += -1.0, "Sz", j, "Sz", j + 1
    end
    for j in 1:N
        os += -0.5, "Sx", j
    end
    H = MPO(os, sites)
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi_init = MPS(sites, state)
    _, mps = dmrg(H, psi_init; nsweeps=2, maxdim=[10], cutoff=[1e-10])

    # CPU baseline always available.
    cpu_fp64_circuit = MPSCircuits.compile_mps_circuit(
        mps,
        MPSCircuits.DecomposeAllAnalytical();
        n_layers_max=2,
        tolerance=1e-8,
        backend=:cpu,
        precision=:fp64,
        transfer_policy=:strict_single_backend,
    )
    cpu_fp64_fidelity = MPSCircuits.evaluate_circuit_fidelity(cpu_fp64_circuit, mps; cutoff=1e-12)
    @test cpu_fp64_fidelity > 0.70

    has_functional_gpu = MPSCircuits.resolve_backend(MPSCircuits.BackendConfig(backend=MPSCircuits.BackendAuto)) == MPSCircuits.BackendGPU
    if has_functional_gpu
        import CUDA

        mps_dense = ITensorMPS.dense(mps)
        mps_gpu_fp32 = MPSCircuits.to_backend(mps_dense, MPSCircuits.BackendGPU; precision=:fp32)
        mps_gpu_fp64 = MPSCircuits.to_backend(mps_dense, MPSCircuits.BackendGPU; precision=:fp64)

        mps_gpu_fp32_arr = begin
            inds = ITensors.inds(mps_gpu_fp32[1])
            ITensors.array(mps_gpu_fp32[1], inds...)
        end
        mps_gpu_fp64_arr = begin
            inds = ITensors.inds(mps_gpu_fp64[1])
            ITensors.array(mps_gpu_fp64[1], inds...)
        end

        @test mps_gpu_fp32_arr isa CUDA.CuArray
        @test mps_gpu_fp64_arr isa CUDA.CuArray
        @test eltype(mps_gpu_fp32_arr) in (Float32, ComplexF32)
        @test eltype(mps_gpu_fp64_arr) in (Float64, ComplexF64)

        gpu_fp32_circuit = MPSCircuits.compile_mps_circuit(
            mps,
            MPSCircuits.DecomposeAllAnalytical();
            n_layers_max=2,
            tolerance=1e-8,
            backend=:gpu,
            gpu_fallback=false,
            precision=:fp32,
            transfer_policy=:strict_single_backend,
        )
        gpu_fp64_circuit = MPSCircuits.compile_mps_circuit(
            mps,
            MPSCircuits.DecomposeAllAnalytical();
            n_layers_max=2,
            tolerance=1e-8,
            backend=:gpu,
            gpu_fallback=false,
            precision=:fp64,
            transfer_policy=:strict_single_backend,
        )

        # Evaluate fidelity on CPU to avoid mixed-device behavior in this test.
        gpu_fp32_circuit_cpu = MPSCircuits.to_backend(gpu_fp32_circuit, MPSCircuits.BackendCPU; precision=:preserve)
        gpu_fp64_circuit_cpu = MPSCircuits.to_backend(gpu_fp64_circuit, MPSCircuits.BackendCPU; precision=:preserve)
        gpu_fp32_fidelity = MPSCircuits.evaluate_circuit_fidelity(gpu_fp32_circuit_cpu, mps; cutoff=1e-12)
        gpu_fp64_fidelity = MPSCircuits.evaluate_circuit_fidelity(gpu_fp64_circuit_cpu, mps; cutoff=1e-12)

        @test abs(cpu_fp64_fidelity - gpu_fp32_fidelity) <= 5e-2
        @test abs(cpu_fp64_fidelity - gpu_fp64_fidelity) <= 2e-2
    else
        @test_throws ErrorException MPSCircuits.compile_mps_circuit(
            mps,
            MPSCircuits.DecomposeAllAnalytical();
            n_layers_max=2,
            tolerance=1e-8,
            backend=:gpu,
            gpu_fallback=false,
            precision=:fp32,
        )

        # Fallback mode should keep CPU-only environments green.
        gpu_fallback_circuit = MPSCircuits.compile_mps_circuit(
            mps,
            MPSCircuits.DecomposeAllAnalytical();
            n_layers_max=2,
            tolerance=1e-8,
            backend=:gpu,
            gpu_fallback=true,
            precision=:fp32,
        )
        @test gpu_fallback_circuit isa Vector
    end
end

@testset "Basic Compilation Tests" begin
    # 1. Setup Parameters
    N = 50
    j_coupling = 1.0
    h_field = 0.5  # Transverse field

    # 2. Define Site Indices
    # "S=1/2" is the standard site type for qubits/spins
    sites = siteinds("S=1/2", N)

    # 3. Construct Hamiltonian using OpSum
    os = OpSum()
    for j in 1:(N-1)
        # Interaction term: -J * Z_i * Z_{i+1}
        os += -j_coupling, "Sz", j, "Sz", j + 1
    end
    for j in 1:N
        # Field term: -h * X_i
        os += -h_field, "Sx", j
    end

    # Convert OpSum to MPO
    H = MPO(os, sites)

    # 4. Initialize State
    # Start with a random product state (bond dimension 1)
    # or a specific one like "Up"
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi_init = MPS(sites, state)

    # 5. DMRG Settings (Sweeps)
    # Each sweep gradually increases bond dimension (maxdim) 
    # and decreases the truncation error (cutoff)
    nsweeps = 5
    maxdim = [10, 20, 100]
    cutoff = [1e-10]

    # 6. Run DMRG
    energy, mps = dmrg(H, psi_init; nsweeps, maxdim, cutoff)

    circuit = MPSCircuits.compile_mps_circuit(mps, MPSCircuits.DecomposeAllAnalytical(); n_layers_max=10, tolerance=1e-4)
    fidelity = MPSCircuits.evaluate_circuit_fidelity(circuit, mps; cutoff=1e-12)
    @test fidelity > 0.95

    circuit = MPSCircuits.compile_mps_circuit(mps, MPSCircuits.IterativeDecomposeOptimizeAll(); optimization_protocol=MPSCircuits.TelescopingEnvironment(), n_layers_max=10, n_iterations_per_layer=1, tolerance=1e-4)
    fidelity = MPSCircuits.evaluate_circuit_fidelity(circuit, mps; cutoff=1e-12)
    @test fidelity > 0.99
end

@testset "Procrustes Complex Environment Conjugation" begin
    # The gate enters the fidelity network linearly (no conjugate), so the maximizer of the linear
    # objective Re(sum(G .* E)) is conj(U*Vt), NOT the standard orthogonal-Procrustes result U*Vt.
    # On a genuinely complex environment the two diverge; only the conjugated form attains the
    # theoretical maximum sum(svdvals(E)). This is a no-op on real environments (regression-safe).
    # Use properly tagged site indices (n=X) so the UnitaryGate constructor can infer site numbers.
    sites = siteinds("S=1/2", 2)
    s1, s2 = sites[1], sites[2]

    # A fixed, genuinely complex 4x4 environment matrix (no RNG dependency for reproducibility).
    Emat = ComplexF64[
        1.0+0.3im    0.2-0.5im   -0.4+0.1im    0.7+0.2im
        -0.3+0.8im   0.9+0.1im    0.5-0.6im   -0.2+0.4im
        0.6-0.2im   -0.7+0.3im    0.8+0.5im    0.1-0.9im
        0.4+0.6im    0.3-0.1im   -0.5+0.7im    1.1-0.4im
    ]

    # Wrap as an ITensor with the exact index ordering new_optimal_gate reads: (s1', s2', s1, s2).
    E = ITensors.itensor(reshape(Emat, 2, 2, 2, 2), ITensors.prime(s1), ITensors.prime(s2), s1, s2)

    gate = MPSCircuits.new_optimal_gate(E, [s1, s2])
    Gmat = reshape(
        ITensors.array(MPSCircuits.tensor(gate), ITensors.prime(s1), ITensors.prime(s2), s1, s2),
        4, 4,
    )

    # Linear fidelity objective: contract the gate into the environment (elementwise on matching indices).
    overlap = sum(Gmat .* Emat)
    max_overlap = sum(LinearAlgebra.svdvals(Emat))

    # Conjugated solution: real, non-negative, and attains the theoretical maximum.
    @test isapprox(imag(overlap), 0.0; atol=1e-10)
    @test isapprox(real(overlap), max_overlap; rtol=1e-8)

    # The unconjugated (buggy) maximizer is strictly worse on this complex environment.
    F = LinearAlgebra.svd(Emat)
    overlap_bug = sum((F.U * F.Vt) .* Emat)
    @test real(overlap_bug) < real(overlap) - 1e-6

    # The returned gate is still unitary.
    @test isapprox(Gmat * Gmat', Matrix(LinearAlgebra.I, 4, 4); atol=1e-10)
end
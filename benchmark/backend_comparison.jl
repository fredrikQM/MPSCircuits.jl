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

#!/usr/bin/env julia

using MPSCircuits
using ITensors
using ITensorMPS
using Printf
using Random
using LinearAlgebra

const HAS_CUDA = try
    @eval using CUDA
    true
catch
    false
end

function maybe_cuda_sync()
    if HAS_CUDA
        CUDA.synchronize()
    end
    return nothing
end

function maybe_cuda_sync(backend::Symbol)
    if HAS_CUDA && backend == :gpu
        CUDA.synchronize()
    end
    return nothing
end

function maybe_cuda_sync(backend::MPSCircuits.ComputeBackend)
    if HAS_CUDA && backend == MPSCircuits.BackendGPU
        CUDA.synchronize()
    end
    return nothing
end

function synced_elapsed(f::Function, backend)
    maybe_cuda_sync(backend)
    t = @elapsed f()
    maybe_cuda_sync(backend)
    return t
end

function has_functional_gpu()
    if !HAS_CUDA
        return false
    end
    return MPSCircuits.resolve_backend(MPSCircuits.BackendConfig(backend=MPSCircuits.BackendAuto)) == MPSCircuits.BackendGPU
end

function parse_chi_targets(arg::String)
    values = Int[]
    for token in split(arg, ',')
        stripped = strip(token)
        isempty(stripped) && continue
        push!(values, parse(Int, stripped))
    end
    isempty(values) && error("No valid χ targets found in input: $(arg)")
    return unique(values)
end

function parse_bool(arg::String)
    lowered = lowercase(strip(arg))
    return lowered in ("1", "true", "yes", "y", "on")
end

function build_fixture_mps_dmrg(N::Int, chi_target::Int)
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
    maxdim_schedule = unique([min(8, chi_target), min(16, chi_target), chi_target])
    cutoff_schedule = fill(1e-10, length(maxdim_schedule))
    _, mps = dmrg(H, psi_init; nsweeps=length(maxdim_schedule), maxdim=maxdim_schedule, cutoff=cutoff_schedule)
    return mps
end

function build_fixture_mps_random(N::Int, chi_target::Int; seed::Int=1234)
    sites = siteinds("S=1/2", N)
    Random.seed!(seed + chi_target)
    mps = random_mps(sites; linkdims=chi_target)
    ITensorMPS.orthogonalize!(mps, 1)
    return mps
end

function build_fixture_mps(N::Int, chi_target::Int; fixture_mode::Symbol=:random, seed::Int=1234)
    if fixture_mode == :random
        return build_fixture_mps_random(N, chi_target; seed=seed)
    elseif fixture_mode == :dmrg
        return build_fixture_mps_dmrg(N, chi_target)
    end
    error("Unknown fixture mode: $(fixture_mode). Expected :random or :dmrg")
end

function time_transfer_pair(mps::ITensorMPS.MPS; precision::Symbol=:fp32)
    if !has_functional_gpu()
        return (NaN, NaN)
    end
    dense = ITensorMPS.dense(mps)

    # Warm up both directions so one-time compilation doesn't pollute timing.
    warm_gpu = MPSCircuits.to_backend(dense, MPSCircuits.BackendGPU; precision=precision)
    maybe_cuda_sync()
    _ = MPSCircuits.to_backend(warm_gpu, MPSCircuits.BackendCPU; precision=:preserve)
    maybe_cuda_sync()

    mps_gpu = dense
    t_to_gpu = synced_elapsed(:gpu) do
        mps_gpu = MPSCircuits.to_backend(dense, MPSCircuits.BackendGPU; precision=precision)
    end
    t_to_cpu = synced_elapsed(:gpu) do
        _ = MPSCircuits.to_backend(mps_gpu, MPSCircuits.BackendCPU; precision=:preserve)
    end
    return (t_to_gpu, t_to_cpu)
end

function _resolve_backend_and_policy(backend::Symbol, gpu_fallback::Bool)
    backend_config = MPSCircuits.BackendConfig(; backend=MPSCircuits.backend_from_symbol(backend), gpu_fallback=gpu_fallback)
    resolved_backend = MPSCircuits.resolve_backend(backend_config)
    resolved_policy = MPSCircuits.transfer_policy_from_symbol(:strict_single_backend)
    return resolved_backend, resolved_policy
end

function _profile_decompose_all(
    mps::ITensorMPS.MPS;
    backend::Symbol,
    gpu_fallback::Bool,
    precision::Symbol,
    n_layers_max::Int,
    tolerance::Float64,
)
    resolved_backend, resolved_policy = _resolve_backend_and_policy(backend, gpu_fallback)

    mps_work = ITensorMPS.dense(mps)
    t_setup_boundary = synced_elapsed(resolved_backend) do
        mps_work = MPSCircuits.place_on_backend_boundary(mps_work, resolved_backend, resolved_policy; precision=precision)
    end

    preparation_circuit = MPSCircuits.UnitaryGate[]
    t_generate_layers = 0.0
    t_layer_boundary = 0.0
    t_single_qubit = 0.0

    for layer in 1:n_layers_max
        entangling_layer = nothing
        flag_disentangled = false
        t_generate_layers += synced_elapsed(resolved_backend) do
            mps_work, entangling_layer, flag_disentangled = MPSCircuits.generate_layer!(
                mps_work;
                tolerance=tolerance,
                layer_number=layer,
                max_bond_dim=ITensorMPS.maxlinkdim(mps),
                working_cutoff=1e-12,
                backend=resolved_backend,
            )
        end
        if flag_disentangled
            break
        end
        t_layer_boundary += synced_elapsed(resolved_backend) do
            entangling_layer = MPSCircuits.place_on_backend_boundary(entangling_layer, resolved_backend, resolved_policy; precision=precision)
            prepend!(preparation_circuit, entangling_layer)
        end
    end

    t_single_qubit += synced_elapsed(resolved_backend) do
        MPSCircuits.apply_single_qubit_rotations!(preparation_circuit, mps_work)
    end

    phase = (
        setup_boundary_s=t_setup_boundary,
        layer_extract_s=t_generate_layers,
        layer_boundary_s=t_layer_boundary,
        optimization_s=0.0,
        single_qubit_s=t_single_qubit,
        optimization_gate_updates=0,
    )
    return preparation_circuit, phase
end

function _profile_iterative_opt_all(
    mps::ITensorMPS.MPS;
    backend::Symbol,
    gpu_fallback::Bool,
    precision::Symbol,
    n_layers_max::Int,
    n_iterations_per_layer::Int,
    tolerance::Float64,
)
    resolved_backend, resolved_policy = _resolve_backend_and_policy(backend, gpu_fallback)

    mps_clean = ITensorMPS.dense(mps)
    t_setup_boundary = synced_elapsed(resolved_backend) do
        mps_clean = MPSCircuits.place_on_backend_boundary(mps_clean, resolved_backend, resolved_policy; precision=precision)
    end
    mps_work = deepcopy(mps_clean)
    preparation_circuit = MPSCircuits.UnitaryGate[]

    t_generate_layers = 0.0
    t_layer_boundary = 0.0
    t_optimization = 0.0
    t_single_qubit = 0.0
    optimization_gate_updates = 0

    for layer in 1:n_layers_max
        entangling_layer = nothing
        flag_disentangled = false
        t_generate_layers += synced_elapsed(resolved_backend) do
            mps_work, entangling_layer, flag_disentangled = MPSCircuits.generate_layer!(
                mps_work;
                tolerance=tolerance,
                layer_number=layer,
                max_bond_dim=2 * ITensorMPS.maxlinkdim(mps),
                working_cutoff=1e-12,
                backend=resolved_backend,
            )
        end

        if flag_disentangled
            t_single_qubit += synced_elapsed(resolved_backend) do
                MPSCircuits.apply_single_qubit_rotations!(preparation_circuit, mps_work)
            end
            break
        end

        t_layer_boundary += synced_elapsed(resolved_backend) do
            entangling_layer = MPSCircuits.place_on_backend_boundary(entangling_layer, resolved_backend, resolved_policy; precision=precision)
            prepend!(preparation_circuit, entangling_layer)
        end

        t_single_qubit += synced_elapsed(resolved_backend) do
            MPSCircuits.apply_single_qubit_rotations!(preparation_circuit, mps_work)
        end

        t_layer_boundary += synced_elapsed(resolved_backend) do
            preparation_circuit = MPSCircuits.place_on_backend_boundary(preparation_circuit, resolved_backend, resolved_policy; precision=precision)
        end

        t_optimization += synced_elapsed(resolved_backend) do
            for _ in 1:n_iterations_per_layer
                MPSCircuits.replace_gates!(
                    mps_clean,
                    preparation_circuit,
                    MPSCircuits.TelescopingEnvironment(),
                    1:length(preparation_circuit);
                    max_bond_dim=2 * ITensorMPS.maxlinkdim(mps),
                    working_cutoff=1e-12,
                    backend=resolved_backend,
                )
            end
            optimization_gate_updates += n_iterations_per_layer * length(preparation_circuit)
        end
    end

    phase = (
        setup_boundary_s=t_setup_boundary,
        layer_extract_s=t_generate_layers,
        layer_boundary_s=t_layer_boundary,
        optimization_s=t_optimization,
        single_qubit_s=t_single_qubit,
        optimization_gate_updates=optimization_gate_updates,
    )
    return preparation_circuit, phase
end

function _probe_tiny_kernel_breakdown(
    mps::ITensorMPS.MPS,
    circuit::Vector{<:MPSCircuits.AbstractGate};
    repeats::Int=400,
)
    if isempty(circuit)
        return (tiny_extract_s=NaN, tiny_transfer_s=NaN, tiny_cpu_svd_s=NaN)
    end

    gate_index = 1
    environment, _, _ = MPSCircuits.environment_tensor(
        mps,
        convert(Vector{MPSCircuits.UnitaryGate}, circuit),
        MPSCircuits.TelescopingEnvironment();
        gate_index=gate_index,
        max_bond_dim=2 * ITensorMPS.maxlinkdim(mps),
        working_cutoff=1e-12,
    )

    site_indices = circuit[gate_index].site_indices
    i1 = site_indices[1]
    i2 = site_indices[2]

    env_array = ITensors.array(environment, ITensors.prime(i1), ITensors.prime(i2), i1, i2)
    t_extract = synced_elapsed(:gpu) do
        for _ in 1:repeats
            _ = ITensors.array(environment, ITensors.prime(i1), ITensors.prime(i2), i1, i2)
        end
    end

    env_cpu = MPSCircuits.to_tiny_kernel_cpu(env_array)
    t_transfer = synced_elapsed(:gpu) do
        for _ in 1:repeats
            _ = MPSCircuits.to_tiny_kernel_cpu(env_array)
        end
    end

    env_matrix = reshape(env_cpu, 4, 4)
    t_cpu_svd = @elapsed begin
        for _ in 1:repeats
            _ = LinearAlgebra.svd(env_matrix)
        end
    end

    return (
        tiny_extract_s=t_extract / repeats,
        tiny_transfer_s=t_transfer / repeats,
        tiny_cpu_svd_s=t_cpu_svd / repeats,
    )
end

function run_case(
    mps::ITensorMPS.MPS;
    protocol::Symbol,
    label::String,
    chi_target::Int,
    chi_actual::Int,
    backend::Symbol,
    precision::Symbol,
    gpu_fallback::Bool,
    n_layers_max::Int,
    n_iterations_per_layer::Int,
    tolerance::Float64,
    phase_profile::Bool,
    warmup::Bool,
)
    kwargs = (
        n_layers_max=n_layers_max,
        tolerance=tolerance,
        backend=backend,
        gpu_fallback=gpu_fallback,
        precision=precision,
        transfer_policy=:strict_single_backend,
    )

    function compile_once()
        if protocol == :decompose_all
            return MPSCircuits.compile_mps_circuit(mps, MPSCircuits.DecomposeAllAnalytical(); kwargs...)
        elseif protocol == :iterative_opt_all
            return MPSCircuits.compile_mps_circuit(
                mps,
                MPSCircuits.IterativeDecomposeOptimizeAll();
                optimization_protocol=MPSCircuits.TelescopingEnvironment(),
                n_iterations_per_layer=n_iterations_per_layer,
                kwargs...,
            )
        end
        error("Unknown protocol: $(protocol)")
    end

    function compile_with_profile()
        if protocol == :decompose_all
            return _profile_decompose_all(
                mps;
                backend=backend,
                gpu_fallback=gpu_fallback,
                precision=precision,
                n_layers_max=n_layers_max,
                tolerance=tolerance,
            )
        elseif protocol == :iterative_opt_all
            return _profile_iterative_opt_all(
                mps;
                backend=backend,
                gpu_fallback=gpu_fallback,
                precision=precision,
                n_layers_max=n_layers_max,
                n_iterations_per_layer=n_iterations_per_layer,
                tolerance=tolerance,
            )
        end
        error("Unknown protocol: $(protocol)")
    end

    if warmup
        _ = compile_once()
        maybe_cuda_sync(backend)
    end

    if phase_profile
        circuit = MPSCircuits.UnitaryGate[]
        phase = (
            setup_boundary_s=NaN,
            layer_extract_s=NaN,
            layer_boundary_s=NaN,
            optimization_s=NaN,
            single_qubit_s=NaN,
            optimization_gate_updates=0,
        )
        t_compile = synced_elapsed(backend) do
            circuit, phase = compile_with_profile()
        end
    else
        t_compile = synced_elapsed(backend) do
            circuit = compile_once()
        end
        phase = (
            setup_boundary_s=NaN,
            layer_extract_s=NaN,
            layer_boundary_s=NaN,
            optimization_s=NaN,
            single_qubit_s=NaN,
            optimization_gate_updates=0,
        )
    end

    # Evaluate fidelity on CPU to avoid mixed-device policy artifacts in this benchmark.
    circuit_cpu = MPSCircuits.to_backend(circuit, MPSCircuits.BackendCPU; precision=:preserve)
    t_fidelity = @elapsed fidelity = MPSCircuits.evaluate_circuit_fidelity(circuit_cpu, mps; cutoff=1e-12)

    tiny_probe = (tiny_extract_s=NaN, tiny_transfer_s=NaN, tiny_cpu_svd_s=NaN)
    if phase_profile && protocol == :iterative_opt_all && backend == :gpu
        mps_gpu = MPSCircuits.to_backend(ITensorMPS.dense(mps), MPSCircuits.BackendGPU; precision=precision)
        circuit_gpu = MPSCircuits.to_backend(circuit, MPSCircuits.BackendGPU; precision=:preserve)
        tiny_probe = _probe_tiny_kernel_breakdown(mps_gpu, circuit_gpu)
    end

    est_tiny_extract_total_s = NaN
    est_tiny_transfer_total_s = NaN
    est_tiny_cpu_svd_total_s = NaN
    est_tiny_total_s = NaN
    if phase_profile && protocol == :iterative_opt_all && backend == :gpu && phase.optimization_gate_updates > 0 && !isnan(tiny_probe.tiny_transfer_s)
        est_tiny_extract_total_s = tiny_probe.tiny_extract_s * phase.optimization_gate_updates
        est_tiny_transfer_total_s = tiny_probe.tiny_transfer_s * phase.optimization_gate_updates
        est_tiny_cpu_svd_total_s = tiny_probe.tiny_cpu_svd_s * phase.optimization_gate_updates
        est_tiny_total_s = est_tiny_extract_total_s + est_tiny_transfer_total_s + est_tiny_cpu_svd_total_s
    end

    return (
        label=label,
        protocol=protocol,
        chi_target=chi_target,
        chi_actual=chi_actual,
        backend=backend,
        precision=precision,
        compile_s=t_compile,
        fidelity_s=t_fidelity,
        fidelity=fidelity,
        n_gates=length(circuit),
        phase_setup_boundary_s=phase.setup_boundary_s,
        phase_layer_extract_s=phase.layer_extract_s,
        phase_layer_boundary_s=phase.layer_boundary_s,
        phase_optimization_s=phase.optimization_s,
        phase_single_qubit_s=phase.single_qubit_s,
        phase_optimization_gate_updates=phase.optimization_gate_updates,
        tiny_extract_s=tiny_probe.tiny_extract_s,
        tiny_transfer_s=tiny_probe.tiny_transfer_s,
        tiny_cpu_svd_s=tiny_probe.tiny_cpu_svd_s,
        est_tiny_extract_total_s=est_tiny_extract_total_s,
        est_tiny_transfer_total_s=est_tiny_transfer_total_s,
        est_tiny_cpu_svd_total_s=est_tiny_cpu_svd_total_s,
        est_tiny_total_s=est_tiny_total_s,
    )
end

function print_results_table(rows)
    println("\nBenchmark results")
    println("protocol          label                 χ_target  χ_actual  backend  precision  status    compile_s    fidelity_s   fidelity      n_gates")
    println("-----------------------------------------------------------------------------------------------------------------------------------------")
    for r in rows
        if get(r, :ok, true)
            @printf("%-16s  %-20s  %8d  %8d  %-7s  %-9s  %-7s  %10.4f  %10.4f  %11.6f  %7d\n",
                String(r.protocol),
                r.label,
                r.chi_target,
                r.chi_actual,
                String(r.backend),
                String(r.precision),
                "ok",
                r.compile_s,
                r.fidelity_s,
                r.fidelity,
                r.n_gates,
            )
        else
            @printf("%-16s  %-20s  %8d  %8d  %-7s  %-9s  %-7s  %10s  %10s  %11s  %7s\n",
                String(get(r, :protocol, :unknown)),
                r.label,
                get(r, :chi_target, -1),
                get(r, :chi_actual, -1),
                String(r.backend),
                String(r.precision),
                "error",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
            )
            println("  error: ", r.error)
        end
    end
end

function print_phase_summary(rows)
    println("\nPhase breakdown (seconds)")
    println("protocol          label                 χ_actual  setup      layers     boundary   optimize   singleq")
    println("-----------------------------------------------------------------------------------------------")
    for r in rows
        if get(r, :ok, true) && !isnan(get(r, :phase_setup_boundary_s, NaN))
            @printf("%-16s  %-20s  %8d  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f\n",
                String(r.protocol),
                r.label,
                r.chi_actual,
                r.phase_setup_boundary_s,
                r.phase_layer_extract_s,
                r.phase_layer_boundary_s,
                r.phase_optimization_s,
                r.phase_single_qubit_s,
            )
        end
    end

    println("\nTiny-kernel probe (average per call, seconds)")
    println("protocol          label                 χ_actual  array_extract  gpu_to_cpu     cpu_svd")
    println("---------------------------------------------------------------------------------------")
    for r in rows
        if get(r, :ok, true) && !isnan(get(r, :tiny_transfer_s, NaN))
            @printf("%-16s  %-20s  %8d  %12.6e  %10.6e  %10.6e\n",
                String(r.protocol),
                r.label,
                r.chi_actual,
                r.tiny_extract_s,
                r.tiny_transfer_s,
                r.tiny_cpu_svd_s,
            )
        end
    end

    println("\nEstimated tiny-kernel totals inside optimization")
    println("protocol          label                 χ_actual  gate_updates   est_extract   est_transfer    est_cpu_svd     est_total")
    println("-----------------------------------------------------------------------------------------------------------------------")
    for r in rows
        if get(r, :ok, true) && !isnan(get(r, :est_tiny_total_s, NaN))
            @printf("%-16s  %-20s  %8d  %11d  %11.4f  %13.4f  %13.4f  %11.4f\n",
                String(r.protocol),
                r.label,
                r.chi_actual,
                r.phase_optimization_gate_updates,
                r.est_tiny_extract_total_s,
                r.est_tiny_transfer_total_s,
                r.est_tiny_cpu_svd_total_s,
                r.est_tiny_total_s,
            )
        end
    end
end

function safe_run_case(rows, mps; kwargs...)
    try
        push!(rows, run_case(mps; kwargs...))
    catch err
        push!(rows, (
            protocol=kwargs[:protocol],
            label=kwargs[:label],
            chi_target=kwargs[:chi_target],
            chi_actual=kwargs[:chi_actual],
            backend=kwargs[:backend],
            precision=kwargs[:precision],
            ok=false,
            error=sprint(showerror, err),
        ))
    end
end

function main()
    N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 16
    n_layers_max = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    tolerance = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1e-8
    chi_targets = length(ARGS) >= 4 ? parse_chi_targets(ARGS[4]) : [20, 40, 80]
    fixture_mode = length(ARGS) >= 5 ? Symbol(strip(lowercase(ARGS[5]))) : :random
    n_iterations_per_layer = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 3
    phase_profile = length(ARGS) >= 7 ? parse_bool(ARGS[7]) : true
    protocols = [:decompose_all, :iterative_opt_all]

    println("Preparing fixture MPS set (mode=$(fixture_mode), N=$(N), χ targets=$(chi_targets))...")
    println("Protocols: $(protocols), iterative n_iterations_per_layer=$(n_iterations_per_layer), phase_profile=$(phase_profile)")

    rows = NamedTuple[]

    for chi_target in chi_targets
        println("\nPreparing fixture for χ_target=$(chi_target)...")
        mps = build_fixture_mps(N, chi_target; fixture_mode=fixture_mode)
        chi_actual = ITensorMPS.maxlinkdim(mps)
        println("Actual χ=$(chi_actual)")

        for protocol in protocols
            safe_run_case(rows, mps;
                protocol=protocol,
                label="cpu-fp64",
                chi_target=chi_target,
                chi_actual=chi_actual,
                backend=:cpu,
                precision=:fp64,
                gpu_fallback=true,
                n_layers_max=n_layers_max,
                n_iterations_per_layer=n_iterations_per_layer,
                tolerance=tolerance,
                phase_profile=phase_profile,
                warmup=true,
            )

            safe_run_case(rows, mps;
                protocol=protocol,
                label="cpu-fp32",
                chi_target=chi_target,
                chi_actual=chi_actual,
                backend=:cpu,
                precision=:fp32,
                gpu_fallback=true,
                n_layers_max=n_layers_max,
                n_iterations_per_layer=n_iterations_per_layer,
                tolerance=tolerance,
                phase_profile=phase_profile,
                warmup=true,
            )

            if has_functional_gpu()
                safe_run_case(rows, mps;
                    protocol=protocol,
                    label="gpu-fp32",
                    chi_target=chi_target,
                    chi_actual=chi_actual,
                    backend=:gpu,
                    precision=:fp32,
                    gpu_fallback=false,
                    n_layers_max=n_layers_max,
                    n_iterations_per_layer=n_iterations_per_layer,
                    tolerance=tolerance,
                    phase_profile=phase_profile,
                    warmup=true,
                )

                safe_run_case(rows, mps;
                    protocol=protocol,
                    label="gpu-fp64",
                    chi_target=chi_target,
                    chi_actual=chi_actual,
                    backend=:gpu,
                    precision=:fp64,
                    gpu_fallback=false,
                    n_layers_max=n_layers_max,
                    n_iterations_per_layer=n_iterations_per_layer,
                    tolerance=tolerance,
                    phase_profile=phase_profile,
                    warmup=true,
                )
            else
                println("CUDA not functional: GPU rows skipped for χ_target=$(chi_target), protocol=$(protocol).")
            end
        end

        if has_functional_gpu()
            t_gpu, t_cpu = time_transfer_pair(mps; precision=:fp32)
            @printf("Transfer timing for χ=%d (dense MPS, fp32): to_gpu=%.4fs to_cpu=%.4fs\n", chi_actual, t_gpu, t_cpu)
        end
    end

    print_results_table(rows)
    if phase_profile
        print_phase_summary(rows)
    end
end

main()

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

abstract type AbstractProcrustesProtocol end
struct FullRebuild <: AbstractProcrustesProtocol end
struct RollingEnvironment <: AbstractProcrustesProtocol end
struct TelescopingEnvironment <: AbstractProcrustesProtocol end

@inline function _optional_env_norm(environment::ITensors.ITensor, progress::AbstractProgressTracker)
    if !progress_include_env_norm(progress)
        return NaN
    end
    arr = ITensors.array(environment, ITensors.inds(environment)...)
    arr_cpu = to_tiny_kernel_cpu(arr)
    return Float64(LinearAlgebra.norm(arr_cpu))
end

function _mps_precision_symbol(mps::ITensorMPS.MPS)
    if length(mps) == 0
        return :preserve
    end
    inds = ITensors.inds(mps[1])
    dense_array = ITensors.array(mps[1], inds...)
    T = eltype(dense_array)
    if T <: ComplexF32 || T <: Float32
        return :fp32
    elseif T <: ComplexF64 || T <: Float64
        return :fp64
    end
    return :preserve
end

"""
    partial_inner(mps_bra::ITensorMPS.MPS, mps_ket::ITensorMPS.MPS, site1::Int, site2::Int)

Compute the partial inner product between `mps_bra` and `mps_ket` by contracting all sites except `site1` and `site2`.
Convenience function for computing environment tensors.
"""
function partial_inner(mps_bra::ITensorMPS.MPS, mps_ket::ITensorMPS.MPS, site1::Int, site2::Int)
    N = length(mps_bra)
    bra = copy(mps_bra) # Create a shallow copy of the bra array so we don't mutate the original MPS
    # Get the specific physical indices at the target sites
    s1 = ITensorMPS.siteind(bra, site1)
    s2 = ITensorMPS.siteind(bra, site2)
    # Prime to prevent contractions with the other mPS
    bra[site1] = ITensors.prime(bra[site1], s1)
    bra[site2] = ITensors.prime(bra[site2], s2)
    # Sweep from left to right to build the overlap network.
    # Initialize from the first site pair to avoid constructing a default Float64 scalar tensor.
    E = (mps_ket[1]) * ITensors.dag(bra[1])
    for k in 2:N
        E = (E * mps_ket[k]) * ITensors.dag(bra[k]) # Parentheses dicate contraction order, ensures we build a χ×χ environment matrix at each step.
    end
    return E
end


"""
    environment_tensor(mps::ITensorMPS.MPS, circuit::Vector{UnitaryGate}; gate_index::Int)

Compute the environment tensor for the gate at `gate_index` in `circuit`: this is the fidelity's tensor network, but with said gate's indices left open.
This object describes the possible effects of all possible gates at that location on fidelity.
This procedure fully rebuilds the environment from scratch, as opposed to reusing the partial contractions.
It returns the ket and bra 'peephole', so can be used to initialize the efficient-reuse method.
"""
function environment_tensor(
    mps::ITensorMPS.MPS,
    circuit::Vector{UnitaryGate},
    protocol::FullRebuild;
    gate_index::Int,
    max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(mps),
    working_cutoff::Float64=1e-12
)
    if gate_index < 1 || gate_index > length(circuit)
        error("gate_index must be between 1 and $(length(circuit)); got $(gate_index).")
    end
    sites = ITensorMPS.siteinds(mps)
    # take the start of circuit, apply it to zero state (ket)
    initial_state = ["0" for _ in 1:length(sites)]
    ket = ITensorMPS.MPS(sites, initial_state)
    ket = to_backend(ket, backend_of(mps); precision=_mps_precision_symbol(mps))
    for gate in circuit[1:(gate_index-1)]
        ket = apply_gate(gate, ket; mixed_device=:coerce, conversion_precision=:preserve, cutoff=working_cutoff, maxdim=max_bond_dim)
    end

    # take the end of circuit, apply it left to working MPS (bra)
    bra = deepcopy(mps)
    for gate in circuit[end:-1:(gate_index+1)]
        bra = apply_gate(dagger(gate), bra; mixed_device=:coerce, conversion_precision=:preserve, cutoff=working_cutoff, maxdim=max_bond_dim)
    end

    environment = partial_inner(bra, ket, circuit[gate_index].site_numbers...)
    return environment, ket, bra
end

"""
    next_environment!(circuit::Vector{UnitaryGate}, next_position::Int, ket::ITensorMPS.MPS, bra::ITensorMPS.MPS, protocol::RollingEnvironment; max_bond_dim::Int, working_cutoff::Float64=1e-12)

Replace the gates at `gate_indices` in `circuit` with new optimal gates computed from their environments, i.e. Procrustes optimization.
This is the "telescoping environment" protocol where we store partial contractions in memory. It saves on contractions and should match FullRebuild in accuracy, but uses much more memory.
"""
function environment_tensor(
    mps::ITensorMPS.MPS,
    circuit::Vector{UnitaryGate},
    protocol::TelescopingEnvironment;
    gate_index::Int,
    max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(mps),
    working_cutoff::Float64=1e-12
)
    if gate_index < 1 || gate_index > length(circuit)
        error("gate_index must be between 1 and $(length(circuit)); got $(gate_index).")
    end
    sites = ITensorMPS.siteinds(mps)
    # take the start of circuit, apply it to zero state (ket)
    initial_state = ["0" for _ in 1:length(sites)]
    ket = ITensorMPS.MPS(sites, initial_state)
    ket = to_backend(ket, backend_of(mps); precision=_mps_precision_symbol(mps))
    for gate in circuit[1:(gate_index-1)]
        ket = apply_gate(gate, ket; mixed_device=:coerce, conversion_precision=:preserve, cutoff=working_cutoff, maxdim=max_bond_dim)
    end

    # take the end of circuit, apply it left to working MPS (bra). store each partial contraction for telescoping reuse
    cached_bras = ITensorMPS.MPS[]
    current_bra = deepcopy(mps)
    push!(cached_bras, current_bra)
    for gate in circuit[end:-1:(gate_index+1)]
        current_bra = apply_gate(dagger(gate), current_bra; mixed_device=:coerce, conversion_precision=:preserve, cutoff=working_cutoff, maxdim=max_bond_dim)
        push!(cached_bras, current_bra)
    end

    environment = partial_inner(current_bra, ket, circuit[gate_index].site_numbers...)
    return environment, ket, cached_bras
end

"""
    next_environment!(circuit::Vector{UnitaryGate}, next_position::Int, ket::ITensorMPS.MPS, bra::ITensorMPS.MPS, protocol::RollingEnvironment; max_bond_dim::Int, working_cutoff::Float64=1e-12)

Given the current environment for `circuit[next_position-1]`, compute the next environment for `circuit[next_position]` by applying the appropriate gates to the ket and bra.
This is used for the rolling environment protocol, where we reuse the partially contracted ket and bra from the previous step to avoid redundant contractions. 
"""
function next_environment!(
    circuit::Vector{UnitaryGate},
    next_position::Int,
    ket::ITensorMPS.MPS,
    bra::ITensorMPS.MPS,
    protocol::RollingEnvironment;
    max_bond_dim::Int,
    working_cutoff::Float64=1e-12
)
    ket = apply_gate(circuit[next_position-1], ket; mixed_device=:coerce, conversion_precision=:preserve, cutoff=working_cutoff, maxdim=max_bond_dim)
    bra = apply_gate(circuit[next_position], bra; mixed_device=:coerce, conversion_precision=:preserve, cutoff=working_cutoff, maxdim=max_bond_dim)
    environment = partial_inner(bra, ket, circuit[next_position].site_numbers...)
    return environment, ket, bra
end

"""
    next_environment!(circuit::Vector{UnitaryGate}, next_position::Int, ket::ITensorMPS.MPS, next_bra::ITensorMPS.MPS, protocol::TelescopingEnvironment; max_bond_dim::Int, working_cutoff::Float64=1e-12)

Given the current environment for `circuit[next_position-1]`, compute the next environment for `circuit[next_position]`.
This is used for the telescoping environment protocol, where we reuse the partially contracted ket and update it, but re-use cached bras.
"""
function next_environment!(
    circuit::Vector{UnitaryGate},
    next_position::Int,
    ket::ITensorMPS.MPS,
    next_bra::ITensorMPS.MPS,
    protocol::TelescopingEnvironment;
    max_bond_dim::Int,
    working_cutoff::Float64=1e-12
)
    ket = apply_gate(circuit[next_position-1], ket; mixed_device=:coerce, conversion_precision=:preserve, cutoff=working_cutoff, maxdim=max_bond_dim)
    environment = partial_inner(next_bra, ket, circuit[next_position].site_numbers...)
    return environment, ket
end

"""
    new_optimal_gate(environment::ITensors.ITensor, site_indices::Vector{<:ITensors.Index})

Given an environment tensor for a 2-site gate, compute the optimal gate by performing an SVD (Procrustes problem).
"""
function new_optimal_gate(
    environment::ITensors.ITensor,
    site_indices::Vector{<:ITensors.Index}
)
    # Perform SVD and make sure to get the indices right
    environment_array = ITensors.array(environment, ITensors.prime(site_indices[1]), ITensors.prime(site_indices[2]), site_indices[1], site_indices[2])
    environment_array = to_tiny_kernel_cpu(environment_array)
    environment_matrix = reshape(environment_array, 4, 4)
    F = LinearAlgebra.svd(environment_matrix)
    gate_array = reshape(conj(F.U * F.Vt), 2, 2, 2, 2) # conj necessary for complex-valued environments, due to E^T convention

    new_gate = ITensors.itensor(gate_array, ITensors.prime(site_indices[1]), ITensors.prime(site_indices[2]), site_indices[1], site_indices[2])
    return UnitaryGate(new_gate)
end


"""
    replace_gates!(mps::ITensorMPS.MPS, circuit::Vector{UnitaryGate}; protocol::AbstractProcrustesProtocol, gate_indices::AbstractVector{<:Integer}, max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(mps), working_cutoff::Float64=1e-12)

Replace the gates at `gate_indices` in `circuit` with new optimal gates computed from their environments, i.e. Procrustes optimization.
This is the "full rebuild" protocol where we build a new environment at each step. It's computationally expensive and best used for testing/verification.
"""
function replace_gates!(
    mps::ITensorMPS.MPS,
    circuit::Vector{UnitaryGate},
    protocol::FullRebuild,
    gate_indices::AbstractVector{<:Integer};
    max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(mps),
    working_cutoff::Float64=1e-12,
    backend::ComputeBackend=BackendCPU,
    progress::AbstractProgressTracker=NoProgressTracker(),
    layer_number::Int=0,
    iteration_index::Int=1,
    n_iterations_total::Int=1,
)
    gate_count = length(gate_indices)
    gate_step_in_iteration = 0
    for gate_index in gate_indices
        gate_step_in_iteration += 1
        environment, ket, bra = environment_tensor(mps, circuit, FullRebuild(); gate_index=gate_index, max_bond_dim=max_bond_dim, working_cutoff=working_cutoff)
        new_gate = new_optimal_gate(environment, circuit[gate_index].site_indices)
        new_gate = to_backend(new_gate, backend_of(mps); precision=:preserve)
        circuit[gate_index] = new_gate
        record_progress_opt_step!(progress; layer=layer_number, iter_in_layer=iteration_index, iter_total_for_layer=n_iterations_total, gate_index=gate_index, gate_count=gate_count, gate_step_in_iteration=gate_step_in_iteration, ket_maxlinkdim=ITensorMPS.maxlinkdim(ket), bra_maxlinkdim=ITensorMPS.maxlinkdim(bra), env_norm=_optional_env_norm(environment, progress))
    end
    return nothing
end

"""
    replace_gates!(mps::ITensorMPS.MPS, circuit::Vector{UnitaryGate}; protocol::RollingEnvironment, gate_indices::AbstractVector{<:Integer}, max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(mps), working_cutoff::Float64=1e-12)

Replace the gates at `gate_indices` in `circuit` with new optimal gates computed from their environments, i.e. Procrustes optimization.
This is the "rolling environment" protocol where we reuse the partially contracted ket and bra from the previous step to avoid redundant contractions.
It's more memory-efficient than telescoping environments, but slightly less accurate and slightly less efficient.
"""
function replace_gates!(
    mps::ITensorMPS.MPS,
    circuit::Vector{UnitaryGate},
    protocol::RollingEnvironment,
    gate_indices::AbstractVector{<:Integer};
    max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(mps),
    working_cutoff::Float64=1e-12,
    backend::ComputeBackend=BackendCPU,
    progress::AbstractProgressTracker=NoProgressTracker(),
    layer_number::Int=0,
    iteration_index::Int=1,
    n_iterations_total::Int=1,
)
    if length(gate_indices) > 1 && any(diff(gate_indices) .!= 1)
        error("gate_indices must be a list of consecutive integers, got: $(gate_indices)")
    end
    gate_count = length(gate_indices)
    gate_step_in_iteration = 1
    # Start at first gate. NB: we use FullRebuild for code reuse / convenience, since this -initializes- it.
    environment, ket, bra = environment_tensor(mps, circuit, FullRebuild(); gate_index=gate_indices[1], max_bond_dim=max_bond_dim, working_cutoff=working_cutoff)
    circuit[gate_indices[1]] = to_backend(new_optimal_gate(environment, circuit[gate_indices[1]].site_indices), backend_of(mps); precision=:preserve)
    record_progress_opt_step!(progress; layer=layer_number, iter_in_layer=iteration_index, iter_total_for_layer=n_iterations_total, gate_index=gate_indices[1], gate_count=gate_count, gate_step_in_iteration=gate_step_in_iteration, ket_maxlinkdim=ITensorMPS.maxlinkdim(ket), bra_maxlinkdim=ITensorMPS.maxlinkdim(bra), env_norm=_optional_env_norm(environment, progress))
    # Now iterate over remaining gates
    for gate_index in gate_indices[2:end]
        gate_step_in_iteration += 1
        environment, ket, bra = next_environment!(circuit, gate_index, ket, bra, RollingEnvironment(); max_bond_dim=max_bond_dim, working_cutoff=working_cutoff)
        new_gate = new_optimal_gate(environment, circuit[gate_index].site_indices)
        new_gate = to_backend(new_gate, backend_of(mps); precision=:preserve)
        circuit[gate_index] = new_gate
        record_progress_opt_step!(progress; layer=layer_number, iter_in_layer=iteration_index, iter_total_for_layer=n_iterations_total, gate_index=gate_index, gate_count=gate_count, gate_step_in_iteration=gate_step_in_iteration, ket_maxlinkdim=ITensorMPS.maxlinkdim(ket), bra_maxlinkdim=ITensorMPS.maxlinkdim(bra), env_norm=_optional_env_norm(environment, progress))
    end
    return nothing
end


"""
    replace_gates!(mps::ITensorMPS.MPS, circuit::Vector{UnitaryGate}; protocol::TelescopingEnvironment, gate_indices::AbstractVector{<:Integer}, max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(mps), working_cutoff::Float64=1e-12)

Replace the gates at `gate_indices` in `circuit` with new optimal gates computed from their environments, i.e. Procrustes optimization.
This is the "telescoping environment" protocol where we reuse the partially contracted ket and update it, but re-use cached bras.
It saves on contractions and should match FullRebuild in accuracy, but uses much more memory.
"""
function replace_gates!(
    mps::ITensorMPS.MPS,
    circuit::Vector{UnitaryGate},
    protocol::TelescopingEnvironment,
    gate_indices::AbstractVector{<:Integer};
    max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(mps),
    working_cutoff::Float64=1e-12,
    backend::ComputeBackend=BackendCPU,
    progress::AbstractProgressTracker=NoProgressTracker(),
    layer_number::Int=0,
    iteration_index::Int=1,
    n_iterations_total::Int=1,
)
    if length(gate_indices) > 1 && any(diff(gate_indices) .!= 1)
        error("gate_indices must be a list of consecutive integers, got: $(gate_indices)")
    end
    gate_count = length(gate_indices)
    gate_step_in_iteration = 1
    # Start at first gate. This also collects the telescoping list of cached bras.
    environment, ket, cached_bras = environment_tensor(mps, circuit, TelescopingEnvironment(); gate_index=gate_indices[1], max_bond_dim=max_bond_dim, working_cutoff=working_cutoff)
    circuit[gate_indices[1]] = to_backend(new_optimal_gate(environment, circuit[gate_indices[1]].site_indices), backend_of(mps); precision=:preserve)
    record_progress_opt_step!(progress; layer=layer_number, iter_in_layer=iteration_index, iter_total_for_layer=n_iterations_total, gate_index=gate_indices[1], gate_count=gate_count, gate_step_in_iteration=gate_step_in_iteration, ket_maxlinkdim=ITensorMPS.maxlinkdim(ket), bra_maxlinkdim=ITensorMPS.maxlinkdim(cached_bras[length(circuit)-gate_indices[1]+1]), env_norm=_optional_env_norm(environment, progress))
    # Now iterate over remaining gates. cached_bras are stored in contraction order:
    # cached_bras[k] corresponds to applying daggers of circuit[end], ..., circuit[end-k+2] to mps.
    for gate_index in gate_indices[2:end]
        gate_step_in_iteration += 1
        bra_cache_index = length(circuit) - gate_index + 1
        next_bra = cached_bras[bra_cache_index]
        environment, ket = next_environment!(circuit, gate_index, ket, next_bra, TelescopingEnvironment(); max_bond_dim=max_bond_dim, working_cutoff=working_cutoff)
        new_gate = new_optimal_gate(environment, circuit[gate_index].site_indices)
        new_gate = to_backend(new_gate, backend_of(mps); precision=:preserve)
        circuit[gate_index] = new_gate
        record_progress_opt_step!(progress; layer=layer_number, iter_in_layer=iteration_index, iter_total_for_layer=n_iterations_total, gate_index=gate_index, gate_count=gate_count, gate_step_in_iteration=gate_step_in_iteration, ket_maxlinkdim=ITensorMPS.maxlinkdim(ket), bra_maxlinkdim=ITensorMPS.maxlinkdim(next_bra), env_norm=_optional_env_norm(environment, progress))
    end
    return nothing
end
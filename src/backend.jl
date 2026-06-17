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

@enum ComputeBackend begin
    BackendAuto
    BackendCPU
    BackendGPU
end

Base.@kwdef struct BackendConfig
    backend::ComputeBackend = BackendAuto
    gpu_fallback::Bool = true
end

@enum BackendTransferPolicy begin
    PreferGPUResidency
    StrictSingleBackend
end

function backend_from_symbol(backend::Symbol)
    if backend == :auto
        return BackendAuto
    elseif backend == :cpu
        return BackendCPU
    elseif backend == :gpu
        return BackendGPU
    end
    throw(ArgumentError("Unknown backend: $(backend). Expected one of :auto, :cpu, :gpu."))
end

function transfer_policy_from_symbol(policy::Symbol)
    if policy == :prefer_gpu_residency
        return PreferGPUResidency
    elseif policy == :strict_single_backend
        return StrictSingleBackend
    end
    throw(ArgumentError("Unknown transfer policy: $(policy). Expected one of :prefer_gpu_residency, :strict_single_backend."))
end

# Extension hook: default CPU-only behavior. CUDA extension provides helper APIs.
_is_gpu_array_impl(::Any) = false

function _cuda_is_functional()
    ext = Base.get_extension(MPSCircuits, :MPSCircuitsCUDA)
    if ext !== nothing && isdefined(ext, :cuda_is_functional)
        return getfield(ext, :cuda_is_functional)()
    end
    return false
end

is_gpu_array(array_data) = _is_gpu_array_impl(array_data)

"""
    to_tiny_kernel_cpu(array_data::AbstractArray)

Materialize tiny dense arrays on CPU for fixed-size kernels where GPU launch and
transfer overhead usually dominates arithmetic cost.
"""
function to_tiny_kernel_cpu(array_data::AbstractArray)
    return to_backend(array_data, BackendCPU; precision=:preserve)
end

function backend_of(tensor_data::ITensors.ITensor)
    inds = ITensors.inds(tensor_data)
    dense_array = ITensors.array(tensor_data, inds...)
    return is_gpu_array(dense_array) ? BackendGPU : BackendCPU
end

function backend_of(mps::ITensorMPS.MPS)
    if length(mps) == 0
        return BackendCPU
    end
    return backend_of(mps[1])
end

backend_of(gate::AbstractGate) = backend_of(tensor(gate))

function backend_of(circuit::AbstractVector{<:AbstractGate})
    if isempty(circuit)
        return BackendCPU
    end
    first_backend = backend_of(first(circuit))
    for gate in circuit
        if backend_of(gate) != first_backend
            throw(ArgumentError("Circuit has mixed gate backends."))
        end
    end
    return first_backend
end

function resolve_backend(config::BackendConfig)
    if config.backend == BackendCPU
        return BackendCPU
    elseif config.backend == BackendGPU
        if _cuda_is_functional()
            return BackendGPU
        elseif config.gpu_fallback
            return BackendCPU
        end
        error("GPU backend requested, but CUDA is unavailable or non-functional.")
    elseif config.backend == BackendAuto
        if _cuda_is_functional()
            return BackendGPU
        else
            return BackendCPU
        end
    end
    error("Unhandled backend mode: $(config.backend)")
end

function _validate_precision(precision::Symbol)
    if precision in (:fp32, :fp64, :preserve)
        return precision
    end
    throw(ArgumentError("Unknown precision mode: $(precision). Expected one of :fp32, :fp64, :preserve."))
end

"""
    to_backend(data, backend::ComputeBackend; precision::Symbol=:fp32)

Move `data` to the requested compute backend with explicit precision semantics.
Use `precision=:fp32` for throughput-oriented GPU transfers, `precision=:fp64`
for high-precision runs, or `precision=:preserve` to keep existing floating-point types.
"""
function to_backend(data, backend::ComputeBackend; precision::Symbol=:fp32)
    resolved_precision = _validate_precision(precision)
    if backend == BackendCPU
        return _to_backend_cpu(data; precision=resolved_precision)
    elseif backend == BackendGPU
        return _to_backend_gpu(data; precision=resolved_precision)
    elseif backend == BackendAuto
        return data
    end
    error("Unhandled backend mode: $(backend)")
end

# CPU-safe defaults. CUDA extension methods specialize these for GPU-aware movement.
_to_backend_cpu(data; precision::Symbol=:fp32) = data
_to_backend_gpu(data; precision::Symbol=:fp32) = error("GPU backend conversion is unavailable. Install CUDA.jl and ensure the MPSCircuitsCUDA extension is loaded.")

function _cast_backend_precision(array_data::AbstractArray, precision::Symbol)
    if precision == :preserve
        return array_data
    elseif precision == :fp32
        target_eltype = eltype(array_data) <: Complex ? ComplexF32 : (eltype(array_data) <: AbstractFloat ? Float32 : eltype(array_data))
    elseif precision == :fp64
        target_eltype = eltype(array_data) <: Complex ? ComplexF64 : (eltype(array_data) <: AbstractFloat ? Float64 : eltype(array_data))
    else
        return array_data
    end
    if target_eltype === eltype(array_data)
        return array_data
    end
    return convert.(target_eltype, array_data)
end

function _to_backend_cpu(array_data::AbstractArray; precision::Symbol=:preserve)
    host_array = is_gpu_array(array_data) ? Array(array_data) : array_data
    return _cast_backend_precision(host_array, precision)
end

function _to_backend_cpu(tensor::ITensors.ITensor; precision::Symbol=:preserve)
    inds = ITensors.inds(tensor)
    dense_array = ITensors.array(tensor, inds...)
    cast_array = _to_backend_cpu(dense_array; precision=precision)
    return ITensors.itensor(cast_array, inds...)
end

function _to_backend_cpu(mps::ITensorMPS.MPS; precision::Symbol=:preserve)
    converted = deepcopy(mps)
    for n in 1:length(converted)
        converted[n] = _to_backend_cpu(converted[n]; precision=precision)
    end
    return converted
end

function to_backend(gate::UnitaryGate, backend::ComputeBackend; precision::Symbol=:fp32)
    converted_tensor = to_backend(gate.tensor, backend; precision=precision)
    return UnitaryGate(converted_tensor, gate.site_numbers, gate.site_indices)
end

function to_backend(gate::SU2Gate{Seq}, backend::ComputeBackend; precision::Symbol=:fp32) where {Seq<:EulerSequence}
    converted_tensor = to_backend(gate.tensor, backend; precision=precision)
    return SU2Gate{Seq}(converted_tensor, gate.site_numbers, gate.site_indices, gate.theta_1, gate.theta_2, gate.theta_3)
end

function to_backend(gate::KAKCore, backend::ComputeBackend; precision::Symbol=:fp32)
    converted_tensor = to_backend(gate.tensor, backend; precision=precision)
    return KAKCore(converted_tensor, gate.site_numbers, gate.site_indices, gate.alpha, gate.beta, gate.gamma)
end

function to_backend(gate::KAKGateSU4, backend::ComputeBackend; precision::Symbol=:fp32)
    converted_tensor = to_backend(gate.tensor, backend; precision=precision)
    converted_core = to_backend(gate.core, backend; precision=precision)
    converted_A_L1 = to_backend(gate.A_L1, backend; precision=precision)
    converted_A_L2 = to_backend(gate.A_L2, backend; precision=precision)
    converted_A_R1 = to_backend(gate.A_R1, backend; precision=precision)
    converted_A_R2 = to_backend(gate.A_R2, backend; precision=precision)
    return KAKGateSU4(
        converted_tensor,
        gate.site_numbers,
        gate.site_indices,
        converted_core,
        converted_A_L1,
        converted_A_L2,
        converted_A_R1,
        converted_A_R2,
    )
end

to_backend(gate::AbstractGate, ::ComputeBackend; precision::Symbol=:fp32) = gate

function to_backend(circuit::AbstractVector{T}, backend::ComputeBackend; precision::Symbol=:fp32) where {T<:AbstractGate}
    converted = Vector{T}(undef, length(circuit))
    for (i, gate) in pairs(circuit)
        converted[i] = to_backend(gate, backend; precision=precision)
    end
    return converted
end

"""
    place_on_backend_boundary(data, backend::ComputeBackend, policy::BackendTransferPolicy; precision::Symbol=:fp32)

Place data on the selected backend at coarse phase boundaries. In Stage 3, both
policies co-locate data with `backend`; future stages can differentiate
small-kernel CPU islands while preserving this single entry point.
"""
function place_on_backend_boundary(data, backend::ComputeBackend, policy::BackendTransferPolicy; precision::Symbol=:fp32)
    if policy == PreferGPUResidency || policy == StrictSingleBackend
        return to_backend(data, backend; precision=precision)
    end
    error("Unhandled transfer policy: $(policy)")
end


function _validate_mixed_device_policy(policy::Symbol)
    if policy in (:error, :coerce)
        return policy
    end
    throw(ArgumentError("Unknown mixed_device policy: $(policy). Expected one of :error, :coerce."))
end

_backend_label(backend) = backend == BackendGPU ? "GPU" : "CPU"
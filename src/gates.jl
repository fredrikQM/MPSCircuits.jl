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

abstract type AbstractGate end
struct DummyGate <: AbstractGate end

"""
    UnitaryGate(tensor::ITensors.ITensor, site_numbers::Vector{Int}, site_indices::Vector{ITensors.Index})

Gate abstraction that stores a unitary ITensor and the site indices it acts on.
"""
struct UnitaryGate <: AbstractGate
    tensor::ITensors.ITensor
    site_numbers::Vector{Int}
    site_indices::Vector{<:ITensors.Index}
end
tensor(gate::AbstractGate) = gate.tensor # assumes every gate has a .tensor field, will need to change this if we ever have lazy evaluation / caching

"""
    UnitaryGate(unitary::ITensors.ITensor)

Convenience constructor for UnitaryGate that extracts the site numbers and indices from the ITensor.
"""
function UnitaryGate(unitary::ITensors.ITensor)
    return UnitaryGate(unitary, gate_tensor_indices(unitary)...)
end

"""
    gate_tensor_indices(gate::ITensor)

Given an ITensor representing a gate, extract the site numbers (and sites) that the gate acts on.
This function assumes that the gate's indices are tagged with "n=X" where X is the site number.
Note that the string comprehension is probably slow vs. 4x4 tensor manipulations, so use this with care inside hot loops.
"""
function gate_tensor_indices(gate_tensor::ITensors.ITensor)
    unprimed_inds = collect(filter(i -> ITensors.plev(i) == 0, ITensors.inds(gate_tensor)))
    site_numbers = [parse(Int, match(r"n=(\d+)", string(ITensors.tags(i)))[1]) for i in unprimed_inds]
    perm = sortperm(site_numbers)

    return site_numbers[perm], unprimed_inds[perm]
end

"""
    compose(left::AbstractGate, right::AbstractGate)

Compose two gates in the same way `ITensors.apply` composes the underlying ITensors.
"""
function compose(left::AbstractGate, right::AbstractGate)
    composed_tensor = ITensors.apply(tensor(left), tensor(right))
    # collate the site numbers and indices, incl duplicates
    known_indices = vcat(left.site_indices, right.site_indices)
    known_sites = vcat(left.site_numbers, right.site_numbers)
    # filter out the unprimed indices of the new composite tensor
    new_unprimed = collect(filter(i -> ITensors.plev(i) == 0, ITensors.inds(composed_tensor)))
    # now get the integer site numbers
    new_site_numbers = Int[]
    for idx in new_unprimed
        parent_pos = findfirst(==(idx), known_indices) # == comparison on indices will use a fast ID check under the hood
        push!(new_site_numbers, known_sites[parent_pos])
    end

    perm = sortperm(new_site_numbers)
    return UnitaryGate(composed_tensor, new_site_numbers[perm], new_unprimed[perm])
end

"""
    dagger(gate::AbstractGate)

Return the adjoint gate: conjugated and transposed (i.e. prime level swapped)
"""
function dagger(gate::AbstractGate)
    return UnitaryGate(ITensors.swapprime(ITensors.dag(tensor(gate)), 0 => 1), gate.site_numbers, gate.site_indices)
end

"""
    apply_gate(gate::AbstractGate, mps::ITensorMPS.MPS; mixed_device::Symbol=:error, conversion_precision::Symbol=:preserve, kwargs...)

Apply a single Gate to an MPS by extracting its underlying unitary ITensor.
If `mixed_device=:error`, throw when gate and MPS backends differ.
If `mixed_device=:coerce`, convert the gate to the MPS backend first.
"""
function apply_gate(
    gate::AbstractGate,
    mps::ITensorMPS.MPS;
    mixed_device::Symbol=:error,
    conversion_precision::Symbol=:preserve,
    kwargs...
)
    policy = _validate_mixed_device_policy(mixed_device)
    gate_backend = backend_of(gate)
    mps_backend = backend_of(mps)
    gate_co_located = gate
    if gate_backend != mps_backend
        if policy == :coerce
            gate_co_located = to_backend(gate, mps_backend; precision=conversion_precision)
        else
            throw(ArgumentError("Mixed backend in apply_gate: gate on $(_backend_label(gate_backend)) and MPS on $(_backend_label(mps_backend)). Set mixed_device=:coerce to auto-convert."))
        end
    end
    return ITensorMPS.apply(tensor(gate_co_located), mps; kwargs...)
end

"""
    apply_circuit(circuit::AbstractVector{<:AbstractGate}, mps::ITensorMPS.MPS; mixed_device::Symbol=:error, conversion_precision::Symbol=:preserve, kwargs...)

Apply a Gate circuit to an MPS by extracting all underlying unitaries.
If `mixed_device=:error`, throw when any gate backend differs from the MPS backend.
If `mixed_device=:coerce`, convert the whole circuit to the MPS backend first.
"""
function apply_circuit(
    circuit::AbstractVector{<:AbstractGate},
    mps::ITensorMPS.MPS;
    mixed_device::Symbol=:error,
    conversion_precision::Symbol=:preserve,
    kwargs...
)
    policy = _validate_mixed_device_policy(mixed_device)
    if isempty(circuit)
        return mps
    end
    mps_backend = backend_of(mps)
    if policy == :coerce
        circuit_co_located = to_backend(circuit, mps_backend; precision=conversion_precision)
        return ITensorMPS.apply(tensor.(circuit_co_located), mps; kwargs...)
    end

    for (gate_index, gate) in pairs(circuit)
        gate_backend = backend_of(gate)
        if gate_backend != mps_backend
            throw(ArgumentError("Mixed backend in apply_circuit: gate $(gate_index) on $(_backend_label(gate_backend)) and MPS on $(_backend_label(mps_backend)). Set mixed_device=:coerce to auto-convert."))
        end
    end
    return ITensorMPS.apply(tensor.(circuit), mps; kwargs...)
end

# Support for future Euler rotations (e.g. XYZ), for now just ZYZ
abstract type EulerSequence end

"""
    SU2Gate{Seq}(tensor::ITensors.ITensor, site_numbers::Vector{Int}, site_indices::Vector{ITensors.Index}, theta_1::Float64, theta_2::Float64, theta_3::Float64)

Extension of UnitaryGate that also stores the Euler angles for the underlying SU(2) operation, for use in transpilation.
"""
struct SU2Gate{Seq<:EulerSequence} <: AbstractGate # N.B.: this is a parametric struct so we can flag methods for Euler angle methods
    tensor::ITensors.ITensor
    site_numbers::Vector{Int}
    site_indices::Vector{<:ITensors.Index}
    theta_1::Float64
    theta_2::Float64
    theta_3::Float64
end

struct ZYZ <: EulerSequence end

"""
    SU2Gate{ZYZ}(tensor::ITensors.ITensor, site_numbers::Vector{Int}, site_indices::Vector{<:ITensors.Index})

Construct a single-qubit `SU2Gate{ZYZ}` from an ITensor and metadata.
The ZYZ Euler angles are extracted from the underlying 2x2 matrix and stored
on the returned gate for downstream transpilation.
"""
function SU2Gate{ZYZ}(tensor::ITensors.ITensor, site_numbers::Vector{Int}, site_indices::Vector{<:ITensors.Index})
    gate_matrix = ITensors.array(tensor, ITensors.prime(site_indices[1]), site_indices[1])
    theta_1, theta_2, theta_3 = zyz_decomposition(gate_matrix)
    return SU2Gate{ZYZ}(tensor, site_numbers, site_indices, theta_1, theta_2, theta_3)
end

"""
    SU2Gate{ZYZ}(gate::UnitaryGate)

Convert a `UnitaryGate` into an `SU2Gate{ZYZ}` by decomposing it into ZYZ Euler
angles. The input must act on exactly one site.
"""
function SU2Gate{ZYZ}(gate::UnitaryGate)
    if length(gate.site_numbers) != 1
        throw(ArgumentError("SU(2) decomposition only possible for single-qubit gates, but input gate acts on $(length(gate.site_numbers)) qubits."))
    end
    return SU2Gate{ZYZ}(tensor(gate), gate.site_numbers, gate.site_indices)
end

# Identity conversion: if it's already a ZYZ SU(2) gate, return it unchanged.
SU2Gate{ZYZ}(gate::SU2Gate{ZYZ}) = gate

"""
    SU2Gate{ZYZ}(theta_1::Real, theta_2::Real, theta_3::Real, site_numbers::Vector{Int}, site_indices::Vector{<:ITensors.Index})

Construct a single-qubit `SU2Gate{ZYZ}` directly from ZYZ Euler angles and site
metadata. This builds the corresponding 2x2 unitary and wraps it as an ITensor.
"""
function SU2Gate{ZYZ}(theta_1::Real, theta_2::Real, theta_3::Real, site_numbers::Vector{Int}, site_indices::Vector{<:ITensors.Index})
    # precompute amplitudes
    c2 = cos(theta_2 / 2.0)
    s2 = sin(theta_2 / 2.0)
    phase_00 = exp(-im * (theta_1 + theta_3) / 2.0)
    phase_11 = exp(im * (theta_1 + theta_3) / 2.0)
    phase_01 = exp(im * (theta_1 - theta_3) / 2.0)
    phase_10 = exp(-im * (theta_1 - theta_3) / 2.0)
    # evaluate matrix in one go
    gate_matrix = [
        phase_00*c2 -phase_01*s2;
        phase_10*s2 phase_11*c2
    ]
    gate_tensor = ITensors.itensor(gate_matrix, ITensors.prime(site_indices[1]), site_indices[1])

    return SU2Gate{ZYZ}(gate_tensor, site_numbers, site_indices, theta_1, theta_2, theta_3)
end

"""
TODO: In future if other Euler sequences are added we can avoid repeated boilerplate with a signature like this:
function SU2Gate{Seq}(
    tensor::ITensors.ITensor, 
    site_numbers::Vector{Int}, 
    site_indices::Vector{ITensors.Index}
) where {Seq <: EulerSequence}
"""



"""
    KAKCore

Entangling core (Weyl chamber) of a Cartan KAK decomposition, can be represented as 3 CNOTs and 3 single-qubit rotations.
"""
struct KAKCore <: AbstractGate
    tensor::ITensors.ITensor
    site_numbers::Vector{Int}
    site_indices::Vector{<:ITensors.Index}
    alpha::Float64
    beta::Float64
    gamma::Float64
end

"""
    KAKCore(site_numbers::Vector{Int}, site_indices::Vector{<:ITensors.Index}, alpha::Real, beta::Real, gamma::Real)

Initializer for a KAK entangling core from the angles.
"""
function KAKCore(site_numbers::Vector{Int}, site_indices::Vector{<:ITensors.Index}, alpha::Real, beta::Real, gamma::Real)
    # Apply the rotations from Vatan (arXiv:quant-ph/0308006), Figure 6. N.B.: our Rz and Ry are reversed, so angle signs are flipped. Our convention follows e.g. Qiskit.
    rot_1 = LinearAlgebra.kron(Rz(-2 * gamma + pi / 2), Ry(-pi / 2 + 2 * alpha))
    rot_2 = LinearAlgebra.kron(ComplexF64[1 0; 0 1], Ry(-2 * beta + pi / 2))
    core_matrix = CNOT2 * rot_2 * CNOT1 * rot_1 * CNOT2
    # N.B.: the ordering is site_indices[2], site_indices[1] because of Julia's column-major ordering; this ensures the usual physics convention for A⊗B
    core_tensor = ITensors.itensor(core_matrix, ITensors.prime(site_indices[2]), ITensors.prime(site_indices[1]), site_indices[2], site_indices[1])
    return KAKCore(core_tensor, site_numbers, site_indices, alpha, beta, gamma)
end



"""
    KAKGateSU4

Representation of a two-qubit gate in Cartan KAK-decomposed form.
"""
struct KAKGateSU4 <: AbstractGate
    tensor::ITensors.ITensor
    site_numbers::Vector{Int}
    site_indices::Vector{<:ITensors.Index}
    core::KAKCore
    A_L1::SU2Gate{ZYZ}
    A_L2::SU2Gate{ZYZ}
    A_R1::SU2Gate{ZYZ}
    A_R2::SU2Gate{ZYZ}
end

"""
    KAKGateSU4(input_gate::UnitaryGate)

Perform a Cartan KAK decomposition of a two-qubit `UnitaryGate` and output a `KAKGateSU4`.
"""
function KAKGateSU4(
    input_gate::UnitaryGate
)
    if length(input_gate.site_numbers) != 2
        throw(ArgumentError("KAK decomposition only implemented for 2-qubit gates, but input gate acts on $(length(input_gate.site_numbers)) qubits"))
    end
    site_numbers = input_gate.site_numbers
    site_indices = input_gate.site_indices
    # N.B.: the ordering is site_indices[2], site_indices[1] because of Julia's column-major ordering; this ensures the usual physics convention for A⊗B
    gate_matrix = LinearAlgebra.reshape(ITensors.array(tensor(input_gate), ITensors.prime(site_indices[2]), ITensors.prime(site_indices[1]), site_indices[2], site_indices[1]), (4, 4))
    α, β, γ, A_L1, A_L2, A_R1, A_R2 = cartan_KAK_decomposition(gate_matrix)
    tensor_A_L1 = ITensors.itensor(A_L1, ITensors.prime(site_indices[1]), site_indices[1])
    tensor_A_L2 = ITensors.itensor(A_L2, ITensors.prime(site_indices[2]), site_indices[2])
    tensor_A_R1 = ITensors.itensor(A_R1, ITensors.prime(site_indices[1]), site_indices[1])
    tensor_A_R2 = ITensors.itensor(A_R2, ITensors.prime(site_indices[2]), site_indices[2])
    gate_A_L1 = SU2Gate{ZYZ}(tensor_A_L1, [site_numbers[1]], [site_indices[1]])
    gate_A_L2 = SU2Gate{ZYZ}(tensor_A_L2, [site_numbers[2]], [site_indices[2]])
    gate_A_R1 = SU2Gate{ZYZ}(tensor_A_R1, [site_numbers[1]], [site_indices[1]])
    gate_A_R2 = SU2Gate{ZYZ}(tensor_A_R2, [site_numbers[2]], [site_indices[2]])
    # Now compose the full KAK gate
    core = KAKCore(site_numbers, site_indices, α, β, γ)
    su4_tensor = ITensors.apply(tensor(core), tensor(gate_A_R1))
    su4_tensor = ITensors.apply(su4_tensor, tensor(gate_A_R2))
    su4_tensor = ITensors.apply(tensor(gate_A_L1), su4_tensor)
    su4_tensor = ITensors.apply(tensor(gate_A_L2), su4_tensor)
    return KAKGateSU4(su4_tensor, site_numbers, site_indices, core, gate_A_L1, gate_A_L2, gate_A_R1, gate_A_R2)
end
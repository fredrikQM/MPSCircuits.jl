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

import JSON3

"""
    gate_SU2_to_dicts(gate_su2::SU2Gate)

Converts an SU(2) gate to a list of dictionaries representing the Rz, Ry, Rz rotations for the Euler decomposition.
"""
function gate_SU2_to_dicts(gate_su2::SU2Gate)
    rz_1 = Dict(
        "op" => "Rz",
        "target" => gate_su2.site_numbers[1],
        "angle" => gate_su2.theta_1
    )
    ry_1 = Dict(
        "op" => "Ry",
        "target" => gate_su2.site_numbers[1],
        "angle" => gate_su2.theta_2
    )
    rz_2 = Dict(
        "op" => "Rz",
        "target" => gate_su2.site_numbers[1],
        "angle" => gate_su2.theta_3
    )
    return [rz_1, ry_1, rz_2]
end

"""
    gate_KAK_core_to_dicts(kak_core::KAKCore)

Converts a KAK core gate (Weyl chamber) to a list of dictionaries representing the CNOTs and Ry/Rz gates for its standard decomposition.
"""
function gate_KAK_core_to_dicts(kak_core::KAKCore)
    cnot_1 = Dict(
        "op" => "CNOT",
        "control" => kak_core.site_numbers[2],
        "target" => kak_core.site_numbers[1]
    )
    rz_1 = Dict(
        "op" => "Rz",
        "target" => kak_core.site_numbers[1],
        "angle" => -2 * kak_core.gamma + π / 2
    )
    ry_1 = Dict(
        "op" => "Ry",
        "target" => kak_core.site_numbers[2],
        "angle" => -π / 2 + 2 * kak_core.alpha
    )
    cnot_2 = Dict(
        "op" => "CNOT",
        "control" => kak_core.site_numbers[1],
        "target" => kak_core.site_numbers[2]
    )
    ry_2 = Dict(
        "op" => "Ry",
        "target" => kak_core.site_numbers[2],
        "angle" => -2 * kak_core.beta + π / 2
    )
    cnot_3 = Dict(
        "op" => "CNOT",
        "control" => kak_core.site_numbers[2],
        "target" => kak_core.site_numbers[1]
    )
    return [cnot_1, rz_1, ry_1, cnot_2, ry_2, cnot_3]
end

"""
    gate_KAK_SU4_to_dicts(kak_gate::KAKGateSU4)

Converts a KAK-decomposed SU(4) gate to a list of dictionaries representing the core and SU(2) boundary gates for its standard decomposition.
"""
function gate_KAK_SU4_to_dicts(kak_gate::KAKGateSU4)
    dicts = []
    append!(dicts, gate_SU2_to_dicts(kak_gate.A_R1))
    append!(dicts, gate_SU2_to_dicts(kak_gate.A_R2))
    append!(dicts, gate_KAK_core_to_dicts(kak_gate.core))
    append!(dicts, gate_SU2_to_dicts(kak_gate.A_L1))
    append!(dicts, gate_SU2_to_dicts(kak_gate.A_L2))
    return dicts
end


"""
    circuit_to_ops_array(circuit::Vector{<:AbstractGate})

Converts a circuit (vector of AbstractGate) to an array of dictionaries representing the operations in the circuit.
Calls dedicated subroutines for each implemented gate type, ultimately all via a Cartan KAK decomposition to CNOT/Ry/Rz gate-set.
"""
function circuit_to_ops_array(circuit::Vector{<:AbstractGate})
    circuit_ops_array = []
    for gate in circuit
        if gate isa MPSCircuits.UnitaryGate
            if length(gate.site_numbers) == 1
                # Decompose to SU(2) gate and export the Rz, Ry, Rz rotations for the Euler decomposition.
                append!(circuit_ops_array, gate_SU2_to_dicts(MPSCircuits.SU2Gate{ZYZ}(gate)))
            elseif length(gate.site_numbers) == 2
                # Decompose SU(4) gate with KAK and export the core and SU(2) boundary gates.
                append!(circuit_ops_array, gate_KAK_SU4_to_dicts(MPSCircuits.KAKGateSU4(gate)))
            else
                error("Warning: UnitaryGate with unsupported number of sites: ", gate.site_numbers)
            end
        elseif gate isa MPSCircuits.SU2Gate
            # Export the Rz, Ry, Rz rotations for the Euler decomposition.
            append!(circuit_ops_array, gate_SU2_to_dicts(gate))
        elseif gate isa MPSCircuits.KAKCore
            # Export the CNOTs and Ry/Rz gates for the KAK core decomposition.
            append!(circuit_ops_array, gate_KAK_core_to_dicts(gate))
        elseif gate isa MPSCircuits.KAKGateSU4
            # Export the core and SU(2) boundary gates for the KAK SU(4) decomposition.
            append!(circuit_ops_array, gate_KAK_SU4_to_dicts(gate))
        else
            # This shouldn't happen.
            error("Unknown gate type: $(typeof(gate))")
        end
    end
    num_qubits = maximum([maximum(gate.site_numbers) for gate in circuit])
    return circuit_ops_array, num_qubits
end

"""
    write_circuit_json(circuit::Vector{<:AbstractGate}, filename::String)

Writes a circuit (vector of AbstractGate) to a JSON file with the given filename. 
"""
function write_circuit_json(circuit::Vector{<:AbstractGate}, filename::String)
    circuit_ops_array, num_qubits = circuit_to_ops_array(circuit)
    json_data = Dict(
        "num_qubits" => num_qubits,
        "operations" => circuit_ops_array
    )
    open(filename, "w") do io
        JSON3.write(io, json_data)
    end
end
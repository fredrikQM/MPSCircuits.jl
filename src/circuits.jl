# TODO: probably create a Circuit type that holds the Vector{<:AbstractGate} and some metadata, like sites etc.

"""
    expand_SU4s(circuit::Vector{<:AbstractGate})

Expand any KAKGateSU4s in `circuit` into the 5 components (four boundary SU(2)s and one 3-CNOT, 3-rotation core).
"""
function expand_SU4s(circuit::Vector{<:AbstractGate})
    output_circuit = AbstractGate[]
    for gate in circuit
        if gate isa KAKGateSU4
            push!(output_circuit, gate.A_R1, gate.A_R2, gate.core, gate.A_L1, gate.A_L2)
        else
            push!(output_circuit, gate)
        end
    end
    return output_circuit
end

"""
    fuse_SU2s(circuit::Vector{<:AbstractGate}, n_qubits::Int)

Fuse consecutive SU(2) gates on the same qubit in `circuit` into a single SU(2) gate.
"""
function fuse_SU2s(circuit::Vector{<:AbstractGate}, n_qubits::Int)
    output_circuit = AbstractGate[]
    buffer_gates = AbstractGate[DummyGate() for _ in 1:n_qubits] # placeholder gates to hold the buffered SU(2)s for each qubit
    for gate in circuit
        if gate isa SU2Gate{ZYZ}
            target_qubit = gate.site_numbers[1]
            if buffer_gates[target_qubit] isa DummyGate
                buffer_gates[target_qubit] = gate
            else
                buffer_gates[target_qubit] = compose(gate, buffer_gates[target_qubit]) # Fuse the new gate with the buffered one and update the buffer
            end
        else
            # Flush the buffer on affected qubits only
            for qubit in gate.site_numbers
                push!(output_circuit, SU2Gate{ZYZ}(buffer_gates[qubit]))
                buffer_gates[qubit] = DummyGate() # reset the buffer for this qubit
            end
            push!(output_circuit, gate)
        end
    end
    # Flush any remaining buffered gates at the end
    for qubit in 1:n_qubits
        push!(output_circuit, SU2Gate{ZYZ}(buffer_gates[qubit]))
    end
    return output_circuit
end
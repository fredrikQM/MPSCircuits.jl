"""
    head_tensor_gate(mps::ITensorMPS.MPS)

Extract the first 2-qubit gate from a right-canonical MPS.
"""
function _tensor_precision_symbol(reference::ITensors.ITensor)
    inds = ITensors.inds(reference)
    dense_array = ITensors.array(reference, inds...)
    T = eltype(dense_array)
    if T <: ComplexF32 || T <: Float32
        return :fp32
    elseif T <: ComplexF64 || T <: Float64
        return :fp64
    end
    return :preserve
end

function _identity_like(reference::ITensors.ITensor, i1::ITensors.Index, i2::ITensors.Index)
    precision = _tensor_precision_symbol(reference)
    backend = backend_of(reference)
    return to_backend(ITensors.delta(i1, i2), backend; precision=precision)
end

function _scalar_tensor_like(reference::ITensors.ITensor, index::ITensors.Index)
    precision = _tensor_precision_symbol(reference)
    backend = backend_of(reference)
    return to_backend(ITensors.itensor([1.0], index), backend; precision=precision)
end

function head_tensor_gate(mps::ITensorMPS.MPS)
    if mps.llim > 1
        error("MPS is not in the expected right-canonical gauge.")
    end
    # Extract the head tensor and its indices
    A = mps[1]
    current_physical_index = ITensorMPS.siteinds(mps, 1)[1]
    next_physical_index = ITensorMPS.siteinds(mps, 2)[1]
    bond_index = ITensorMPS.linkinds(mps, 1)[1]

    # Remap the isometry to match the physical indices it acts on, and add a trivial index for matricization
    trivial_index = ITensors.Index(1; tags="trivial")
    A_mapped = A *
               _identity_like(A, current_physical_index, ITensors.dag(ITensors.prime(current_physical_index))) *
               _identity_like(A, bond_index, ITensors.dag(ITensors.prime(next_physical_index))) *
               _scalar_tensor_like(A, trivial_index)

    # Matricize A_mapped into an isometric matrix (cols = unprimed indices, rows = primed indices)
    cmb_cols = ITensors.combiner(
        ITensors.prime(current_physical_index),
        ITensors.prime(next_physical_index);
        tags="c_cols",
    )
    ind_cols = ITensors.combinedind(cmb_cols)
    A_mapped = A_mapped * cmb_cols
    isometry_matrix = ITensors.matrix(A_mapped, ind_cols, trivial_index)

    # Use the full positive QR decomposition to get a unitary matrix
    Q_full = full_positive_qr(isometry_matrix)

    # Reshape Q_full back into a tensor. The original entries should map to the first column only.
    cmb_unprimed = ITensors.combiner(current_physical_index, next_physical_index; tags="c_unprimed")
    ind_unprimed = ITensors.combinedind(cmb_unprimed)
    gate_combined = ITensors.itensor(Q_full, ind_cols, ind_unprimed)
    gate = gate_combined * ITensors.dag(cmb_cols) * ITensors.dag(cmb_unprimed)

    # TODO: use index composition to speed this up...
    return UnitaryGate(gate)
end


"""
    bulk_tensor_gate(mps::ITensorMPS.MPS, site_num::Int)

Extract a 2-qubit gate for `site_num` and `site_num+1` from a right-canonical MPS.
"""
function bulk_tensor_gate(mps::ITensorMPS.MPS, site_num::Int)
    # Check: is the MPS in the correct gauge?
    if mps.llim > site_num
        error("MPS is not in the expected right-canonical gauge near site $(site_num).")
    end

    # Extract the bulk tensor and its indices
    A = mps[site_num]
    left_bond = ITensorMPS.linkinds(mps, site_num - 1)[1]
    right_bond = ITensorMPS.linkinds(mps, site_num)[1]
    current_physical_index = ITensorMPS.siteinds(mps, site_num)[1]
    next_physical_index = ITensorMPS.siteinds(mps, site_num + 1)[1]

    # Remap the isometry to match the physical indices it acts on
    A_mapped = A *
               _identity_like(A, right_bond, ITensors.dag(ITensors.prime(next_physical_index))) *
               _identity_like(A, current_physical_index, ITensors.dag(ITensors.prime(current_physical_index))) *
               _identity_like(A, left_bond, ITensors.dag(current_physical_index))

    # Matricize A_mapped into an isometric matrix (cols = unprimed indices, rows = primed indices)
    cmb_cols = ITensors.combiner(
        ITensors.prime(current_physical_index),
        ITensors.prime(next_physical_index);
        tags="c_cols",
    )
    ind_cols = ITensors.combinedind(cmb_cols)
    A_mapped = A_mapped * cmb_cols
    isometry_matrix = ITensors.matrix(A_mapped, ind_cols, current_physical_index)

    Q_full = full_positive_qr(isometry_matrix)

    # Reshape Q_full back into a tensor with the original indices, being very careful about the insertion of the new columns!
    cmb_unprimed = ITensors.combiner(current_physical_index, next_physical_index; tags="c_unprimed")
    ind_unprimed = ITensors.combinedind(cmb_unprimed)
    gate_combined = ITensors.itensor(Q_full, ind_cols, ind_unprimed)
    gate = gate_combined * ITensors.dag(cmb_cols) * ITensors.dag(cmb_unprimed)

    # TODO: use index composition to speed this up...
    return UnitaryGate(gate)
end


"""
    tail_tensor_gate(mps::ITensorMPS.MPS)

Build the final 1-qubit gate from the last MPS tensor.
"""
function tail_tensor_gate(mps::ITensorMPS.MPS)
    N = length(mps)
    tail_tensor = mps[N]
    physical_index = ITensorMPS.siteinds(mps, N)[1]
    bond_index = ITensorMPS.linkinds(mps, N - 1)[1]

    # Remap the isometry to match the physical indices it acts on - if bond dimension is 1, the delta tensor automatically pads it  with zeros to match the full dimension 2, so we'll pad out the nullspace with QR.
    A_mapped = tail_tensor *
               _identity_like(tail_tensor, physical_index, ITensors.dag(ITensors.prime(physical_index))) *
               _identity_like(tail_tensor, bond_index, ITensors.dag(physical_index))

    # Now matricize and use QR to fill out nullspace.
    isometry_matrix = ITensors.matrix(A_mapped, ITensors.prime(physical_index), physical_index)
    Q_full = full_positive_qr(isometry_matrix)
    gate = ITensors.itensor(Q_full, ITensors.prime(physical_index), physical_index)

    # TODO: use index composition to speed this up...
    return UnitaryGate(gate)
end


"""
    truncated_preparation_circuit(mps::ITensorMPS.MPS)

Extract a gate list from an MPS using the notebook-derived one-layer procedure.
The input MPS is copied, truncated, and orthogonalized internally.
"""
function truncated_preparation_circuit(mps::ITensorMPS.MPS)
    N = length(mps)
    if N < 2
        error("truncated_preparation_circuit requires at least 2 sites; got $(N).")
    end
    if ITensorMPS.maxlinkdim(mps) != 2
        error("truncated_preparation_circuit expects an MPS with max bond dimension χ=2; got χ=$(ITensorMPS.maxlinkdim(mps)).")
    end

    mps_work = deepcopy(mps)
    mps_truncated = ITensorMPS.truncate(mps_work; maxdim=2)
    ITensorMPS.orthogonalize!(mps_truncated, 1; maxdim=2)

    head_gate = head_tensor_gate(mps_truncated)
    tail_gate = tail_tensor_gate(mps_truncated)

    if N == 2
        return [compose(tail_gate, head_gate)]
    end

    bulk_gates = [bulk_tensor_gate(mps_truncated, n) for n in 2:(N-1)]
    bulk_gates[end] = compose(tail_gate, bulk_gates[end])

    return [head_gate, bulk_gates...]
end


"""
    product_state_site_gate(mps::ITensorMPS.MPS, site::Int)
    
Take an MPS with χ=1 (product state) and return the 1-qubit gate that prepares the state on `site` from |0>.
"""
function product_state_site_gate(mps::ITensorMPS.MPS, site::Int)
    if ITensorMPS.maxlinkdim(mps) != 1
        error("product_state_site_gate expects an MPS with max bond dimension χ=1; got χ=$(ITensorMPS.maxlinkdim(mps)).")
    end
    # Extract the tensor at this site
    A = mps[site]
    # Trivial index for matricization: bond index for boundary tensors, or combined for bulk
    if site == 1
        trivial_index = ITensorMPS.linkinds(mps, site)[1]
    elseif site == length(mps)
        trivial_index = ITensorMPS.linkinds(mps, site - 1)[1]
    else
        cmb = ITensors.combiner(ITensorMPS.linkinds(mps, site - 1)[1], ITensorMPS.linkinds(mps, site)[1])
        trivial_index = ITensors.combinedind(cmb)
        A = A * cmb
    end
    # Matricize into a trivial isometry (cols = physical index, rows = trivial index)
    physical_index = ITensorMPS.siteinds(mps, site)[1]
    isometry_matrix = ITensors.matrix(A, physical_index, trivial_index)
    # Use the full positive QR decomposition to get a unitary matrix
    Q_full = full_positive_qr(isometry_matrix)
    gate = ITensors.itensor(Q_full, ITensors.prime(physical_index), physical_index)

    # TODO: use index composition to speed this up...
    return UnitaryGate(gate)
end


"""
    single_qubit_frame(mps::ITensorMPS.MPS) -> (layer::Vector{UnitaryGate}, fidelity::Float64)

The product-frame fixup for `mps`, returned as a standalone single-qubit layer rather than
composed into anything. Ran's two-qubit layers drive the state to *a* product state, not
necessarily |00...0>, so a final layer of one-qubit gates carries |00...0> into that frame.

Returns the layer and the fidelity of the χ=1 truncation with `mps`.

Front-of-circuit gates are applied last in the disentangling pass (`evaluate_circuit_fidelity`
walks `circuit[end:-1:1]`), so `vcat(layer, circuit)` places this as the final fixup — see
[`with_single_qubit_rotations`](@ref).
"""
function single_qubit_frame(mps::ITensorMPS.MPS)
    product_mps = ITensorMPS.truncate(mps; maxdim=1) # truncate down to a product state forcibly
    product_truncation_fidelity = abs2(ITensorMPS.inner(product_mps, mps) / ITensorMPS.inner(mps, mps)) # how much fidelity do we preserve dropping to χ=1? should be near 1 if this is to work properly...
    single_qubit_layer = [product_state_site_gate(product_mps, n) for n in 1:length(product_mps)]
    target_backend = backend_of(mps)
    single_qubit_layer = to_backend(single_qubit_layer, target_backend; precision=:preserve)
    return (single_qubit_layer, product_truncation_fidelity)
end


"""
    with_single_qubit_rotations(circuit::Vector{UnitaryGate}, mps::ITensorMPS.MPS)
        -> (framed::Vector{UnitaryGate}, fidelity::Float64)

Non-mutating counterpart of [`apply_single_qubit_rotations!`](@ref): returns a NEW circuit
carrying the product-frame fixup for `mps`, leaving `circuit` untouched.

**Use this, not the mutating form, anywhere the circuit is still being accumulated.**
`apply_single_qubit_rotations!` composes the fixup into `circuit[1:N-1]`, i.e. the FRONT of
the vector — which, after a `prepend!`, is the layer that was just added. Calling it once per
layer therefore leaves one stale fixup buried inside the circuit per iteration, and the
assembled circuit is no longer the algorithm's `U_D'...U_1'|0>` (arXiv:1908.07958 Eq. 12).
The mutating form is correct only when called exactly once, after the layer loop has finished
(see `compilers.jl`).

The returned circuit carries `length(mps)` extra single-qubit gates at the front. Two-qubit
gate counts are unaffected; count with `count_two_qubit`, never `length`.
"""
function with_single_qubit_rotations(circuit::Vector{UnitaryGate}, mps::ITensorMPS.MPS)
    single_qubit_layer, product_truncation_fidelity = single_qubit_frame(mps)
    return (vcat(single_qubit_layer, circuit), product_truncation_fidelity)
end


"""
    apply_single_qubit_rotations!(circuit::Vector{UnitaryGate}, mps::ITensorMPS.MPS)

Apply single-qubit gates to the first layer of `circuit` to fix up the final product state if necessary.
This is because the disentangling to χ=1 might produce a product state that isn't |00...0>.
Returns the fidelity of the final product state with the original MPS.

!!! warning "Call at most once per circuit"
    This mutates `circuit` in place, composing the fixup into its FRONT gates. Calling it
    again after more layers have been prepended does not replace the earlier fixup — it adds
    a second one and strands the first mid-circuit, silently corrupting the result. Inside a
    layer loop use [`with_single_qubit_rotations`](@ref) instead.
"""
function apply_single_qubit_rotations!(circuit::Vector{UnitaryGate}, mps::ITensorMPS.MPS)
    single_qubit_layer, product_truncation_fidelity = single_qubit_frame(mps)
    circuit[1] = compose(circuit[1], single_qubit_layer[1])
    for qubit in 2:length(single_qubit_layer)
        circuit[qubit-1] = compose(circuit[qubit-1], single_qubit_layer[qubit])
    end
    return product_truncation_fidelity
end


"""
    generate_layer!(mps::ITensorMPS.MPS; tolerance::Float64, max_bond_dim::Int, layer::Int)

Generates a single layer of the entangling circuit by truncating the MPS to χ=2, extracting the gates.
It's an in-place (!) function since it also applies the disentangling circuit to the working MPS.
"""
function generate_layer!(
    mps::ITensorMPS.MPS;
    tolerance::Float64,# Note: this is the truncation tolerance for layer generation
    layer_number::Int,
    max_bond_dim::Int,
    working_cutoff::Float64=1e-12,# Note: this is the working cutoff for the working MPS when applying gates
    backend::ComputeBackend=BackendCPU
)
    flag_disentangled = false
    entangling_layer = nothing
    mps_truncated = ITensorMPS.truncate(mps; maxdim=2, cutoff=tolerance) # Truncate the working MPS so layer can be extracted
    if ITensorMPS.maxlinkdim(mps_truncated) == 1
        println("MPS is disentangled to χ=1 within tolerance $(tolerance) after $(layer_number-1) layers.")
        flag_disentangled = true
    end
    if !flag_disentangled
        entangling_layer = truncated_preparation_circuit(mps_truncated) # Extract the gates for this layer
        disentangling_layer = reverse(dagger.(entangling_layer))
        mps = apply_circuit(disentangling_layer, mps; mixed_device=:coerce, conversion_precision=:preserve, cutoff=working_cutoff, maxdim=max_bond_dim) # Apply the gates to the working MPS
    end
    return mps, entangling_layer, flag_disentangled
end

# TODO: Add a Colbeck decomposition method for converting isometries to unitaries.
# Will also require a Gate format for storing both matrices and internal representation in decomposed form.
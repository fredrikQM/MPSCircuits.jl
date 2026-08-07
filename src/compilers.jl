abstract type AbstractCompilerProtocol end
struct DecomposeAllAnalytical <: AbstractCompilerProtocol end
struct IterativeDecomposeOptimizeLayer <: AbstractCompilerProtocol end
struct IterativeDecomposeOptimizeAll <: AbstractCompilerProtocol end

"""
    compile_mps_circuit(mps::ITensorMPS.MPS, protocol::DecomposeAllAnalytical; n_layers_max::Int, tolerance::Float64=1e-12, max_bond_dim::Int=ITensorMPS.maxlinkdim(mps))

Extract a preparation circuit from an MPS by `analytic disentangling`: iteratively truncating to χ=2, extracting a layer of gates, applying the inverse to the working MPS, and recording the circuit layer.
The process is repeated until the MPS is disentangled to χ=1 within the specified tolerance.
This is the `D_all` procedure from arXiv:2209.00595.
"""
function compile_mps_circuit(
    mps::ITensorMPS.MPS,
    protocol::DecomposeAllAnalytical;
    n_layers_max::Int,
    tolerance::Float64=1e-8,
    max_bond_dim::Int=ITensorMPS.maxlinkdim(mps),
    working_cutoff::Float64=1e-12,
    backend::Symbol=:auto,
    gpu_fallback::Bool=true,
    precision::Symbol=:fp32,
    transfer_policy::Symbol=:prefer_gpu_residency,
    progress::AbstractProgressTracker=NoProgressTracker(),
)
    backend_config = BackendConfig(; backend=backend_from_symbol(backend), gpu_fallback=gpu_fallback)
    resolved_backend = resolve_backend(backend_config)
    resolved_policy = transfer_policy_from_symbol(transfer_policy)
    run_t_start = time()
    layers_done = 0
    mps_work = ITensorMPS.dense(mps) # strip out the QN stuff if a fermionic state is supplied
    mps_work = place_on_backend_boundary(mps_work, resolved_backend, resolved_policy; precision=precision)
    preparation_circuit = UnitaryGate[] # blank array to hold the gates as we extract them

    record_progress_run_start!(progress; protocol=:decompose_all, backend=resolved_backend, precision=precision, n_layers_max=n_layers_max, n_iterations_per_layer=0)

    # Loop over layers: each one truncates to χ=2, extracts a layer, applies the inverse to the working MPS, and records the circuit layer.
    for layer in 1:n_layers_max
        layers_done = layer
        layer_t_start = time()
        record_progress_layer_start!(progress; layer=layer, total_layers=n_layers_max, current_circuit_len=length(preparation_circuit), mps_work_maxlinkdim=ITensorMPS.maxlinkdim(mps_work))
        mps_work, entangling_layer, flag_disentangled = generate_layer!(mps_work; tolerance=tolerance, layer_number=layer, max_bond_dim=max_bond_dim, working_cutoff=working_cutoff, backend=resolved_backend)
        record_progress_layer_generated!(progress; layer=layer, entangling_layer_len=(flag_disentangled || entangling_layer === nothing) ? 0 : length(entangling_layer), mps_work_maxlinkdim=ITensorMPS.maxlinkdim(mps_work), flag_disentangled=flag_disentangled)
        if flag_disentangled
            break # if we've already disentangled to χ=1, exit this loop
        else
            entangling_layer = place_on_backend_boundary(entangling_layer, resolved_backend, resolved_policy; precision=precision)
            prepend!(preparation_circuit, entangling_layer) # record the layer in the circuit
        end

        record_progress_layer_done!(progress; layer=layer, circuit_len=length(preparation_circuit), layer_elapsed_s=time() - layer_t_start)
    end

    product_state_truncation_fidelity = apply_single_qubit_rotations!(preparation_circuit, mps_work) # apply single-qubit rotations to fix up the final product state if necessary

    record_progress_run_done!(progress; total_layers_done=layers_done, total_elapsed_s=time() - run_t_start, final_circuit_len=length(preparation_circuit))

    return preparation_circuit
end

#=
"""
    compile_mps_circuit(mps::ITensorMPS.MPS, protocol::DecomposeAllAnalytical; n_layers_max::Int, tolerance::Float64=1e-12, max_bond_dim::Int=ITensorMPS.maxlinkdim(mps))

Extract a preparation circuit from an MPS by analytic disentangling and layer optimization.
This adds new layers and optimizes them one at a time.
This is the `Iter[D_i O_i]` procedure from arXiv:2209.00595. It isn't very good!

N.B.: This has been deprecated as it simply isn't very performant.
"""
function compile_mps_circuit(
    mps::ITensorMPS.MPS,
    protocol::IterativeDecomposeOptimizeLayer;
    n_layers_max::Int,
    n_iterations_per_layer::Int=10,
    tolerance::Float64=1e-8,
    max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(mps),
    working_cutoff::Float64=1e-12
)
    mps_clean = ITensorMPS.dense(mps) # strip out the QN stuff if a fermionic state is supplied
    mps_work = deepcopy(mps_clean)
    preparation_circuit = UnitaryGate[] # blank array to hold the gates as we extract them

    # Loop over layers: each one truncates to χ=2, extracts a layer, applies the inverse to the working MPS, and records the circuit layer.
    for layer in 1:n_layers_max
        mps_work, entangling_layer, flag_disentangled = generate_layer!(mps_work; tolerance=tolerance, layer_number=layer, max_bond_dim=max_bond_dim, working_cutoff=working_cutoff)
        if flag_disentangled
            product_state_truncation_fidelity = apply_single_qubit_rotations!(preparation_circuit, mps_work)
            break # if we've already disentangled to χ=1, exit this loop
        else
            prepend!(preparation_circuit, entangling_layer) # record the layer in the circuit
            # N.B.: we apply the single-qubit rotations here because we need to do optimization after w.r.t. fidelity
            product_state_truncation_fidelity = apply_single_qubit_rotations!(preparation_circuit, mps_work)
            for iteration in 1:n_iterations_per_layer
                replace_gates!(mps_clean, preparation_circuit, 1:(length(mps)-1), max_bond_dim=max_bond_dim, working_cutoff=working_cutoff) # optimize the first layer w.r.t. fidelity after each layer is added
            end
        end
    end

    return preparation_circuit
end
=#

"""
    compile_mps_circuit(mps::ITensorMPS.MPS, protocol::DecomposeAllAnalytical; n_layers_max::Int, tolerance::Float64=1e-12, max_bond_dim::Int=ITensorMPS.maxlinkdim(mps))

Extract a preparation circuit from an MPS by analytic disentangling and layer optimization.
This adds new layers and then optimizes -all- layers after each addition.
This is the `Iter[D_i O_all]` procedure from arXiv:2209.00595 and is typically the best option (albeit expensive).
"""
function compile_mps_circuit(
    mps::ITensorMPS.MPS,
    protocol::IterativeDecomposeOptimizeAll;
    optimization_protocol::AbstractProcrustesProtocol=TelescopingEnvironment(),
    n_layers_max::Int,
    n_iterations_per_layer::Int=10,
    tolerance::Float64=1e-8,
    max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(mps),
    working_cutoff::Float64=1e-12,
    backend::Symbol=:auto,
    gpu_fallback::Bool=true,
    precision::Symbol=:fp32,
    transfer_policy::Symbol=:prefer_gpu_residency,
    progress::AbstractProgressTracker=NoProgressTracker(),
)
    backend_config = BackendConfig(; backend=backend_from_symbol(backend), gpu_fallback=gpu_fallback)
    resolved_backend = resolve_backend(backend_config)
    resolved_policy = transfer_policy_from_symbol(transfer_policy)
    run_t_start = time()
    layers_done = 0
    mps_clean = ITensorMPS.dense(mps) # strip out the QN stuff if a fermionic state is supplied
    mps_clean = place_on_backend_boundary(mps_clean, resolved_backend, resolved_policy; precision=precision)
    mps_work = deepcopy(mps_clean)
    preparation_circuit = UnitaryGate[] # blank array to hold the gates as we extract them

    record_progress_run_start!(progress; protocol=:iterative_opt_all, backend=resolved_backend, precision=precision, n_layers_max=n_layers_max, n_iterations_per_layer=n_iterations_per_layer)

    # Loop over layers: each one truncates to χ=2, extracts a layer, applies the inverse to the working MPS, and records the circuit layer.
    for layer in 1:n_layers_max
        layers_done = layer
        layer_t_start = time()
        record_progress_layer_start!(progress; layer=layer, total_layers=n_layers_max, current_circuit_len=length(preparation_circuit), mps_work_maxlinkdim=ITensorMPS.maxlinkdim(mps_work))
        mps_work, entangling_layer, flag_disentangled = generate_layer!(mps_work; tolerance=tolerance, layer_number=layer, max_bond_dim=max_bond_dim, working_cutoff=working_cutoff, backend=resolved_backend)
        record_progress_layer_generated!(progress; layer=layer, entangling_layer_len=(flag_disentangled || entangling_layer === nothing) ? 0 : length(entangling_layer), mps_work_maxlinkdim=ITensorMPS.maxlinkdim(mps_work), flag_disentangled=flag_disentangled)
        if flag_disentangled
            product_state_truncation_fidelity = apply_single_qubit_rotations!(preparation_circuit, mps_work)
            record_progress_layer_done!(progress; layer=layer, circuit_len=length(preparation_circuit), layer_elapsed_s=time() - layer_t_start)
            break # if we've already disentangled to χ=1, exit this loop
        else
            entangling_layer = place_on_backend_boundary(entangling_layer, resolved_backend, resolved_policy; precision=precision)
            prepend!(preparation_circuit, entangling_layer) # record the layer in the circuit
            # N.B.: we apply the single-qubit rotations here because we need to do optimization after w.r.t. fidelity
            product_state_truncation_fidelity = apply_single_qubit_rotations!(preparation_circuit, mps_work)
            preparation_circuit = place_on_backend_boundary(preparation_circuit, resolved_backend, resolved_policy; precision=precision)
            layer_opt_total_steps = n_iterations_per_layer * length(preparation_circuit)
            record_progress_layer_opt_start!(progress; layer=layer, layer_opt_total_steps=layer_opt_total_steps)
            for iteration in 1:n_iterations_per_layer
                replace_gates!(
                    mps_clean,
                    preparation_circuit,
                    optimization_protocol,
                    1:length(preparation_circuit),
                    max_bond_dim=max_bond_dim,
                    working_cutoff=working_cutoff,
                    backend=resolved_backend,
                    progress=progress,
                    layer_number=layer,
                    iteration_index=iteration,
                    n_iterations_total=n_iterations_per_layer,
                ) # optimize all gates in the circuit w.r.t. fidelity after each layer is added
            end
            # Re-integrate the optimized gates: the next layer is decomposed from the CURRENT circuit's effective working MPS.
            mps_work = apply_circuit(
                reverse(dagger.(preparation_circuit)), deepcopy(mps_clean);
                mixed_device=:coerce, conversion_precision=:preserve,
                cutoff=working_cutoff, maxdim=max_bond_dim,
            )
            record_progress_layer_done!(progress; layer=layer, circuit_len=length(preparation_circuit), layer_elapsed_s=time() - layer_t_start)
        end
    end

    record_progress_run_done!(progress; total_layers_done=layers_done, total_elapsed_s=time() - run_t_start, final_circuit_len=length(preparation_circuit))

    return preparation_circuit
end


"""
    refine_mps_circuit(mps::ITensorMPS.MPS, circuit::Vector{UnitaryGate}; n_iterations::Int, optimization_protocol::AbstractProcrustesProtocol=TelescopingEnvironment(), max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(mps), working_cutoff::Float64=1e-12, backend::Symbol=:auto, gpu_fallback::Bool=true, precision::Symbol=:fp32, transfer_policy::Symbol=:prefer_gpu_residency, progress::AbstractProgressTracker=NoProgressTracker())

Refine an existing preparation circuit by sweeping `n_iterations` times over all gates
with Procrustes updates (`replace_gates!`). The input MPS and circuit are co-located on
the resolved backend before optimization. Progress callbacks follow the same event model
as compiler optimization paths.
"""
function refine_mps_circuit(
    mps::ITensorMPS.MPS,
    circuit::Vector{UnitaryGate};
    n_iterations::Int,
    optimization_protocol::AbstractProcrustesProtocol=TelescopingEnvironment(),
    max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(mps),
    working_cutoff::Float64=1e-12,
    backend::Symbol=:auto,
    gpu_fallback::Bool=true,
    precision::Symbol=:fp32,
    transfer_policy::Symbol=:prefer_gpu_residency,
    progress::AbstractProgressTracker=NoProgressTracker(),
)
    if n_iterations < 1
        throw(ArgumentError("n_iterations must be >= 1, got $(n_iterations)."))
    end
    if isempty(circuit)
        throw(ArgumentError("circuit cannot be empty."))
    end

    backend_config = BackendConfig(; backend=backend_from_symbol(backend), gpu_fallback=gpu_fallback)
    resolved_backend = resolve_backend(backend_config)
    resolved_policy = transfer_policy_from_symbol(transfer_policy)
    run_t_start = time()

    mps_clean = ITensorMPS.dense(mps)
    mps_clean = place_on_backend_boundary(mps_clean, resolved_backend, resolved_policy; precision=precision)
    refined_circuit = deepcopy(circuit)
    refined_circuit = place_on_backend_boundary(refined_circuit, resolved_backend, resolved_policy; precision=precision)

    record_progress_run_start!(progress; protocol=:refine_circuit, backend=resolved_backend, precision=precision, n_layers_max=1, n_iterations_per_layer=n_iterations)
    record_progress_layer_start!(progress; layer=1, total_layers=1, current_circuit_len=length(refined_circuit), mps_work_maxlinkdim=ITensorMPS.maxlinkdim(mps_clean))
    layer_opt_total_steps = n_iterations * length(refined_circuit)
    record_progress_layer_opt_start!(progress; layer=1, layer_opt_total_steps=layer_opt_total_steps)

    layer_t_start = time()
    for iteration in 1:n_iterations
        replace_gates!(
            mps_clean,
            refined_circuit,
            optimization_protocol,
            1:length(refined_circuit),
            max_bond_dim=max_bond_dim,
            working_cutoff=working_cutoff,
            backend=resolved_backend,
            progress=progress,
            layer_number=1,
            iteration_index=iteration,
            n_iterations_total=n_iterations,
        )
    end

    record_progress_layer_done!(progress; layer=1, circuit_len=length(refined_circuit), layer_elapsed_s=time() - layer_t_start)
    record_progress_run_done!(progress; total_layers_done=1, total_elapsed_s=time() - run_t_start, final_circuit_len=length(refined_circuit))

    return refined_circuit
end


# TODO: Implement Floyd's variational recompression approach?
# TODO: Implement Bohun-style CNOT saving approach
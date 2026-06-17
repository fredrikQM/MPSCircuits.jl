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

abstract type AbstractProgressTracker end

struct NoProgressTracker <: AbstractProgressTracker end

mutable struct CallbackProgressTracker <: AbstractProgressTracker
    callback::Function
    enabled::Bool
    emit_every::Int
    include_env_norm::Bool
    run_started_at::Float64
    layers_done::Int
    run_opt_completed_steps::Int
    run_opt_est_total_steps::Int
end

function CallbackProgressTracker(
    callback::Function;
    enabled::Bool=true,
    emit_every::Int=1,
    include_env_norm::Bool=false,
)
    return CallbackProgressTracker(
        callback,
        enabled,
        max(1, emit_every),
        include_env_norm,
        0.0,
        0,
        0,
        0,
    )
end

@inline progress_enabled(::NoProgressTracker) = false
@inline progress_enabled(tracker::CallbackProgressTracker) = tracker.enabled

@inline progress_include_env_norm(::NoProgressTracker) = false
@inline progress_include_env_norm(tracker::CallbackProgressTracker) = tracker.include_env_norm

@inline progress_total_opt_steps(::NoProgressTracker) = 0
@inline progress_total_opt_steps(tracker::CallbackProgressTracker) = tracker.run_opt_completed_steps

@inline function _emit_progress(tracker::NoProgressTracker, ::Symbol, ::NamedTuple)
    return nothing
end

@inline function _emit_progress(tracker::CallbackProgressTracker, event::Symbol, payload::NamedTuple)
    if tracker.enabled
        tracker.callback(event, payload)
    end
    return nothing
end

@inline function record_progress_run_start!(tracker::NoProgressTracker; kwargs...)
    return nothing
end

function record_progress_run_start!(
    tracker::CallbackProgressTracker;
    protocol::Symbol,
    backend,
    precision::Symbol,
    n_layers_max::Int,
    n_iterations_per_layer::Int,
)
    if !tracker.enabled
        return nothing
    end
    tracker.run_started_at = time()
    tracker.layers_done = 0
    tracker.run_opt_completed_steps = 0
    tracker.run_opt_est_total_steps = 0
    _emit_progress(tracker, :run_start, (
        protocol=protocol,
        backend=backend,
        precision=precision,
        n_layers_max=n_layers_max,
        n_iterations_per_layer=n_iterations_per_layer,
    ))
    return nothing
end

@inline function record_progress_layer_start!(tracker::NoProgressTracker; kwargs...)
    return nothing
end

function record_progress_layer_start!(
    tracker::CallbackProgressTracker;
    layer::Int,
    total_layers::Int,
    current_circuit_len::Int,
    mps_work_maxlinkdim::Int,
)
    if !tracker.enabled
        return nothing
    end
    _emit_progress(tracker, :layer_start, (
        layer=layer,
        total_layers=total_layers,
        current_circuit_len=current_circuit_len,
        mps_work_maxlinkdim=mps_work_maxlinkdim,
    ))
    return nothing
end

@inline function record_progress_layer_generated!(tracker::NoProgressTracker; kwargs...)
    return nothing
end

function record_progress_layer_generated!(
    tracker::CallbackProgressTracker;
    layer::Int,
    entangling_layer_len::Int,
    mps_work_maxlinkdim::Int,
    flag_disentangled::Bool,
)
    if !tracker.enabled
        return nothing
    end
    _emit_progress(tracker, :layer_generated, (
        layer=layer,
        entangling_layer_len=entangling_layer_len,
        mps_work_maxlinkdim=mps_work_maxlinkdim,
        flag_disentangled=flag_disentangled,
    ))
    return nothing
end

@inline function record_progress_layer_opt_start!(tracker::NoProgressTracker; kwargs...)
    return nothing
end

function record_progress_layer_opt_start!(
    tracker::CallbackProgressTracker;
    layer::Int,
    layer_opt_total_steps::Int,
)
    if !tracker.enabled
        return nothing
    end
    tracker.run_opt_est_total_steps += layer_opt_total_steps
    _emit_progress(tracker, :layer_opt_start, (
        layer=layer,
        layer_opt_total_steps=layer_opt_total_steps,
        run_opt_completed_steps=tracker.run_opt_completed_steps,
        run_opt_est_total_steps=tracker.run_opt_est_total_steps,
    ))
    return nothing
end

@inline function record_progress_layer_done!(tracker::NoProgressTracker; kwargs...)
    return nothing
end

function record_progress_layer_done!(
    tracker::CallbackProgressTracker;
    layer::Int,
    circuit_len::Int,
    layer_elapsed_s::Float64,
)
    if !tracker.enabled
        return nothing
    end
    tracker.layers_done = layer
    _emit_progress(tracker, :layer_done, (
        layer=layer,
        circuit_len=circuit_len,
        layer_elapsed_s=layer_elapsed_s,
    ))
    return nothing
end

@inline function record_progress_opt_step!(tracker::NoProgressTracker; kwargs...)
    return nothing
end

function record_progress_opt_step!(
    tracker::CallbackProgressTracker;
    layer::Int,
    iter_in_layer::Int,
    iter_total_for_layer::Int,
    gate_index::Int,
    gate_count::Int,
    gate_step_in_iteration::Int,
    ket_maxlinkdim::Int,
    bra_maxlinkdim::Int,
    env_norm::Float64,
)
    tracker.run_opt_completed_steps += 1
    if !tracker.enabled
        return nothing
    end
    if tracker.run_opt_completed_steps % tracker.emit_every != 0
        return nothing
    end
    layer_step_done = (iter_in_layer - 1) * gate_count + gate_step_in_iteration
    layer_step_total = iter_total_for_layer * gate_count
    _emit_progress(tracker, :opt_step_done, (
        layer=layer,
        iter_in_layer=iter_in_layer,
        iter_total_for_layer=iter_total_for_layer,
        gate_index=gate_index,
        gate_count=gate_count,
        layer_step_done=layer_step_done,
        layer_step_total=layer_step_total,
        run_step_done=tracker.run_opt_completed_steps,
        run_step_est_total=tracker.run_opt_est_total_steps,
        ket_maxlinkdim=ket_maxlinkdim,
        bra_maxlinkdim=bra_maxlinkdim,
        env_norm=env_norm,
    ))
    return nothing
end

@inline function record_progress_run_done!(tracker::NoProgressTracker; kwargs...)
    return nothing
end

function record_progress_run_done!(
    tracker::CallbackProgressTracker;
    total_layers_done::Int,
    total_elapsed_s::Float64,
    final_circuit_len::Int,
)
    if !tracker.enabled
        return nothing
    end
    _emit_progress(tracker, :run_done, (
        total_layers_done=total_layers_done,
        total_opt_steps_done=tracker.run_opt_completed_steps,
        total_elapsed_s=total_elapsed_s,
        final_circuit_len=final_circuit_len,
    ))
    return nothing
end

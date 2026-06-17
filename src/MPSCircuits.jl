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

module MPSCircuits

import LinearAlgebra
import ITensors
import ITensorMPS
import Random

export compile_mps_circuit
export refine_mps_circuit
export evaluate_circuit_fidelity
export write_circuit_json

include("gates.jl")
include("circuits.jl")
include("backend.jl")
include("progress.jl")
include("helper.jl")
include("analytic_disentangling.jl")
include("procrustes.jl")
include("compilers.jl")
include("export.jl")

module Chemistry

# export fcidump_to_mpo
# export store_mps
# export load_mps

include("chemistry.jl")

end

end
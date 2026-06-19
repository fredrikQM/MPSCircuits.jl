module MPSCircuitsCUDA

using MPSCircuits
import CUDA
import ITensors
import ITensorMPS

import MPSCircuits: _cast_backend_precision, _is_gpu_array_impl, _to_backend_gpu

cuda_is_functional() = CUDA.functional()
_is_gpu_array_impl(::CUDA.CuArray) = true

function _itensor_to_backend(tensor::ITensors.ITensor; precision::Symbol, to_gpu::Bool)
	inds = ITensors.inds(tensor)
	dense_array = ITensors.array(tensor, inds...)
	cast_array = _cast_backend_precision(dense_array, precision)
	backend_array = if to_gpu
		cast_array isa CUDA.CuArray ? cast_array : CUDA.CuArray(cast_array)
	else
		Array(cast_array)
	end
	return ITensors.itensor(backend_array, inds...)
end

function _to_backend_gpu(tensor::ITensors.ITensor; precision::Symbol=:fp32)
	return _itensor_to_backend(tensor; precision=precision, to_gpu=true)
end

function _to_backend_gpu(mps::ITensorMPS.MPS; precision::Symbol=:fp32)
	converted = deepcopy(mps)
	for n in 1:length(converted)
		converted[n] = _to_backend_gpu(converted[n]; precision=precision)
	end
	return converted
end

end
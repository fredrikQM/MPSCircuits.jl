using Test
using MPSCircuits, ITensors, ITensorMPS, LinearAlgebra

@testset "MPSCircuits.jl" begin
    include("compilation_tests.jl")
end
import ITensors
import ITensorMPS
import HDF5

const fcidump_pattern = r"\-?\d{1,}\.?\d{0,}e?[\-\+]?\d{0,}"
const chemistry_debug = false
const chemistry_debug_detailed = false

"""
    fcidump_to_mpo(fcidumpfile::String; atol::Float64=1e-12, sitetype::String="Fermion", encoding::String="JW")

Parse an FCIDUMP file and build an MPO Hamiltonian plus a closed-shell Hartree-Fock initial state.
Returns `(H, s, init_state)` where `H` is an `ITensorMPS.MPO`, `s` is the site index vector,
and `init_state` is the integer occupation vector for the initial guess.
"""
function fcidump_to_mpo(
    fcidumpfile::String;
    atol::Float64=1e-12,
    sitetype::String="Fermion",
    encoding::String="JW",
)
    nel::Int64 = 0
    norb::Int64 = 0
    ns::Int64 = 0

    hamiltonian = ITensors.OpSum()

    if sitetype == "Qubit"
        if encoding == "JW"
            sitetype = "Fermion"
        else
            throw(ArgumentError("Encoding $encoding not yet implemented!"))
        end
    end

    for (i, line) in enumerate(eachline(fcidumpfile))
        matches = getfield.(eachmatch(fcidump_pattern, line), :match)
        if i == 1
            norb = parse(Int64, matches[1])
            nel = parse(Int64, matches[2])
            ns = parse(Int64, matches[4])
            if chemistry_debug
                println("nel=", nel)
                println("norb=", norb)
                println("ns=", ns)
            end
        elseif i > 4
            coeff = parse(Float64, matches[1])
            i_idx = parse(Int64, matches[2])
            j_idx = parse(Int64, matches[3])
            k_idx = parse(Int64, matches[4])
            l_idx = parse(Int64, matches[5])
            if abs(coeff) < atol
                continue
            end

            if k_idx == 0 && l_idx == 0
                if i_idx == 0 && j_idx == 0
                    ITensorMPS.add!(hamiltonian, coeff, "Id", 1)
                else
                    if sitetype == "Electron"
                        ITensorMPS.add!(hamiltonian, coeff, "c†↑", i_idx, "c↑", j_idx)
                        ITensorMPS.add!(hamiltonian, coeff, "c†↓", i_idx, "c↓", j_idx)
                        if i_idx != j_idx
                            ITensorMPS.add!(hamiltonian, coeff, "c†↑", j_idx, "c↑", i_idx)
                            ITensorMPS.add!(hamiltonian, coeff, "c†↓", j_idx, "c↓", i_idx)
                        end
                    elseif sitetype == "Fermion"
                        ITensorMPS.add!(hamiltonian, coeff, "c†", 2*i_idx-1, "c", 2*j_idx-1)
                        ITensorMPS.add!(hamiltonian, coeff, "c†", 2*i_idx, "c", 2*j_idx)
                        if i_idx != j_idx
                            ITensorMPS.add!(hamiltonian, coeff, "c†", 2*j_idx-1, "c", 2*i_idx-1)
                            ITensorMPS.add!(hamiltonian, coeff, "c†", 2*j_idx, "c", 2*i_idx)
                        end
                    else
                        throw(ArgumentError("Site type $sitetype not yet implemented!"))
                    end
                end
            else
                labels = twoterm_labels(i_idx, j_idx, k_idx, l_idx)
                for (p, q, r, s_) in labels
                    if !(r == q || p == s_)
                        if sitetype == "Electron"
                            ITensorMPS.add!(hamiltonian, 0.5 * coeff, "c†↑", r, "c†↑", q, "c↑", p, "c↑", s_)
                            ITensorMPS.add!(hamiltonian, 0.5 * coeff, "c†↓", r, "c†↓", q, "c↓", p, "c↓", s_)
                        elseif sitetype == "Fermion"
                            ITensorMPS.add!(hamiltonian, 0.5 * coeff, "c†", 2*r-1, "c†", 2*q-1, "c", 2*p-1, "c", 2*s_-1)
                            ITensorMPS.add!(hamiltonian, 0.5 * coeff, "c†", 2*r, "c†", 2*q, "c", 2*p, "c", 2*s_)
                        end
                    end
                    if sitetype == "Electron"
                        ITensorMPS.add!(hamiltonian, 0.5 * coeff, "c†↑", r, "c†↓", q, "c↓", p, "c↑", s_)
                        ITensorMPS.add!(hamiltonian, 0.5 * coeff, "c†↓", r, "c†↑", q, "c↑", p, "c↓", s_)
                    elseif sitetype == "Fermion"
                        ITensorMPS.add!(hamiltonian, 0.5 * coeff, "c†", 2*r-1, "c†", 2*q, "c", 2*p, "c", 2*s_-1)
                        ITensorMPS.add!(hamiltonian, 0.5 * coeff, "c†", 2*r, "c†", 2*q-1, "c", 2*p-1, "c", 2*s_)
                    end
                end
            end
        end
    end

    if chemistry_debug_detailed
        println(hamiltonian)
    end

    init_state = Int64[]
    if sitetype == "Electron"
        nsites = norb
        for _ in 1:(nel ÷ 2)
            push!(init_state, 4)
        end
        for _ in (nel ÷ 2 + 1):nsites
            push!(init_state, 1)
        end
    elseif sitetype == "Fermion" || sitetype == "Qubit"
        nsites = 2 * norb
        for _ in 1:nel
            push!(init_state, 2)
        end
        for _ in (nel + 1):nsites
            push!(init_state, 1)
        end
    else
        throw(ArgumentError("Site type $sitetype not yet implemented!"))
    end

    if chemistry_debug
        println("Initial state:", init_state)
    end

    s = ITensorMPS.siteinds(sitetype, nsites; conserve_qns=true)
    H = ITensorMPS.MPO(hamiltonian, s)

    return H, s, init_state
end

"""
    opt_mps(H::ITensorMPS.MPO, psi0::ITensorMPS.MPS; nsweeps::Int64=10, maxdim::Int64=100, cutoff::Float64=1e-8, truncation::Float64=1e-8)

Run DMRG on the given MPO with an initial MPS guess, using the specified convergence paramters, and return the optimized state.
The routine also truncates the resulting MPS and verifies the truncated energy before returning.
"""
function opt_mps(
    H::ITensorMPS.MPO,
    psi0::ITensorMPS.MPS;
    nsweeps::Int64=10,
    maxdim::Int64=100,
    cutoff::Float64=1e-8,
    truncation::Float64=1e-8,
)
    energy, psi = ITensorMPS.dmrg(
        H,
        psi0;
        nsweeps=nsweeps,
        maxdim=maxdim,
        cutoff=cutoff,
        eigsolve_krylovdim=10,
        noise=1e-8,
        eigsolve_tol=1e-8,
    )
    println("Final energy = ", energy)
    println("Bond dimensions before truncation:", ITensorMPS.linkdims(psi))

    psi_trunc = ITensorMPS.truncate(psi, cutoff=truncation)
    energy_truncated = ITensorMPS.inner(psi_trunc', H, psi_trunc)
    println("Truncated energy = ", energy_truncated)

    if abs(energy_truncated - energy) > 1.6e-3
        println("Truncated energy differs too much, returning full MPS.")
        return energy, psi
    end

    println("Returning truncated MPS.")
    return energy_truncated, psi_trunc
end

"""
    store_mps(psi::ITensorMPS.MPS, fnam::String)

Write an MPS to an HDF5 file at the specified path.
"""
function store_mps(psi::ITensorMPS.MPS, fnam::String)
    f = HDF5.h5open(fnam, "w")
    HDF5.write(f, "psi", psi)
    HDF5.close(f)
end

"""
    load_mps(fnam::String)

Read an MPS previously written with `store_mps`.
"""
function load_mps(fnam::String)
    f = HDF5.h5open(fnam, "r")
    psi = HDF5.read(f, "psi", ITensorMPS.MPS)
    HDF5.close(f)
    return psi
end

"""
    twoterm_labels(i::Int64, j::Int64, k::Int64, l::Int64)

Return a list of unique index permutations of four indices.
"""
function twoterm_labels(i::Int64, j::Int64, k::Int64, l::Int64)
    labels = [(i, j, k, l)]
    if !((i, j, l, k) in labels)
        push!(labels, (i, j, l, k))
    end
    if !((j, i, k, l) in labels)
        push!(labels, (j, i, k, l))
    end
    if !((j, i, l, k) in labels)
        push!(labels, (j, i, l, k))
    end
    if !((k, l, i, j) in labels)
        push!(labels, (k, l, i, j))
    end
    if !((l, k, i, j) in labels)
        push!(labels, (l, k, i, j))
    end
    if !((k, l, j, i) in labels)
        push!(labels, (k, l, j, i))
    end
    if !((l, k, j, i) in labels)
        push!(labels, (l, k, j, i))
    end
    return labels
end

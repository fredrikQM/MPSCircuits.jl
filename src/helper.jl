# Magic basis: following convention of Tucci's paper (arXiv:quant-ph/0507171), which differs from Vatan's (arXiv:quant-ph/0308006).
const M_magic = 1 / √2 * [
    1 0 0 im
    0 im 1 0
    0 im -1 0
    1 0 0 -im
]

const CNOT1 = [
    1 0 0 0
    0 1 0 0
    0 0 0 1
    0 0 1 0
]

const CNOT2 = [
    1 0 0 0
    0 0 0 1
    0 0 1 0
    0 1 0 0
]

@inline function Rz(θ::Real)
    h = θ / 2
    return ComplexF64[
        exp(-im * h) 0.0im;
        0.0im exp(im * h)
    ]
end

@inline function Ry(θ::Real)
    h = θ / 2
    c, s = cos(h), sin(h)
    return ComplexF64[
        c -s;
        s c
    ]
end

"""
    full_positive_qr(matrix::AbstractMatrix)

Compute a 'full' (not thin) QR decomposition of `matrix` and return the unitary Q factor.
For our purposes, since the input will be an isometry, the R factor will have diagonal elements +/- 1.
We can ensure uniqueness by forcing R to be positive, ensuring that Q is itself a unitary padding of the isometry.
"""
function full_positive_qr(matrix::AbstractMatrix)
    matrix_cpu = to_tiny_kernel_cpu(matrix)
    F = LinearAlgebra.qr(matrix_cpu) # Perform the QR decomposition
    Q_full = F.Q * LinearAlgebra.I # full unitary
    R_thin = F.R # 'thin' R
    #Force R to be positive so that it doesn't mess up the phases
    diag_signs = sign.(LinearAlgebra.diag(R_thin))
    diag_signs[diag_signs.==0] .= 1.0 # Handle zero entries by treating them as positive, I think there shouldn't be any anyway
    Q_full[:, 1:length(diag_signs)] .*= diag_signs' # only map onto the image cols

    return Q_full
end


"""
    evaluate_circuit_fidelity(circuit::Vector, target_mps::ITensorMPS.MPS; cutoff::Float64=1e-12, max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(target_mps))

Estimate fidelity between `target_mps` and the state prepared by `circuit` from |00...0>.
This routine applies daggered gates from right to left onto `target_mps` and then
computes overlap with |00...0>, which keeps intermediate bond dimensions lower when
the circuit approximately disentangles the target state.
"""
function evaluate_circuit_fidelity(
    circuit::Vector{<:AbstractGate},
    target_mps::ITensorMPS.MPS;
    cutoff::Float64=1e-12,
    max_bond_dim::Int=2 * ITensorMPS.maxlinkdim(target_mps),
)
    if length(circuit) == 0
        error("circuit cannot be empty.")
    end

    mps_work = ITensorMPS.dense(target_mps) # strip out the QN stuff if a fermionic state is supplied
    mps_backend = backend_of(mps_work)

    sites = ITensorMPS.siteinds(mps_work)
    N = length(sites)
    initial_state = ["0" for _ in 1:N]
    psi0 = ITensorMPS.MPS(sites, initial_state)
    psi0 = to_backend(psi0, mps_backend; precision=:preserve)

    bra_work = deepcopy(mps_work)
    for gate in circuit[end:-1:1]
        bra_work = apply_gate(dagger(gate), bra_work; mixed_device=:coerce, conversion_precision=:preserve, cutoff=cutoff, maxdim=max_bond_dim)
    end

    overlap = ITensorMPS.inner(bra_work, psi0)

    return abs2(overlap) / ITensorMPS.inner(mps_work, mps_work)
end

"""
    untensor(K::AbstractMatrix)

Given a factorizable 4x4 matrix K, return the two 2x2 unitaries U_A and U_B such that K = kron(U_A, U_B).
Note: this doesn't check that K is factorizable, so you had better know this a priori.
"""
function untensor(K::AbstractMatrix)
    K_cpu = to_tiny_kernel_cpu(K)
    matrix_blocks = [K_cpu[1:2, 1:2], K_cpu[1:2, 3:4],
        K_cpu[3:4, 1:2], K_cpu[3:4, 3:4]]
    max_block = matrix_blocks[argmax(LinearAlgebra.norm.(matrix_blocks))] # use max (Frobenius) norm block for stability

    U_B = max_block / sqrt(Complex(LinearAlgebra.det(max_block))) # use max block as U_B (up to phase)

    # now extract U_A by projecting K onto U_B
    U_A = zeros(ComplexF64, 2, 2)
    U_A = [LinearAlgebra.tr(U_B' * matrix_blocks[(i-1)*2+j]) / 2.0 for i in 1:2, j in 1:2]

    return U_A, U_B
end

"""
    zyz_decomposition(U::AbstractMatrix)

Given a 2x2 unitary U, return the angles (theta1, theta2, theta3) such that U = Rz(theta3) * Ry(theta2) * Rz(theta1) up to global phase.
"""
function zyz_decomposition(U::AbstractMatrix)
    U_cpu = to_tiny_kernel_cpu(U)
    U_SU2 = U_cpu / sqrt(Complex(LinearAlgebra.det(U_cpu))) # factor out global phase
    theta2 = 2 * acos(min(1.0, abs(U_SU2[1, 1]))) # clamp to avoid domain error

    # Gimbal-locked edge cases
    if isapprox(theta2, 0.0, atol=1e-7)
        theta3 = 2 * angle(U_SU2[2, 2])
        theta1 = 0.0
    elseif isapprox(theta2, pi, atol=1e-7)
        theta3 = 2 * angle(U_SU2[2, 1])
        theta1 = 0.0
        # Standard case
    else
        phi_22 = angle(U_SU2[2, 2])
        phi_21 = angle(U_SU2[2, 1])

        theta3 = phi_22 + phi_21
        theta1 = phi_22 - phi_21
    end

    return theta1, theta2, theta3
end


"""
    cartan_KAK_decomposition(X::AbstractMatrix)

Given a 4x4 unitary X, return the parameters of its Cartan KAK decomposition: α, β, γ for the core and A_L1, A_L2, A_R1, A_R2 for the boundary SU(2)s.
Uses the approach of Tucci (arXiv:quant-ph/0507171).
"""
function cartan_KAK_decomposition(X::AbstractMatrix)
    X_cpu = to_tiny_kernel_cpu(X)
    # Transform to magic basis (Tucci, Eq. 35)
    X_prime = M_magic' * X_cpu * M_magic
    # Setup per Tucci Lemma 3
    X_R = real.(X_prime)
    X_I = imag.(X_prime)
    # Per Tucci Eq. 16
    svd_A = LinearAlgebra.svd(X_R)
    U_A = svd_A.U
    V_A = svd_A.Vt'  # V_A is V (since svd returns Vt)
    # Tucci Eq. 17
    B_prime = U_A' * X_I * V_A
    # Determine the rank of X_R to partition B' into G and H blocks (up to tolerance)
    tol = 1e-12 * maximum(svd_A.S)
    r = count(x -> x > tol, svd_A.S)
    U_inner = zeros(Float64, 4, 4)
    V_inner = zeros(Float64, 4, 4)
    # Diagonalize symmetric G block (Tucci, Eq. 24)
    if r > 0
        G = B_prime[1:r, 1:r]
        # Wrap in Symmetric to enforce strictly real orthogonal eigenvectors
        eigen_G = LinearAlgebra.eigen(LinearAlgebra.Symmetric(G))
        P = eigen_G.vectors
        U_inner[1:r, 1:r] = P
        V_inner[1:r, 1:r] = P
    end
    # SVD the H-block (Tucci, Eq. 25)
    if r < 4
        H = B_prime[r+1:end, r+1:end]
        svd_H = LinearAlgebra.svd(H)
        U_inner[r+1:end, r+1:end] = svd_H.U
        V_inner[r+1:end, r+1:end] = svd_H.Vt'
    end
    # Final orthogonal matrices per Tucci Eq. 26
    Q_L = U_A * U_inner
    Q_R = V_A * V_inner
    # Force Q_L and Q_R into SO(4) (fix arbitrary phases in decomposition)
    if LinearAlgebra.det(Q_L) < 0
        Q_L[:, 4] = -Q_L[:, 4]
    end
    if LinearAlgebra.det(Q_R) < 0
        Q_R[:, 4] = -Q_R[:, 4]
    end
    # Diagonal core per Tucci Eq. 27/28
    D_R_diag = LinearAlgebra.diag(Q_L' * X_R * Q_R)
    D_I_diag = LinearAlgebra.diag(Q_L' * X_I * Q_R)
    D_diag = LinearAlgebra.Diagonal(LinearAlgebra.normalize.(D_R_diag .+ im .* D_I_diag))
    # Now leave the magic basis. K_L and K_R will be SU(2)⊗SU(2) unitaries so we can untensor them.
    K_L = M_magic * Q_L * M_magic'
    K_R = M_magic * Q_R' * M_magic'
    A_L1, A_L2 = untensor(K_L)
    A_R1, A_R2 = untensor(K_R)
    # Now extract the angles for the exp(i * (α X⊗X + β Y⊗Y + γ Z⊗Z)) core
    phases = angle.(LinearAlgebra.diag(D_diag))
    # global_phase = ( phases[1] + phases[2] + phases[3] + phases[4]) / 4
    α = (phases[1] + phases[2] - phases[3] - phases[4]) / 4
    β = (-phases[1] + phases[2] - phases[3] + phases[4]) / 4
    γ = (phases[1] - phases[2] - phases[3] + phases[4]) / 4
    # Integrate the boundary SU(2)s that sit outside the core to their neighbours in the K-layers (outermost Rzs in Fig. 6 of Vatan)
    A_R2 = Rz(-π / 2) * A_R2
    A_L1 = A_L1 * Rz(π / 2)
    return α, β, γ, A_L1, A_L2, A_R1, A_R2
end
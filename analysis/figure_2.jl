envpath = normpath(joinpath((@__DIR__, "../envs/env1")))
srcpath = normpath(joinpath((@__DIR__, "../src/")))
using Pkg; Pkg.activate(envpath)

using SparseArrays
using DifferentialEquations
using LinearAlgebra
using Catalyst
using Distributions
using FiniteStateProjection: FSPSystem, singleindices, pairedindices, netstoichmat

include(srcpath*"plotting.jl")

# Function for computing waiting times using a modification of the Finite State Projection method, 
# developed in Ham et al,, A stochastic vs deterministic perspective on the timing of cellular events, Nat Comms, 15, (2024).  
function create_sparsematrix(sys::FSPSystem, dims::NTuple, ps, t; idx_filter = x -> true)
    Ntot = prod(dims)
    lind = LinearIndices(sys.ih, dims)

    I = Int[]
    J = Int[]
    V = Float64[]

    predsize = Ntot * (length(Catalyst.get_eqs(sys.rs)) + 1)

    sizehint!(I, predsize)
    sizehint!(J, predsize)
    sizehint!(V, predsize)

    for idx_cart in singleindices(sys.ih, dims)
        idx_lin = lind[idx_cart]
        push!(I, idx_lin)
        push!(J, idx_lin)

        if !idx_filter(idx_cart)
            # Using -1 instead of 1 here ensures that all eigenvalues are negative
            push!(V, -1.)
            continue
        end

        rate = 0.0
        for rf in sys.rfs
            rate -= rf(idx_cart, t, ps...)
        end

        push!(V, -rate)
    end

    S::Matrix{Int64} = netstoichmat(sys.rs)
    for (i, rf) in enumerate(sys.rfs)
        for (idx_cin, idx_cout) in pairedindices(sys.ih, dims, CartesianIndex(S[:,i]...))
            if !idx_filter(idx_cin) || !idx_filter(idx_cout)
                continue
            end

            idx_lin = lind[idx_cin]
            idx_lout = lind[idx_cout]
            push!(I, idx_lout)
            push!(J, idx_lin)

            rate = rf(idx_cin, t, ps...)
            push!(V, -rate)
        end
    end

    I, J, V
end


# Toggle triad model
# 0 is unbound (leak), 1 is ON and 2 is blocked
@parameters kb1, kb2, kbb1, kbb2, k1, k2, d1, d2, mu11, mu22, mu12, mu21, l11, l12, l21, l22

rs = @reaction_network begin
    kb1*(X==0), 0 --> M1                  #leakage from gene 1 native state X_10
    kb2*(Y==0), 0 --> M2                  #leakage from gene 2 native state X_20
    kbb1*(X==2), 0 --> M1                  #leakage from gene 1 native state X_10
    kbb2*(Y==2), 0 --> M2                  #leakage from gene 2 native state X_20
    k1*(X==1), 0 --> M1                   #mRNA synthesis from gene 1 active state X_11
    k2*(Y==1), 0 --> M2                   #mRNA synthesis from gene 2 active state X_22

    d1, M1 --> 0                          #mRNA degradation for M1
    d2, M2 --> 0                          #mRNA degradation for M2
    mu11*(X==1), X ⇒ M1                   #M1 unbinds 0.1 X_11--> X_10 + M1
    mu22*(Y==1), Y ⇒ M2                   #M2 unbinds 0.1
    mu12*(X==2), 2X ⇒ M2                  #M2 unbinds 10
    mu21*(Y==2), 2Y ⇒ M1                  #M1 unbinds 10
    l11*(X==0), M1 --> X                  #gene 1 self enhancement
    l12* (X==0), M2 --> 2X                #gene 2 blocks gene 1
    l21* (Y==0), M1 --> 2Y                #gene 1 blocks gene 2
    l22* (Y==0), M2 --> Y                 #gene 2 self enhancement
end 

g1_target = 0
g2_target = 29
charfunc_Y(idx) = idx[1] - 1 == g1_target && idx[2] - 1 == g2_target  # Stop when we have 29 copies of P1
Mmax = 120
num_entries = Mmax * Mmax * 3 * 3
u0_wt = zeros(Mmax, Mmax, 3, 3)
u0_wt[1, 1, 2, 2] = 1.0                 
fspsys_wt = FSPSystem(rs)

signal_strengths = 1:1:10
wait_times = Vector{Float64}(undef, length(signal_strengths))

for (i, signal) in enumerate(signal_strengths)
    k11 = 30 
    K = k11 + signal
    decay = K / k11
    leak = 0.1 + signal

    p = [kb1 => leak 
         kb2 => 0.1
         kbb1 => leak 
         kbb2 => 0.1 
         k1 => K
         k2 => 30
         d1 => decay
         d2 => 1
         mu11 => 0.1
         mu22 => 0.1
         mu12 => 10
         mu21 => 10
         l11 => 1
         l12 => 1
         l21 => 1
         l22 => 1  ]

    ## Compute expected waiting times
    A_I, A_J, A_V = create_sparsematrix(fspsys_wt, size(u0_wt), last.(p), 0.; idx_filter = !charfunc_Y)
    AYT = SparseArrays.sparse(A_J, A_I, A_V)
    sol = AYT \ ones(length(u0_wt))
    sol_reshape = reshape(sol, (Mmax, Mmax, 3, 3))
    wait_times[i] = sol_reshape[29, 1, 1, 1]
end 

f = CairoMakie.Figure(size = (size_pt[1]*1.2, size_pt[2]*1.1), figure_padding = 2)
ax = Axis(f[1, 1], xlabel = "Signal strength", ylabel = "log₁₀(MFPT)")
CairoMakie.lines!(ax, signal_strengths, log10.(wait_times), linewidth=1.5, color=red_col)
f
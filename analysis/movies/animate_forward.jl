envpath = normpath(joinpath((@__DIR__, "../../envs/env2")))
srcpath = normpath(joinpath((@__DIR__, "../../src/")))
datapath = normpath(joinpath((@__DIR__, "../../data/")))
using Pkg; Pkg.activate(envpath)
using JLD2

include(srcpath*"plotting.jl");
include(srcpath*"animation.jl")
include(srcpath*"toggle_triad.jl")

get_protein(cells) = map(x -> Tuple(view(x, 1, :)), cells)

function update_model_forward(state::AbstractMatrix, ps::AbstractArray, tau::Real, 
                   trans::AbstractArray, 
                   propensities::AbstractArray, 
                   pweights::AbstractWeights,
                   signal::AbstractMatrix, 
                   sum_signal::AbstractArray,
                   decay::AbstractArray, 
                   config_new::AbstractMatrix)
    σ_u, σ_b, ρ_u, ρ_b, η, δ = ps
    @views protein = state[1, :]
    @views gene = state[2, :]
    
    for k in 1:3
        if iszero(gene[k]) #if unbound, then leak
            trans[k] = rand(Poisson(ρ_u[k] * tau))
        elseif gene[k] == k #if positively bound, transcribe
            trans[k] = rand(Poisson(ρ_b[k] * tau))
        else
            trans[k] = 0
        end
    end

    # NOTE: extra forward loop: protein 1 activates gene 2,
    # protein 2 activates gene 3, and protein 3 activates gene 1
    trans[2] = gene[2] == 1 ? rand(Poisson(ρ_b[2] * tau)) : 0
    trans[3] = gene[3] == 2 ? rand(Poisson(ρ_b[3] * tau)) : 0
    trans[1] = gene[1] == 3 ? rand(Poisson(ρ_b[1] * tau)) : 0
        
    # Number of proteins that leak in the six neighbouring directions
    for k in 1:3
        dp = Poisson(protein[k] * η[k] * tau)
        signal[k, :] .= rand.(dp)
    end

    # Number of proteins that decay
    for k in 1:3
        decay[k] = rand(Binomial(protein[k], δ[k] * tau))
    end

    for k in 1:3
        if iszero(gene[k])
            propensities .= σ_b .* protein
            s = sum(propensities)
            if rand(Exponential(1 / s)) < tau
                pweights.values .= propensities ./ s 
                i = sample(1:3, pweights)
                gene[k] = i
                trans[i] -= 1 
            end
        elseif rand(Poisson(σ_u[k] * tau)) > 0
            trans[gene[k]] += 1
            gene[k] = 0
        end
    end    

    @views config_new[1, :] .= protein .+ trans .- decay .- sum!(sum_signal, signal)
    @views config_new[2, :] .= gene

    return config_new, signal
end

function get_frames_TT(nframes, niter_per_frame, ps, 
                       update_f=update_model_forward,
                       init_f=nothing)
    config = create_grid_TT(n, m)
    buf = Array{Int64, 3}(undef, n, m, 3)
    helpers = (MVector{3, Int}(undef),            # trans
               MVector{3, Float64}(undef),        # propensities
               ProbabilityWeights(ones(3) ./ 3),  # helper weights
               MMatrix{3, 6, Int}(undef),         # signal 
               MVector{3, Int}(undef),            # sum_signal
               MVector{3, Int}(undef),            # decay
               MMatrix{2, 3, Int}(undef))         # res
    frames = Vector{Matrix{Tuple{Int, Int, Int}}}(undef, nframes)
    
    if !isnothing(init_f)
        init_f(config)
    end

    frames[1] = get_protein(config)
    for i in 2:nframes
        for _ in 1:niter_per_frame
            run_iter_TT!(config, buf, ps, tau, helpers..., update_f)
        end
        frames[i] = get_protein(config)
    end

    return frames
end

# Model parameters
n = 128
m = 192

σ_u = [0.2, 0.2, 0.2]
σ_b = [0.06, 0.06, 0.06]
ρ_u = [0.01, 0.01, 0.01]
ρ_b = [100.0, 20.0, 100.0]
δ = ones(3) 
η = [0.0, 0.08, 0.0]
ps = [σ_u, σ_b, ρ_u, ρ_b, η, δ]

# Simulation parameters
tau = 0.01 
nframes = 150
niter_per_frame = 10#0

@time frames = get_frames_TT(nframes, niter_per_frame, ps, 
                             update_model_forward,
                             config -> init_cells_uniform_TT!(config, p=0.05))

mcs = [RGBA(249/255, 61/255, 46/255), RGBA(0.05, 0.05, 0.8), RGBA(0.53, 0.81, 1.0)]
cs = [mcs[2], mcs[3], mcs[1], mcs[2]]
cc = cgrad(cs)

@time hex_animation(frames, hexsize=3, creategif=false, createmovie=true, fps=15, cmap=cc,
                    fname=datapath*"animations/SI_Movie_5.mp4")
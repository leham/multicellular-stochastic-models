#=
Code for simulating a 2D cell tissue, where each cell contains a toggle triad, with diffusion to neighbours enabled.  
The cell tissue is a hexagonal grid with periodic boundary conditions (toroid geometry).
The diffusion rate is η and is fixed for all cells in the grid.
=#

using StatsBase
using Distributions
using StaticArrays

"""
Data is to be represented as a n × m matrix of tuples.
These matrix entries record physical positions: neighbouring values correspond to neighbouring cells. 
For simplicity, the right boundary will be identified with the left boundary, and the top with the bottom (torus).
We consider a hexagonal 2D tissue (3D version is not implemented).
"""

# Creates an n × m matrix of cells
function create_grid_TT(n::Int, m::Int)
    cells = [ @MMatrix zeros(Int, 2, 3) for _ in 1:n, _ in 1:m ]
    cells
end

# Reset the matrix of cells to zeros
function reset_grid_TT!(cells::AbstractMatrix)
    for i in eachindex(cells)
        fill!(cells[i], 0)
    end
    return nothing
end

# (re)initialise the cell grid so that it has 12 sparsely placed cells in the on state
function init_cells_TT!(cells::AbstractMatrix)
    reset_grid_TT!(cells)
    n, m = size(cells)
    # NOTE: at the moment this only works for even values of n and m
    cells[Int(n/4), Int(m/4)][2, 1] = 1
    cells[Int(n/4), Int(3*m/4)][2, 1] = 1
    cells[Int(3*n/4), Int(3*m/4)][2, 1] = 1
    cells[Int(3*n/4), Int(m/4)][2, 1] = 1
    cells[Int(n/4+2), Int(m/4+2)][2, 1] = 1
    cells[Int(n/4+2), Int(3*m/4+2)][2, 1] = 1
    cells[Int(3*n/4+2), Int(3*m/4+2)][2, 1] = 1
    cells[Int(3*n/4+2), Int(m/4+2)][2, 1] = 1
    cells[Int(n/4-2), Int(m/4-2)][2, 1] = 1
    cells[Int(n/4-2), Int(3*m/4-2)][2, 1] = 1
    cells[Int(3*n/4-2), Int(3*m/4-2)][2, 1] = 1
    cells[Int(3*n/4-2), Int(m/4-2)][2, 1] = 1
    return nothing
end

# (re)initialise the cell grid so that each gene is bound with some protein with a certain probability p
function init_cells_uniform_TT!(cells::AbstractMatrix; p::Real=0.01)
    reset_grid_TT!(cells)
    pweights = ProbabilityWeights([1-3*p, p, p, p])
    for i in eachindex(cells)
        cells[i][2, 1] = sample(0:3, pweights)
        cells[i][2, 2] = sample(0:3, pweights)
        cells[i][2, 3] = sample(0:3, pweights)
    end
    return nothing 
end    

get_mean_protein_number_TT(cells::AbstractMatrix) = mapreduce(x -> x[1, 1], +, cells) / length(cells)

function init_sim_utils_TT(n::Int, m::Int, ps::AbstractArray, ntasks::Int)
    chnl_buf = Channel{Array{Int64, 3}}(ntasks)
    chnl_ps = Channel{Vector{Vector{Float64}}}(ntasks)
    chnl_config = Channel{Matrix{MMatrix{2, 3, Int64, 6}}}(ntasks)
    chnl_helper = Channel{Tuple{MVector{3, Int64}, 
                                MVector{3, Float64},
                                ProbabilityWeights{Float64, Float64, Vector{Float64}}, 
                                MMatrix{3, 6, Int64, 18},
                                MVector{3, Int64},
                                MVector{3, Int64},
                                MMatrix{2, 3, Int64, 6}}}(ntasks)
    foreach(1:ntasks) do _
        put!(chnl_buf, Array{Int64, 3}(undef, n, m, 3))
        put!(chnl_ps, deepcopy(ps))
        put!(chnl_config, create_grid_TT(n, m))
        put!(chnl_helper, (MVector{3, Int}(undef),            # trans
                           MVector{3, Float64}(undef),        # propensities
                           ProbabilityWeights(ones(3) ./ 3),  # helper weights
                           MMatrix{3, 6, Int}(undef),         # signal 
                           MVector{3, Int}(undef),            # sum_signal
                           MVector{3, Int}(undef),            # decay
                           MMatrix{2, 3, Int}(undef)))        # res
    end
    return chnl_buf, chnl_ps, chnl_config, chnl_helper
end


function update_TT(state::AbstractMatrix, ps::AbstractArray, tau::Real, 
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
        #we don't do anything if blocked (bound by B[k] neither 0 nor k)
    end

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


function run_iter_TT!(cells, buf, ps, tau,
                       trans::AbstractArray, 
                       propensities::AbstractArray, 
                       pweights::AbstractWeights,
                       signal::AbstractMatrix, 
                       sum_signal::AbstractArray, 
                       decay::AbstractArray, 
                       config_new::AbstractMatrix,
                       update_f = update_TT)
    # Size of the matrix
    n, m = size(cells)   
    # Clear diffusion buffer
    fill!(buf, 0)

    # Run simulations for each cell
    @views for i in 1:n, j in 1:m
        config_new, signal = update_f(cells[i, j], ps, tau, 
                                      trans, propensities, pweights, signal, sum_signal, decay, config_new)
        copyto!(cells[i, j], config_new)
        # Apply hexagonal diffusion using wrapped indices 
        # in 6 directions (E, SE, SW, W, NE, NW)
        buf[i, mod1(j+1, end), :] .+= signal[:, 2] # E
        buf[mod1(i+1, end), j, :] .+= signal[:, 3] # SE
        buf[mod1(i+1, end), mod1(j-1, end), :] .+= signal[:, 4] # SW
        buf[i, mod1(j-1, end), :] .+= signal[:, 5] # W
        buf[mod1(i-1, end), j, :] .+= signal[:, 1] # NW
        buf[mod1(i-1, end), mod1(j+1, end), :] .+= signal[:, 6] # NE
    end

    # Add diffused molecules to their target cells
    # Ensure non-negative values
    @views for i in 1:n, j in 1:m
        cells[i, j][1, :] .+= buf[i, j, :]
        cells[i, j][1, :] .= max.(0, cells[i, j][1, :])
    end

    return nothing
end
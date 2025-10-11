#=
Code for simulating a 2D cell tissue using tau leaping, where each cell contains a feedback loop, with diffusion to neighbours enabled.  
The cell tissue is a hexagonal grid with periodic boundary conditions (toroid geometry).
=#

using StatsBase
using Distributions
using StaticArrays

# Creates an n × m matrix of cells
function create_grid_FL(n::Int, m::Int)
    cells = [ @MVector zeros(Int, 2) for _ in 1:n, _ in 1:m ]
    cells
end

# Reset the matrix of cells to zeros
function reset_grid_FL!(cells::AbstractMatrix)
    for i in eachindex(cells)
        fill!(cells[i], 0)
    end
    return nothing
end

# (re)initialise the cell grid so that it has 12 sparsely placed cells in the on state
function init_cells_FL!(cells::AbstractMatrix)
    reset_grid_FL!(cells)
    n, m = size(cells)
    # NOTE: at the moment this only works for even values of n and m
    cells[Int(n/4), Int(m/4)][2] = 1
    cells[Int(n/4), Int(3*m/4)][2] = 1
    cells[Int(3*n/4), Int(3*m/4)][2] = 1
    cells[Int(3*n/4), Int(m/4)][2] = 1
    cells[Int(n/4+2), Int(m/4+2)][2] = 1
    cells[Int(n/4+2), Int(3*m/4+2)][2] = 1
    cells[Int(3*n/4+2), Int(3*m/4+2)][2] = 1
    cells[Int(3*n/4+2), Int(m/4+2)][2] = 1
    cells[Int(n/4-2), Int(m/4-2)][2] = 1
    cells[Int(n/4-2), Int(3*m/4-2)][2] = 1
    cells[Int(3*n/4-2), Int(3*m/4-2)][2] = 1
    cells[Int(3*n/4-2), Int(m/4-2)][2] = 1
    return nothing
end

# (re)initialise the cell grid so that each gene is ON with a certain probability p
function init_cells_uniform_FL!(cells::AbstractMatrix; p::Real=0.1)
    reset_grid_FL!(cells)
    for i in axes(cells, 1), j in axes(cells, 2)
        cells[i, j][2] = Int(rand(Bernoulli(p)))
    end
    return nothing 
end    


get_mean_protein_number_FL(cells::AbstractMatrix) = mapreduce(x -> x[1], +, cells) / length(cells)

# utility function for preallocating different variables for multithreaded simulations
function init_sim_utils_FL(n::Int, m::Int, ps::AbstractArray, ntasks::Int)
    chnl_buf = Channel{Matrix{Int64}}(ntasks)
    chnl_ps = Channel{Vector{Float64}}(ntasks)
    chnl_config = Channel{Matrix{MVector{2, Int64}}}(ntasks)
    chnl_helper = Channel{Tuple{MVector{6, Int64}, 
                                MVector{2, Int64}}}(ntasks)
    foreach(1:ntasks) do _
        put!(chnl_buf, Matrix{Int64}(undef, n, m))
        put!(chnl_ps, deepcopy(ps))
        put!(chnl_config, create_grid_FL(n, m))
        put!(chnl_helper, (MVector{6, Int}(undef),      # signal 
                           MVector{2, Int}(undef)))     # res
    end
    return chnl_buf, chnl_ps, chnl_config, chnl_helper
end


# update a specified cell using tau leaping
function update_FL(state::AbstractVector, ps::AbstractArray, tau::Real, 
                    signal::AbstractVector, 
                    config_new::AbstractVector)
    σ_u, σ_b, ρ_u, ρ_b, η, δ = ps
    @views protein = state[1]
    @views gene = state[2]
    
    if iszero(gene) #if unbound, then leak
        trans = rand(Poisson(ρ_u * tau))
    else #if positively bound, transcribe
        trans = rand(Poisson(ρ_b * tau))
    end

    # Number of proteins that leak in the six neighbouring directions
    dp = Poisson(protein * η * tau)
    signal[:] .= rand.(dp)

    # Number of proteins that decay
    decay = rand(Binomial(protein, δ * tau))

    if iszero(gene) 
        s = σ_b * protein
        if rand(Exponential(1 / s)) < tau
            gene = 1
            protein -= 1
        end
    elseif rand(Poisson(σ_u * tau)) > 0
        gene = 0
        protein += 1
    end  

    @views config_new[1] = protein + trans - decay - sum(signal)
    @views config_new[2] = gene

    return config_new, signal
end

# run a single simulation step over the entire tissue
function run_iter_FL!(cells, buf, ps, tau,
                       signal::AbstractVector, 
                       config_new::AbstractVector)
    # Size of the matrix
    n, m = size(cells)   
    # Clear diffusion buffer
    fill!(buf, 0)

    # Run simulations for each cell
    @views for i in 1:n, j in 1:m
        config_new, signal = update_FL(cells[i, j], ps, tau, 
                                        signal, config_new)
        copyto!(cells[i, j], config_new)
        # Apply hexagonal diffusion using wrapped indices 
        # in 6 directions (E, SE, SW, W, NE, NW)
        buf[i, mod1(j+1, end)] += signal[2] # E
        buf[mod1(i+1, end), j] += signal[3] # SE
        buf[mod1(i+1, end), mod1(j-1, end)] += signal[4] # SW
        buf[i, mod1(j-1, end)] += signal[5] # W
        buf[mod1(i-1, end), j] += signal[1] # NW
        buf[mod1(i-1, end), mod1(j+1, end)] += signal[6] # NE
    end

    # Add diffused molecules to their target cells
    # Ensure non-negative values
    @views for i in 1:n, j in 1:m
        cells[i, j][1] += buf[i, j]
        cells[i, j][1] = max(0, cells[i, j][1])
    end

    return nothing
end
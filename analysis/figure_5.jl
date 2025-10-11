envpath = normpath(joinpath((@__DIR__, "../envs/env2")))
srcpath = normpath(joinpath((@__DIR__, "../src/")))
using Pkg; Pkg.activate(envpath)
include(srcpath*"plotting.jl")
include(srcpath*"cellular_automaton.jl")

using LinearAlgebra
using Random
using Distributions
using Luxor 
using LsqFit
using JLD2
using SpecialFunctions
using OhMyThreads: tmap
using Base.Threads: nthreads

# ------------------------------------------------------------------------------------------------------------
# Figure 5(C) top
# Example simulation snapshots of the first collision between two active regions

function colmap_SCA(x::Int)
    if iszero(x)
        return light_gray_col
    elseif isone(x)
        return turquoise_col
    else
        return pastel_red_col
    end
end

#=
# Original code that plots the entire hexagonal grid at the time of collision
function plot_collision(n, m, ρ, σ, maxiters=10^3)
    mat, _ = grow_until_collision(n, m, ρ, σ; maxiters)
    hexcorrect!(mat)
    
    # hexagonal grid sizing
    hexsize = 1
    xsize = round(hexsize * sqrt(3) * (m + 1))
    ysize = round(hexsize * 3/2 * (n + 1))
    
    d = @drawsvg begin
        background(light_gray_col)
        setline(0.2)
        vec_i_n = Vector{Luxor.Point}()
        vec_i_1 = Vector{Luxor.Point}()
        vec_j_m = Vector{Luxor.Point}()
        vec_j_1 = Vector{Luxor.Point}()
        for i in 1:n, j in 1:m
            # matrix row <-> y-axis on graph
            # matrix column <-> x-axis on graph
            pgon = Luxor.hextile(HexagonOffsetEvenR(j - Int(m/2), i - Int(n/2), 1))
            pgon = pgon .- Luxor.Point(hexsize * 0.5, hexsize * 0.75)
            nprot = mat[i, j]
            sethue(colmap_SCA(nprot))
            Luxor.poly(pgon, :fill)
            
            if i == n
                append!(vec_i_n, pgon[[3, 2, 1]])
            end

            if i == 1
                append!(vec_i_1, pgon[[4, 5, 6]])
            end
            
            if j == m
                if iseven(i)
                    append!(vec_j_m, pgon[[5, 6, 1, 2]])
                else
                    append!(vec_j_m, pgon[[6, 1]])
                end
            end
            
            if j == 1
                if isodd(i)
                    append!(vec_j_1, pgon[[5, 4, 3, 2]])
                else
                    append!(vec_j_1, pgon[[4, 3]])
                end
            end
            
        end
        sethue(gray_col)
        Luxor.poly(vec_i_1, :stroke, close=false, reversepath=false)
        Luxor.poly(vec_i_n, :stroke, close=false, reversepath=false)
        Luxor.poly(vec_j_m, :stroke, close=false, reversepath=false)
        Luxor.poly(vec_j_1, :stroke, close=false, reversepath=false)
    
    end xsize ysize
    d
end
=#


function plot_collision_zoom(n, m, ρ, σ, maxiters=10^3; n_zoom=96, m_zoom=96)
    # Zoom in on a smaller central region of the full hexagonal grid
    # NOTE: this is currently a bit limited and works only for certain n × m combinations
    # NOTE: have to be careful here with the hexagonal grid dimensions and the odd/even offset

    mat, _ = grow_until_collision(n, m, ρ, σ; maxiters)
    hexcorrect!(mat)
    mat = mat[div(n, 2)-div(n_zoom, 2)-1:1:div(n, 2)+div(n_zoom, 2), 
              div(m, 2)-div(m_zoom, 2)-1:1:div(m, 2)+div(m_zoom, 2)]
    
    # hexagonal grid sizing
    hexsize = 1
    xsize = round(hexsize * sqrt(3) * (m_zoom + 1))
    ysize = round(hexsize * 3/2 * (n_zoom + 1))

    d = @drawsvg begin
        background(light_gray_col)
        setline(0.2)
        vec_i_n = Vector{Luxor.Point}()
        vec_i_1 = Vector{Luxor.Point}()
        vec_j_m = Vector{Luxor.Point}()
        vec_j_1 = Vector{Luxor.Point}()

        for i in 1:n_zoom, j in 1:m_zoom
            # matrix row <-> y-axis on graph
            # matrix column <-> x-axis on graph
            pgon = Luxor.hextile(HexagonOffsetEvenR(j - Int(m_zoom/2), i - Int(n_zoom/2), 1))
            pgon = pgon .- Luxor.Point(hexsize * 0.5, hexsize * 0.75)
            nprot = mat[i, j]
            sethue(colmap_SCA(nprot))
            Luxor.poly(pgon, :fill)
            
            if i == n_zoom
                append!(vec_i_n, pgon[[3, 2, 1]])
            end

            if i == 1
                append!(vec_i_1, pgon[[4, 5, 6]])
            end
            
            if j == m_zoom
                if iseven(i)
                    append!(vec_j_m, pgon[[5, 6, 1, 2]])
                else
                    append!(vec_j_m, pgon[[6, 1]])
                end
            end
            
            if j == 1
                if isodd(i)
                    append!(vec_j_1, pgon[[5, 4, 3, 2]])
                else
                    append!(vec_j_1, pgon[[4, 3]])
                end
            end
            
        end
        sethue(gray_col)
        Luxor.poly(vec_i_1, :stroke, close=false, reversepath=false)
        Luxor.poly(vec_i_n, :stroke, close=false, reversepath=false)
        Luxor.poly(vec_j_m, :stroke, close=false, reversepath=false)
        Luxor.poly(vec_j_1, :stroke, close=false, reversepath=false)
    
    end xsize ysize
    d
end

n, m = 256, 256
ρ = 2e-6 
σ = 0.05

plot_collision_zoom(n, m, ρ, σ)
open("example_1.svg", "w") do f write(f, svgstring()) end

# ------------------------------------------------------------------------------------------------------------
# Figure 5(C) bottom right
# Radial expansion rate ρ: verify linear relationship wrt. σ

function get_growth_rate(n::Int, m::Int, ρ::Real, σ::Real, 
                         cells::AbstractMatrix=zeros(Int, n, m), 
                         buf::AbstractMatrix=zeros(Int, n, m);
                         step_size::Int=1,
                         maxiters::Int=1000)

    # Initalise the cell grid with one active gene in the middle
    fill!(cells, 0)
    cells[div(n, 2), div(m, 2)] = 2 # value of 2 indicates the central region
    collision = false # track whether a collision between central and any other active region has occured
    
    niter = 1
    npoints = 0
    r0 = 0.5
    Δr_sum = 0.0

    while !collision && niter <= maxiters
        collision = step_SCA!(cells, buf, collision, ρ, σ)
        if iszero(niter % step_size)
            r1 = get_radius(buf)
            npoints += 1 
            Δr_sum += (r1 - r0) / step_size
            r0 = r1
        end
        niter += 1
    end

    @assert niter <= maxiters "FAILED TO CONVERGE: maximum number of iterations reached."
    return Δr_sum / npoints
end


# Multithreaded code to measure the growth rate over a range of σ values
function get_growth_rate_vs_σ(prange, ρ, n, m; nsamples::Int=10, step_size::Int=1, ntasks=nthreads())
    chnl_rvec = Channel{Vector{Float64}}(ntasks)
    chnl_cells = Channel{Matrix{Int}}(ntasks)
    chnl_buf = Channel{Matrix{Int}}(ntasks)
    foreach(1:ntasks) do _
        put!(chnl_rvec, Vector{Float64}(undef, nsamples))
        put!(chnl_cells, zeros(Int, n, m))
        put!(chnl_buf, zeros(Int, n, m))
    end
    
    lk = ReentrantLock()
    
    means = tmap(eachindex(prange); ntasks) do i
        rvec = take!(chnl_rvec)
        cells = take!(chnl_cells)
        buf = take!(chnl_buf)
        σ = prange[i]

        t = @elapsed for j in 1:nsamples
            rvec[j] = get_growth_rate(n, m, ρ, σ, cells, buf; maxiters=10^3, step_size)
        end

        lock(lk) do
            println("σ = $σ, t = $t s")
        end

        mean_r = mean(rvec)
        put!(chnl_rvec, rvec)
        put!(chnl_cells, cells)
        put!(chnl_buf, buf)
        mean_r
    end

    return means
end

prange = 0.01:0.005:0.1

mean_growth_rates = @time get_growth_rate_vs_σ(prange, 2e-6, 256, 256, nsamples=1000, step_size=1)
#@load "mean_growth_rates.jld2" mean_growth_rates prange

f = CairoMakie.Figure(size = (size_pt[1]*0.8, size_pt[2]*0.8), figure_padding = 1)
ax = Axis(f[1, 1], xlabel = "Signal strength σ", ylabel = "Growth rate r",
          yticks=0:0.1:0.3)

rate_function(_σ, p) = p[1] .* _σ
res_fit = curve_fit(rate_function, collect(prange), mean_growth_rates, [1.0])
res_fit.param

CairoMakie.lines!(ax, prange, rate_function.(prange, Ref(res_fit.param)), linewidth=1, label="r = kσ", color=turquoise_col)
CairoMakie.scatter!(ax, prange, mean_growth_rates, marker=:xcross, color=red_col, markersize=5.5, label="SCA")
CairoMakie.ylims!(ax, 0, 0.31)

f

# ------------------------------------------------------------------------------------------------------------
# Figure 5(C) (bottom left) and Figure 5(D)
# Cubic root scaling law

# Multithreaded code to record both the timing of the collision and the region size over a range of σ values
# nsamples is the number of independent stochastic simulations run for each parameter value
function collect_radii_vs_σ(prange, ρ, n, m; nsamples::Int=10, ntasks=nthreads())
    chnl_rvec = Channel{Vector{Float64}}(ntasks)
    chnl_tvec = Channel{Vector{Float64}}(ntasks)
    chnl_cells = Channel{Matrix{Int}}(ntasks)
    chnl_buf = Channel{Matrix{Int}}(ntasks)
    foreach(1:ntasks) do _
        put!(chnl_rvec, Vector{Float64}(undef, nsamples))
        put!(chnl_tvec, Vector{Float64}(undef, nsamples))
        put!(chnl_cells, zeros(Int, n, m))
        put!(chnl_buf, zeros(Int, n, m))
    end
    
    lk = ReentrantLock()

    radius_time_vec = tmap(eachindex(prange); ntasks) do i
        rvec = take!(chnl_rvec)
        tvec = take!(chnl_tvec)
        cells = take!(chnl_cells)
        buf = take!(chnl_buf)
        σ = prange[i]
        
        t = @elapsed for j in 1:nsamples
            cells, tf = grow_until_collision(n, m, ρ, σ, cells, buf, maxiters=10^3)
            rvec[j] = get_radius(cells)
            tvec[j] = tf
        end

        lock(lk) do
            println("σ = $σ, t = $t s")
        end

        # TODO: could preallocate rvec/tvec too
        res = copy(rvec), copy(tvec)
        put!(chnl_rvec, rvec)
        put!(chnl_tvec, tvec)
        put!(chnl_cells, cells)
        put!(chnl_buf, buf)
        res
    end

    return radius_time_vec
end

@time rts_vec = collect_radii_vs_σ(prange, 2e-6, 256, 256, nsamples=1000)
#@load "analysis/data/radius_time_vector.jld2" rts_vec prange

# ------------------------------------------------------------------------------------------------------------
# Figure 5(C) (bottom left)

f = Figure(size = (size_pt[1]*0.8, size_pt[2]*0.8), figure_padding = 1)
ax = Axis(f[1, 1], 
          xlabel="Time until the first collision", 
          ylabel="Probability", 
          yticksvisible=false, 
          yticklabelsvisible=false)

tind = 9 # σ = 0.05
collision_times = rts_vec[tind][2]
tmax = ceil(maximum(collision_times), digits=1) + 1
CairoMakie.ylims!(ax, 0, nothing)
CairoMakie.xlims!(ax, 0.0, tmax)

Makie.density!(ax, collision_times, color=(turquoise_col, 0.5))
vlines!(ax, mean(collision_times), linewidth=2, color=(turquoise_col, 1.0), linestyle=:solid)
f

# ------------------------------------------------------------------------------------------------------------
# Figure 5(D)

f = CairoMakie.Figure(size = (size_pt[1]*1.0, size_pt[2]*1.0), figure_padding = 1)
ax = Axis(f[1, 1], 
          xlabel = "Signal strength σ", 
          ylabel = "Mean region size", 
          yticks=(10:5:25))

mean_region_sizes = mean.(first.(rts_vec))

cbrt_function(_σ, p) = p[1] .* cbrt.(_σ)
res_cbrt = curve_fit(cbrt_function, collect(prange), mean_region_sizes, [1.0])
res_cbrt.param

CairoMakie.lines!(ax, prange, cbrt_function.(prange, Ref(res_cbrt.param)), linewidth=1, color=turquoise_col)
CairoMakie.scatter!(ax, prange, mean_region_sizes, marker=:xcross, markersize=7, label=nothing, color=(red_col, 0.8))
CairoMakie.ylims!(9.5, 25.5)
f
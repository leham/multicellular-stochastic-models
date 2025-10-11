#=
Code for simulating a 2D cell tissue using using a stochastic cellular automaton (SCA) model.
The cell tissue is a hexagonal grid with periodic boundary conditions (toroid geometry).

In the model, each cell is either ON or OFF, and ON cells can activate (switch ON) their neighbours
with a fixed probability σ at each simulation time step. In addition, OFF cells can spontaneously switch
on according to a homogeneous Poisson point process with a background rate ρ per unit area.

NOTE: we assume a fixed initial condition -- one active cell in the center of the grid. 
Cells within this growing central region are marked with 2, cells in other active regions are marked as 1, 
and the inactive cells are marked as 0. We run the simulation until a collision between the central region
and any other active region occurs.
=#

# two neighbouring cells (i, j) demarcate a collision between the central and another region 
# if their states are (1, 2) or (2, 1)
hascollided(state_i::Int, state_j::Int) = state_i + state_j == 3 

function hascollided(i::Int, j::Int, cells::AbstractMatrix)
    return hascollided(cells[i, j], cells[i, mod1(j+1, end)]) || # E
           hascollided(cells[i, j], cells[mod1(i+1, end), j]) || # SE
           hascollided(cells[i, j], cells[mod1(i+1, end), mod1(j-1, end)]) || # SW
           hascollided(cells[i, j], cells[i, mod1(j-1, end)]) || # W
           hascollided(cells[i, j], cells[mod1(i-1, end), j]) || # NW
           hascollided(cells[i, j], cells[mod1(i-1, end), mod1(j+1, end)]) # NE
end


function infect_neighbour!(i::Int, j::Int, state::Int, cells::AbstractMatrix, collision::Bool, σ::Real)
    if iszero(cells[i, j])
        if rand() < σ
            # infect the neighbour
            cells[i, j] = state
            # check for collisions
            collision = !(collision) && hascollided(i, j, cells) ? true : collision
        end
    end
    return collision
end


function infect!(i::Int, j::Int, state::Int, cells::AbstractMatrix, collision::Bool, σ::Real)
    n, m = size(cells)
    collision = infect_neighbour!(i, mod1(j+1, m), state, cells, collision, σ) #E
    collision = infect_neighbour!(mod1(i+1, n), j, state, cells, collision, σ) # SE
    collision = infect_neighbour!(mod1(i+1, n), mod1(j-1, m), state, cells, collision, σ) # SW
    collision = infect_neighbour!(i, mod1(j-1, m), state, cells, collision, σ) # W
    collision = infect_neighbour!(mod1(i-1, n), j, state, cells, collision, σ) # NW
    collision = infect_neighbour!(mod1(i-1, n), mod1(j+1, m), state, cells, collision, σ) # NE
    return collision
end


function emerge!(i::Int, j::Int, cells::AbstractMatrix, collision::Bool, ρ::Real)
    if rand() < ρ
        # Poisson emergence
        cells[i, j] = 1
        # check for collisions
        collision = !(collision) && hascollided(i, j, cells) ? true : collision
    end
    return collision
end


function step_SCA!(cells::AbstractMatrix, buf::AbstractMatrix, collision::Bool, ρ::Real, σ::Real)
    n, m = size(cells)
    copyto!(buf, cells) # reset buffer
    
    for i in 1:n, j in 1:m
        if cells[i, j] == 2
            collision = infect!(i, j, 2, buf, collision, σ)
        end
    end

    for i in 1:n, j in 1:m
        if cells[i, j] == 1
            collision = infect!(i, j, 1, buf, collision, σ)
        end
    end

    for i in 1:n, j in 1:m
        if iszero(buf[i, j])
            collision = emerge!(i, j, buf, collision, ρ)
        end
    end

    copyto!(cells, buf)
    collision = sum(cells) == 2 * length(cells) ? true : collision

    return collision
end


function grow_until_collision(n::Int, m::Int, ρ::Real, σ::Real, 
                              cells::AbstractMatrix=zeros(Int, n, m), 
                              buf::AbstractMatrix=zeros(Int, n, m); 
                              maxiters::Int=1000)
    # Initalise the cell grid with one active gene in the middle
    fill!(cells, 0)
    cells[div(n, 2), div(m, 2)] = 2 # value of 2 indicates the central region
    collision = false # track whether a collision between central and any other active region has occured

    niter = 0
    while !collision && niter < maxiters
        collision = step_SCA!(cells, buf, collision, ρ, σ)
        niter += 1
    end

    @assert niter < maxiters "FAILED TO CONVERGE: maximum number of iterations reached."
    return cells, niter
end


# Helper function used to offset the hexagonal grid of cells for more intuitive plotting purposes
function hexcorrect!(mat::AbstractMatrix)
    n, m = size(mat)
    offset = -Int(div(m, 2) / 2) - 1
    for i in 1:n
        row = view(mat, i, :)
        if iseven(i)
            circshift!(row, Int(mod(i/2, m)-1))
        else
            circshift!(row, Int(mod((i + 1)/2, m)-1))
        end
        circshift!(row, offset) # center the grid back on the main region
    end
    return nothing
end

# Compute radius of the central active region of cells (approximated by πr²)
function get_radius(cells::AbstractMatrix)
    area = sum(cell == 2 for cell in cells)
    return sqrt(area/π)
end
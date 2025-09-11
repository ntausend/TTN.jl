
##########################################################
# TTN traversal
##########################################################

"""
    ttn_traversal_least_steps(net; start=(1,1), include_layer0=false, exclude_topnode=true)

Return a route (a vector of positions `(l,i)`) that starts at `start` and
visits every TTN node with the minimum number of edge steps.

- If `include_layer0=false` (default), only virtual layers `l ≥ 1` are traversed.
- If `include_layer0=true`, physical leaf nodes `l == 0` are included.

The returned NamedTuple has:
  - `walk`: the full step-by-step walk (nodes may repeat when backtracking)
  - `visit_order`: nodes in the order they are first visited (no repeats)
  - `end_leaf`: the leaf where the walk finishes (farthest from `start`)
"""
function ttn_traversal_least_steps(net; start=(1,1),
                                   include_layer0::Bool=false,
                                   exclude_topnode::Bool=true)
    maxL = TTN.number_of_layers(net)
    top  = (maxL, 1)
    keep(l) = include_layer0 ? true : (l >= 1)

    # --- Build undirected adjacency, ensuring every touched node exists as a key ---
    adj = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()
    ensure!(d, k) = get!(d, k, Tuple{Int,Int}[])

    for l in 1:maxL
        for i in 1:TTN.number_of_tensors(net, l)
            parent = (l, i)
            keep(l) || continue
            ensure!(adj, parent)
            for ch in TTN.child_nodes(net, parent)
                keep(ch[1]) || continue
                ensure!(adj, ch)              # important for layer-0 children
                push!(adj[parent], ch)
                push!(adj[ch], parent)
            end
        end
    end

    @assert haskey(adj, start) "Start node $start is not in the selected layers."

    # --- BFS to get distances & backpointers from start ---
    dist   = Dict{Tuple{Int,Int}, Int}(n => typemax(Int) for n in keys(adj))
    parent = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()
    dist[start] = 0
    q = Tuple{Int,Int}[start]
    head = 1
    while head <= length(q)
        u = q[head]; head += 1
        for v in adj[u]
            if dist[v] == typemax(Int)
                dist[v] = dist[u] + 1
                parent[v] = u
                push!(q, v)
            end
        end
    end

    # --- farthest leaf (degree==1 unless start is isolated) ---
    leaves = Tuple{Int,Int}[]
    for (n, nbrs) in adj
        if (length(nbrs) == 1 && n != start) || (n == start && isempty(nbrs))
            push!(leaves, n)
        end
    end
    end_leaf = isempty(leaves) ? start : leaves[argmax([dist[n] for n in leaves])]

    # --- unique start→end path (skip final backtracking along it) ---
    ordered_path = Tuple{Int,Int}[]
    cur = end_leaf
    while true
        pushfirst!(ordered_path, cur)
        cur == start && break
        cur = parent[cur]
    end
    path_nodes = Set(ordered_path)
    path_idx   = Dict(n => i for (i,n) in enumerate(ordered_path))

    # --- DFS Euler-like tour with backtrack skipping along start→end path ---
    walk = Tuple{Int,Int}[]
    visit_order = Tuple{Int,Int}[]
    visited = Set{Tuple{Int,Int}}()

    # emit (for visit_order) unless the user asked to exclude the top node
    should_emit(n) = !(exclude_topnode && n == top)

    function dfs(u::Tuple{Int,Int}, p::Union{Nothing,Tuple{Int,Int}})
        push!(walk, u)
        if should_emit(u) && !(u in visited)
            push!(visit_order, u)
            push!(visited, u)
        end

        neighs = [v for v in adj[u] if v != p]
        sort!(neighs; lt = (v1, v2) -> begin
            on1 = v1 in path_nodes; on2 = v2 in path_nodes
            on1 == on2 ? (v1 < v2) : (!on1 && on2)
        end)

        for v in neighs
            dfs(v, u)
            if (u in path_nodes) && (v in path_nodes) && (path_idx[v] == path_idx[u] + 1)
                # skip the backtrack along the final path
            else
                push!(walk, u)
            end
        end
    end

    dfs(start, nothing)
    return (walk=walk, visit_order=visit_order, end_leaf=end_leaf)
end
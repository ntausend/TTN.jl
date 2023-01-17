# adding the qr decomposition for qr-sparse ITensors. Workaround till it is fully suported by ITensors


function add_trivial_index(A::ITensor, Ainds)
  α = ITensors.trivial_index(Ainds) #If Ainds[1] has no QNs makes Index(1), otherwise Index(QN()=>1)
  vα = onehot(eltype(A), α => 1)
  A *= vα
  return A, vα, [α]
end

function add_trivial_index(A::ITensor, Linds, Rinds)
  vαl, vαr = nothing, nothing
  if isempty(Linds)
    A, vαl, Linds = add_trivial_index(A, Rinds)
  end
  if isempty(Rinds)
    A, vαr, Rinds = add_trivial_index(A, Linds)
  end
  return A, vαl, vαr, Linds, Rinds
end

function remove_trivial_index(Q::ITensor, R::ITensor, vαl, vαr)
  if !isnothing(vαl) #should have only dummy + qr,Link
    Q *= dag(vαl)
  end
  if !isnothing(vαr) #should have only dummy + qr,Link
    R *= dag(vαr)
  end
  return Q, R
end

LinearAlgebra.qr(A::ITensor; kwargs...) = error(ITensors.noinds_error_message("qr"))

function LinearAlgebra.qr(A::ITensor, Linds...; kwargs...)
  qtag::TagSet = get(kwargs, :tags, "Link,qr") #tag for new index between Q and R
  Lis = commoninds(A, ITensors.indices(Linds...))
  Ris = uniqueinds(A, Lis)
  lre = isempty(Lis) || isempty(Ris)
  # make a dummy index with dim=1 and incorporate into A so the Lis & Ris can never
  # be empty.  A essentially becomes 1D after collection.
  if (lre)
    A, vαl, vαr, Lis, Ris = add_trivial_index(A, Lis, Ris)
  end

  #
  #  Use combiners to render A down to a rank 2 tensor ready matrix QR routine.
  #
  CL, CR = combiner(Lis...), combiner(Ris...)
  cL, cR = combinedind(CL), combinedind(CR)
  AC = A * CR * CL
  #
  #  Make sure we don't accidentally pass the transpose into the matrix qr routine.
  #
  if inds(AC) != IndexSet(cL, cR)
    AC = ITensors.permute(AC, cL, cR)
  end
  # qr the matrix.
  QT, RT = qr(ITensors.tensor(AC); kwargs...)

  #
  #  Undo the combine oepration, to recover all tensor indices.
  #
  Q, R = itensor(QT) * dag(CL), itensor(RT) * dag(CR)

  # Conditionally remove dummy indices.
  if (lre)
    Q, R = remove_trivial_index(Q, R, vαl, vαr)
  end
  #
  # fix up the tag name for the index between Q and R.
  #  
  q = commonind(Q, R)
  settags!(Q, qtag, q)
  settags!(R, qtag, q)
  q = settags(q, qtag)

  return Q, R, q
end



function LinearAlgebra.qr(T::ITensors.BlockSparseTensor{ElT,2}; kwargs...) where {ElT}

  # getting total number of blocks
  nnzblocksT = nnzblocks(T)
  nzblocksT = nzblocks(T)

  Qs = Vector{ITensors.DenseTensor{ElT,2}}(undef, nnzblocksT)
  Rs = Vector{ITensors.DenseTensor{ElT,2}}(undef, nnzblocksT)

  for (jj, b) in enumerate(eachnzblock(T))
    blockT = ITensors.blockview(T, b)
    QRb = qr(blockT; kwargs...) #call dense qr at src/linearalgebra.jl 387

    if (isnothing(QRb))
      return nothing
    end

    Q, R = QRb
    Qs[jj] = Q
    Rs[jj] = R
  end

  nb1_lt_nb2 = (
    nblocks(T)[1] < nblocks(T)[2] ||
    (nblocks(T)[1] == nblocks(T)[2] && dim_it(T, 1) < dim_it(T, 2))
  )

  # setting the right index of the Q isometry, this should be
  # the smaller index of the two indices of of T
  qindl = ind(T, 1)
  if nb1_lt_nb2
    qindr = sim(ind(T, 1))
  else
    qindr = sim(ind(T, 2))
  end

  # can qindr have more blocks than T?
  if nblocks(qindr) > nnzblocksT
    resize!(qindr, nnzblocksT)
  end

  for n in 1:nnzblocksT
    q_dim_red = minimum(dims_it(Rs[n]))
    NDTensors.setblockdim!(qindr, q_dim_red, n)
  end

  # correcting the direction of the arrow
  # since qind2r is basically a copy of qind1r
  # if one have to be corrected the other one 
  # should also be corrected
  if (dir(qindr) != dir(qindl))
    qindr = dag(qindr)
  end

  indsQ = setindex(inds(T), dag(qindr), 2)
  indsR = setindex(inds(T), qindr, 1)

  nzblocksQ = Vector{Block{2}}(undef, nnzblocksT)
  nzblocksR = Vector{Block{2}}(undef, nnzblocksT)

  for n in 1:nnzblocksT
    blockT = nzblocksT[n]

    blockQ = (blockT[1], UInt(n))
    nzblocksQ[n] = blockQ

    blockR = (UInt(n), blockT[2])
    nzblocksR[n] = blockR
  end

  Q = ITensors.BlockSparseTensor(ElT, undef, nzblocksQ, indsQ)
  R = ITensors.BlockSparseTensor(ElT, undef, nzblocksR, indsR)

  for n in 1:nnzblocksT
    Qb, Rb = Qs[n], Rs[n]
    blockQ = nzblocksQ[n]
    blockR = nzblocksR[n]

    if VERSION < v"1.5"
      # In v1.3 and v1.4 of Julia, Ub has
      # a very complicated view wrapper that
      # can't be handled efficiently
      Qb = copy(Qb)
      Rb = copy(Vb)
    end

    ITensors.blockview(Q, blockQ) .= Qb
    ITensors.blockview(R, blockR) .= Rb
  end

  for b in nzblocks(Q)
    i1 = inds(Q)[1]
    i2 = inds(Q)[2]
    r1 = inds(R)[1]
    newqn = -dir(i2) * flux(i1 => Block(b[1]))
    ITensors.setblockqn!(i2, newqn, b[2])
    ITensors.setblockqn!(r1, newqn, b[2])
  end


  return Q, R
end
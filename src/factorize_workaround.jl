function factorize_own(
  A::ITensor,
  Linds...;
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  ortho=nothing,
  tags=nothing,
  plev=nothing,
  which_decomp=nothing,
  # eigen
  eigen_perturbation=nothing,
  # svd
  svd_alg=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
  min_blockdim=nothing,
  (singular_values!)=nothing,
  dir=nothing,
)
  if !isnothing(eigen_perturbation)
    if !(isnothing(which_decomp) || which_decomp == "eigen")
      error("""when passing a non-trivial eigen_perturbation to `factorize`,
               the which_decomp keyword argument must be either "automatic" or
               "eigen" """)
    end
    which_decomp = "eigen"
  end
  ortho = NDTensors.replace_nothing(ortho, "left")
  tags = NDTensors.replace_nothing(tags, ts"Link,fact")
  plev = NDTensors.replace_nothing(plev, 0)

  # Determines when to use eigen vs. svd (eigen is less precise,
  # so eigen should only be used if a larger cutoff is requested)
  automatic_cutoff = 1e-12
  Lis = commoninds(A, ITensors.indices(Linds...))
  Ris = uniqueinds(A, Lis)
  dL, dR = dim(Lis), dim(Ris)
  # maxdim is forced to be at most the max given SVD
  if isnothing(maxdim)
    maxdim = min(dL, dR)
  end
  #maxdim = min(maxdim, min(dL, dR))
  might_truncate = !isnothing(cutoff) || maxdim < min(dL, dR)

  if isnothing(which_decomp)
    if !might_truncate && ortho != "none"
      which_decomp = "qr"
    elseif isnothing(cutoff) || cutoff ≤ automatic_cutoff
      which_decomp = "svd"
    elseif cutoff > automatic_cutoff
      which_decomp = "eigen"
    end
  end
  if which_decomp == "svd"
    LR = ITensors.factorize_svd(
      A, Linds...; mindim, maxdim, cutoff, tags, ortho, alg=svd_alg, dir, singular_values!
    )
    if isnothing(LR)
      return nothing
    end
    L, R, spec = LR
  elseif which_decomp == "eigen"
    L, R, spec = ITensors.factorize_eigen(A, Linds...; mindim, maxdim, cutoff, tags, ortho, eigen_perturbation)
  elseif which_decomp == "qr"
    L, R = ITensors.factorize_qr(A, Linds...; ortho, tags)
    spec = Spectrum(nothing, 0.0)
  else
    throw(ArgumentError("""In factorize, factorization $which_decomp is not
     currently supported. Use `"svd"`, `"eigen"`, `"qr"` or `nothing`."""))
  end

  # Set the tags and prime level
  l = commonind(L, R)
  l̃ = setprime(settags(l, tags), plev)
  L = replaceind(L, l, l̃)
  R = replaceind(R, l, l̃)
  l = l̃

  return L, R, spec, l
end

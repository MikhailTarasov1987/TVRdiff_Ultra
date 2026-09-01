def TVRdiff_Ultra(data, n, h, alpha, eps=1e-6, itern=50, cg_tol=1e-6, cg_maxiter=200):

  """Total-variation regularized differentiation of a 1D signal, matrix-free.

    Splits `data` into non-overlapping sub-sequences of length `n`, then for
    each sub-sequence recovers a piecewise-smooth derivative estimate by
    minimizing a total-variation regularized least-squares objective via
    Newton's method, where each Newton step solves the Hessian system with
    matrix-free conjugate gradient (no dense or sparse n x n matrices are ever formed).
    Sub-sequences are processed in parallel across the batch dimension via
    `vmap`.

    Parameters
    ----------
    data : array_like, shape (N,)
        1D input signal to differentiate. Will be split into
        `N // n` non-overlapping sub-sequences of length `n`; any
        remainder samples at the end that don't fill a full sub-sequence
        are dropped.
    n : int
        Length of each sub-sequence the signal is split into before
        differentiation. Internally, each sub-sequence is treated on a grid
        of `n + 1` points.
    h : float
        Sample spacing (step size) between consecutive points in `data`,
        used to scale the differentiation and integration operators.
    alpha : float
        Regularization strength controlling the trade-off between fidelity
        to `data` and smoothness (total variation) of the recovered
        derivative. Larger values produce smoother, more piecewise-constant
        derivative estimates; smaller values track the data more closely
        (and more noise).
    eps : float, optional
        Small positive constant added inside the total-variation term to
        keep it differentiable at zero-slope points (avoids division by
        zero). Default is 1e-6.
    itern : int, optional
        Number of outer Newton iterations to run per sub-sequence.
        Default is 50.
    cg_tol : float, optional
        Relative residual tolerance for the inner conjugate-gradient solve
        at each Newton step. Default is 1e-6.
    cg_maxiter : int, optional
        Maximum number of conjugate-gradient iterations per Newton step.
        Default is 200.

    Returns
    -------
    result : dict
        Dictionary with the following keys, each a 1D flattened
        `jax.numpy.ndarray`:

        - ``"data"`` : the original input data, truncated to a multiple of
          `n` and reshaped/raveled back to 1D (i.e. the portion of `data`
          that was actually processed).
        - ``"denoised_data"`` : the denoised reconstruction of the signal,
          obtained by re-integrating the recovered derivative estimate.
        - ``"diff_data"`` : the recovered derivative estimate for the
          processed signal, at the same sample locations as ``"data"``.
    """

  num_sub_sequences = data.shape[0] // n
  f_batch = jnp.stack(jnp.split(data[:n * num_sub_sequences], num_sub_sequences))
  f_batch_ = f_batch - f_batch[:, [0]]
  n_u = n+1
  ShapeDtypeStruct = jax.ShapeDtypeStruct(shape=(n_u,), dtype=np.dtype(np.float32))
  u_init = jnp.ones(n_u)
  iteration_array = jnp.arange(itern)

  def A(v,h):
    return h*(jnp.cumsum(v)[:-1] - 0.5*(v[0]+v[:-1]))

  def D(v,h):
    return jnp.diff(v)/h

  def Dt_(h, ShapeDtypeStruct):
    fun = jax.linear_transpose(lambda v: D(v,h), ShapeDtypeStruct)
    return lambda w : fun(w)[0]

  def At_(h, ShapeDtypeStruct):
    fun = jax.linear_transpose(lambda v: A(v,h), ShapeDtypeStruct)
    return lambda w : fun(w)[0]

  At = At_(h, ShapeDtypeStruct)
  Atv = lambda v : At(v)
  Dt = Dt_(h, ShapeDtypeStruct)

  def en_fun(v, eps):
    return jnp.pow((jnp.pow(jnp.diff(v), 2) + eps), -0.5)

  def Ln(h, Dt, en, D, v):
    return h*Dt(en*D(v,h))

  def AtA(At, A, v, h):
    return At(A(v,h))

  def H(v, AtA, alpha, Ln, Dt, en, D, h):
    return AtA(At, A, v, h) +alpha*Ln(h, Dt, en, D, v)

  def one_iteration(u, h, alpha, eps, AtA, Atf, Ln, Dt, D, H, cg_tol=1e-6, cg_maxiter=200):
    en = en_fun(u, eps)
    gn = AtA(At, A, u, h) - Atf + alpha*Ln(h, Dt, en, D, u)
    H_fun = lambda s : H(s, AtA, alpha, Ln, Dt, en, D, h)
    s, _ = jax.scipy.sparse.linalg.cg(H_fun, -gn, x0=jnp.zeros_like(u),
                                            tol=cg_tol, maxiter=cg_maxiter)

    u = s+u
    return u, u

  @partial(jax.jit, static_argnames=("AtA", "Ln", "Dt", "D", "H"))
  def fit_one(Atf, u_init, h, alpha, eps, AtA, Ln, Dt, D, H, iteration_array, cg_tol, cg_maxiter):

    step_fun = lambda u, _ : one_iteration(u, h, alpha, eps,
                                          AtA, Atf, Ln, Dt,
                                          D, H, cg_tol=cg_tol,
                                          cg_maxiter=cg_maxiter)

    u_final, _  = jax.lax.scan(step_fun, u_init, xs=iteration_array)
    return u_final

  Atf_batch = jax.vmap(Atv)(f_batch_)
  fit_vmap = jax.vmap(lambda Atf: fit_one(Atf, u_init, h, alpha, eps, AtA,
                                          Ln, Dt, D, H, iteration_array,
                                          cg_tol, cg_maxiter))
  u_batch = fit_vmap(Atf_batch)
  f_hat_batch = jax.vmap(lambda u : A(u,h))(u_batch)
  f_hat_batch  = f_hat_batch+f_batch[:, [0]]
  return dict(data=f_batch.ravel(), diff_data=u_batch[:,:-1].ravel(), denoised_data=f_hat_batch.ravel())

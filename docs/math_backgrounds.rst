.. _math_backgrounds:

Mathematical Backgrounds
========================

.. rubric:: Dynamical system

Assume that the system has dimension :math:`d` and invariant distribution :math:`d\mu=\frac{1}{Z} \mathrm{e}^{-\beta V}dx`, where :math:`\beta=(k_BT)^{-1}`, :math:`T` is the temperature, :math:`V` is the potential of the sytem, and :math:`Z` is the normalizing constant.

.. rubric:: Training data

The trajectory data is :math:`(x_i)_{1\le i \le n}` with weights :math:`(v_i)_{1\le i \le n}`, such that one can approximate the distribution :math:`\mu` by

.. math::
   \int_{\mathbb{R}^{d}} f(x) \mu(dx) \approx \frac{\sum_{l=1}^n v_l f(x_l)}{\sum_{l=1}^n v_l}\,,

for test functions :math:`f`. 

.. _rep_colvars:

.. rubric:: Representation of collective variables

In molecular applications, we often expect that the collective variables, denoted by :math:`\xi:\mathbb{R}^{d}\rightarrow \mathbb{R}^k`, are invariant under rotations and translations, or that :math:`\xi` is a function of features (e.g. bond distances, angles). For this purpose, the training tasks in the module :mod:`colvarsfinder.core` look for :math:`\xi` that is of the form

.. math::

    \xi(x)=g(r(x)), 

where :math:`r:\mathbb{R}^{d}\rightarrow \mathbb{R}^{d_r}` could be
the action of certain alignment, a map to certain features, or even the
identity map, and :math:`g` is the map that we want to learn by training neural networks.

In the training tasks in :mod:`colvarsfinder.core`, :math:`r` is specified as the preprocessing layer in the input parameter which is a PyTorch neural network module, and :math:`g` corresponds to (part of) the neural network model that is to be trained. 

.. _loss_autoencoder:

.. rubric:: Loss function for training autoencoder 

The class :class:`colvarsfinder.core.AutoEncoderTask` trains an encoder :math:`f_{enc}:\mathbb{R}^{d_r}\rightarrow \mathbb{R}^k` and a decoder :math:`f_{dec}:\mathbb{R}^{k}\rightarrow \mathbb{R}^{d_r}` by the reconstruction loss in the *transformed* space :math:`\mathbb{R}^{d_r}`:

.. math::

        & \int_{\mathbb{R}^{d_r}} |f_{dec}\circ f_{enc}(y)-y|^2  r_{\#}\mu(dy) \\
       =& \int_{\mathbb{R}^{d}} |f_{dec}\circ f_{enc}(r(x))-r(x)|^2  \mu(dx) \\
    \approx& \frac{\sum_{l=1}^{n} v_l|f_{dec}\circ f_{enc}(y_l) - y_l|^2}{\sum_{l=1}^n v_l},

where :math:`r_{\#}\mu` denotes the pushforward measure of :math:`\mu` by the map :math:`r`, and :math:`y_l = r(x_l)`.

After training, the collective variables are constructed as

.. math::
    \xi = f_{enc}\circ r.

.. _loss_eigenfunction:

.. rubric:: Loss function for training eigenfunctions 

The class :class:`colvarsfinder.core.EigenFunctionTask` finds the leading eigenfunctions :math:`\phi_1, \phi_2, \dots, \phi_k:\mathbb{R}^d\rightarrow \mathbb{R}` of infinitesimal generator or transfer operator. 

1. In the generator case, the generator is assumed to be 

.. math::
    \mathcal{L} = -\nabla V \cdot \nabla f + \frac{1}{\beta} \Delta f\,,

for a test function :math:`f`, and this class computes the eigenfunctions of PDE 

.. math::

    -\mathcal{L}\phi = \lambda \phi,

corresponding to the :math:`k` smallest eigenvalues :math:`0 < \lambda_1 \le \lambda_2 \le \cdots \le \lambda_k`. This is done by training neural network to learn functions :math:`g_1, g_2, \cdots, g_k:\mathbb{R}^{d_r}\rightarrow \mathbb{R}` using the data-version of the loss 

.. _loss_eigen_generator:

.. math::
   & \mathrm{Loss}((g_i)_{1\le i\le K})  \\
   = & \sum_{i=1}^k \omega_i  \frac{\beta^{-1} \mathbf{E}_{\mu} \big[(a \nabla f_i)\cdot \nabla f_i\big]}{\mbox{var}_{\mu} f_i} + \alpha \sum_{1 \le i_1 \le i_2 \le k} \Big[\mathbf{E}_{\mu} \Big((f_{i_1}-\mathbf{E}_{\mu}f_{i_1})(f_{i_2}-\mathbf{E}_{\mu}f_{i_2})\Big) - \delta_{i_1i_2}\Big]^2,

where 

    #. :math:`\mathbf{E}_{\mu}` and :math:`\mbox{var}_{\mu}` are the expectation and variance with respect to :math:`\mu`, respectively;
    #. :math:`\alpha` is the penalty parameter;
    #. :math:`a\in \mathbb{R}^{d\times d}` is a diagonal matrix;
    #. :math:`\omega_1 \ge \omega_2 \ge \dots \ge \omega_k > 0` are :math:`k` positive constants;
    #. :math:`f_i=g_i\circ r, 1\le i \le k`.

After training, the collective variables are constructed by 

.. math::
    \xi = (g_1\circ r, g_2\circ r, \dots, g_k\circ r)^T.

2. In the transfer operator case, assume the lag-time is :math:`\tau` and the transition density at time :math:`\tau` given the state :math:`x` at time zero is :math:`p_\tau(\cdot|x)`. The loss function used to learn eigenfunctions is 

.. _loss_eigen_transfer:

.. math::
   & \mathrm{Loss}((g_i)_{1\le i\le K})  \\
   = & \frac{1}{2\tau}\sum_{i=1}^k \omega_i  \frac{\mathbf{E}_{x\sim\mu, x'\sim p_\tau(\cdot|x)} \big[|f_i(x')- f_i(x)|^2\big]}{\mbox{var}_{\mu} f_i} \\
   & +\alpha \sum_{1 \le i_1 \le i_2 \le k} \Big[\mathbf{E}_{\mu} \Big((f_{i_1}-\mathbf{E}_{\mu}f_{i_1})(f_{i_2}-\mathbf{E}_{\mu}f_{i_2})\Big) - \delta_{i_1i_2}\Big]^2,

In practice, with weighted trajectory data :math:`x^{(1)}, \cdots, x^{(n)}` and assuming :math:`\tau=j\Delta t`, where :math:`\Delta t` is the time interval between two consecutive states and :math:`j` is an integer, then 
the first term in the loss function above is estimated using 

.. math::
    \mathbf{E}_{x\sim\mu, x'\sim p_\tau(\cdot|x)} \big[|f_i(x')- f_i(x)|^2\big] \approx \frac{\sum_{l=1}^{n-j} v_l |f_i(x_{l+j}) - f_i(x_{l})|^2}{\sum_{l=1}^{n-j} v_l}\,.

.. _loss_regautoencoder:

.. rubric:: Loss function for regularized autoencoders

The class :class:`colvarsfinder.core.RegAutoEncoderTask` learns regularized autoencoders using a loss that is the sum of the standard reconstruction loss and several regularization terms. 
The model consists of an encoder :math:`f_{enc}:\mathbb{R}^{d_r}\rightarrow \mathbb{R}^k` and a decoder :math:`f_{dec}:\mathbb{R}^{k}\rightarrow \mathbb{R}^{d_r}`, and regularizers :math:`\widetilde{f}_1,\cdots, \widetilde{f}_K:\mathbb{R}^k\rightarrow \mathbb{R}`. The loss involves two lag-times :math:`\tau_1,\tau_2 \ge 0`, as well as several other parameters. When :math:`\tau_2>0`, regularizers correspond to eigenfunctions of transfer operators, and the loss function is 

.. math::
   & \mathrm{Loss}(f_{enc}, f_{dec}, \{\widetilde{f}_i\}_{1\le i\le K}) \\
     = & \alpha \mathbf{E}_{x\sim\mu, x'\sim p_{\tau_1}(\cdot|x)} |f_{dec}\circ f_{enc}(r(x))-r(x')|^2 \\
   +&  \frac{\gamma_1}{2\tau}\sum_{i=1}^K \omega_i \frac{\mathbf{E}_{x\sim\mu, x'\sim p_{\tau_2}(\cdot|x)} \big[|f_i(x')- f_i(x)|^2\big]}{\mbox{var}_{\mu} f_i} \\
   +& \gamma_2 \sum_{1 \le i_1 \le i_2 \le K} \Big[\mathbf{E}_{\mu} \Big((f_{i_1}-\mathbf{E}_{\mu}f_{i_1})(f_{i_2}-\mathbf{E}_{\mu}f_{i_2})\Big) - \delta_{i_1i_2}\Big]^2 \\
   +& \eta_1 \sum_{i=1}^k \mathbf{E}_{\mu} |(\nabla_y f_{enc,i})\circ r|^2 + \eta_2 \sum_{i=1}^k (\mbox{var}_{\mu} (f_{enc,i}\circ r)-1)^2 \\
   +& \eta_3 \Big[\mathbf{E}_{\mu} \Big((f_{enc, i_1}\circ r-\mathbf{E}_{\mu}(f_{enc,i_1}\circ r))(f_{enc,i_2}\circ r-\mathbf{E}_{\mu}(f_{enc, i_2}\circ r))\Big) - \delta_{i_1i_2}\Big]^2\,,
 
where :math:`f_i = \widetilde{f}_i\circ f_{enc}\circ r`.

When :math:`\tau_2=0`, regularizers correspond to eigenfunctions of generators, and the loss is similar to the one above, except that the third line in the loss above is replaced by 

.. math::

   \gamma_1 \sum_{i=1}^K \omega_i  \frac{\beta^{-1} \mathbf{E}_{\mu} \big[(a \nabla f_i)\cdot \nabla f_i\big]}{\mbox{var}_{\mu} f_i} \,.

.. rubric:: References

#. https://arxiv.org/abs/2307.00365

#. https://doi.org/10.1016/j.jcp.2022.111377

#. https://doi.org/10.1063/1.5092521

#. https://doi.org/10.1038/s41467-017-02388-1

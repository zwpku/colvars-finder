.. _math_backgrounds:

Mathematical Backgrounds
========================

.. rubric:: Dynamical system

Assume that the dimension of the system is :math:`d`. Let the invariant distribution of the system at temperature :math:`T` be
:math:`d\mu=\frac{1}{Z} \mathrm{e}^{-\beta V}`, where :math:`\beta=(k_BT)^{-1}`, :math:`V` is the potential of the sytem, and :math:`Z` is the normalizing constant.

.. rubric:: Training data

Assume that the trajectory data is :math:`(x_i)_{1\le i \le n}` with weights :math:`(v_i)_{1\le i \le n}`,
such that one can approximate the distribution :math:`\mu` by

.. math::
   \int_{\mathbb{R}^{d}} f(x) \mu(dx) \approx \frac{\sum_{l=1}^n v_l f(x_l)}{\sum_{l=1}^n v_l}\,,

for test functions :math:`f`. 

.. _rep_colvars:

.. rubric:: Representation of collective variables

In molecular applications, we often expect that the collective variables, denoted by :math:`\xi:\mathbb{R}^{d}\rightarrow \mathbb{R}^k`, are invariant under rotations and translations, or that :math:`\xi` is a function of features (e.g. bond distances, angles).
To achieve this goal, the training tasks in the module :mod:`colvarsfinder.core` look for :math:`\xi` that is of the form

.. math::

    \xi(x)=g(r(x)), 

where :math:`r:\mathbb{R}^{d}\rightarrow \mathbb{R}^{d_r}` could be
the action of certain alignment, a map to certain features, or even the
identity map, and :math:`g` is the map that we want to learn by training neural networks.

In the training tasks of :mod:`colvarsfinder.core`, :math:`r` is specified in the input parameter
as the preprocessing layer which is a PyTorch neural network module, and :math:`g` corresponds to (part of) the neural network model that is to be trained. 

.. _loss_autoencoder:

.. rubric:: Loss function for training autoencoder 

The class :class:`colvarsfinder.core.AutoEncoderTask` trains :math:`f_{enc}:\mathbb{R}^{d_r}\rightarrow \mathbb{R}^k` and 
:math:`f_{dec}:\mathbb{R}^{k}\rightarrow \mathbb{R}^{d_r}` by the autoencoder
loss in the *transformed* space :math:`\mathbb{R}^{d_r}`:

.. math::

        & \int_{\mathbb{R}^{d_r}} |f_{dec}\circ f_{enc}(y)-y|^2  r_{\#}\mu(dy) \\
       =& \int_{\mathbb{R}^{d}} |f_{dec}\circ f_{enc}(r(x))-r(x)|^2  \mu(dx) \\
    \approx& \frac{\sum_{l=1}^{n} v_l|f_{dec}\circ f_{enc}(y_l) - y_l|^2}{\sum_{l=1}^n v_l},

where :math:`r_{\#}\mu` denotes the pushforward measure of :math:`\mu` by the map :math:`r`, and :math:`y_l = r(x_l)`.

After training, the collective variables are constructed by 

.. math::
    \xi = f_{enc}\circ r.

.. _loss_eigenfunction:

.. rubric:: Loss function for training eigenfunctions 

The class :class:`colvarsfinder.core.EigenFunctionTask` finds the leading eigenfunctions :math:`\phi_1, \phi_2, \dots, \phi_k:\mathbb{R}^d\rightarrow \mathbb{R}` of infinitesimal generator or transfer operator. 

In the generator case, the generator is assumed to be 

.. math::
    \mathcal{L} = -\nabla V \cdot \nabla f + \frac{1}{\beta} \Delta f\,,

for a test function :math:`f`, and this class computes the eigenfunctions of PDE 

.. math::

    -\mathcal{L}\phi = \lambda \phi,

corresponding to the first :math:`k` eigenvalues :math:`0 < \lambda_1 \le \lambda_2 \le \cdots \le \lambda_k`. This is done by training neural network to learn functions :math:`g_1, g_2, \cdots, g_k:\mathbb{R}^{d_r}\rightarrow \mathbb{R}` using the data-version of the loss 

.. _loss_eigen_generator:

.. math::
    \sum_{i=1}^k \omega_i  \frac{\beta^{-1} \mathbf{E}_{\mu} \big[(a \nabla f_i)\cdot \nabla f_i\big]}{\mbox{var}_{\mu} f_i} 
    + \alpha \sum_{1 \le i_1 \le i_2 \le k} \Big[\mathbf{E}_{\mu} \Big((f_{i_1}-\mathbf{E}_{\mu}f_{i_1})(f_{i_2}-\mathbf{E}_{\mu}f_{i_2})\Big) - \delta_{i_1i_2}\Big]^2,

where 

    #. :math:`\mathbf{E}_{\mu}` and :math:`\mbox{var}_{\mu}` are the expectation and variance with respect to :math:`\mu`, respectively;
    #. :math:`\alpha` is the penalty parameter;
    #. :math:`a\in \mathbb{R}^{d\times d}` is a diagonal matrix;
    #. :math:`\omega_1 \ge \omega_2 \ge \dots \ge \omega_k > 0` are :math:`k` positive constants;
    #. :math:`f_i=g_i\circ r, 1\le i \le k`.

In the transfer operator case, assume the lag-time is :math:`\tau` and the transition density at time :math:`\tau` given the state :math:`x` at time zero is :math:`p_\tau(\cdot|x)`. The loss function used to learn eigenfunctions is 

.. _loss_eigen_transfer:

.. math::
    \frac{1}{2\tau}\sum_{i=1}^k \omega_i  \frac{\mathbf{E}_{x\sim\mu, y\sim p_\tau(\cdot|x)} \big[|f_i(y)- f_i(x)|^2\big]}{\mbox{var}_{\mu} f_i} + \alpha \sum_{1 \le i_1 \le i_2 \le k} \Big[\mathbf{E}_{\mu} \Big((f_{i_1}-\mathbf{E}_{\mu}f_{i_1})(f_{i_2}-\mathbf{E}_{\mu}f_{i_2})\Big) - \delta_{i_1i_2}\Big]^2,

In practice, with weighted trajectory data :math:`x^{(1)}, \cdots, x^{(N)}` and assuming :math:`\tau=j\Delta t`, where :math:`\Delta t` is the time interval between two consecutive states and :math:`j` is an integer, then 

.. math::
    \mathbf{E}_{x\sim\mu, y\sim p_\tau(\cdot|x)} \big[|f_i(y)- f_i(x)|^2\big] \approx \frac{\sum_{l=1}^{N-j} v_l |f_i(x^{(l+j}) - f_i(x^{(l)})|^2}{\sum_{l=1}^{N-j} v_l}\,.

After training, the collective variables are constructed by 

.. math::
    \xi = (g_1\circ r, g_2\circ r, \dots, g_k\circ r)^T.


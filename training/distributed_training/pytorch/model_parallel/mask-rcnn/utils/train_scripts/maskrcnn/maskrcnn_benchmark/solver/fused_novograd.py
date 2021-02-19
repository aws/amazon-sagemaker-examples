import torch
from apex.multi_tensor_apply import multi_tensor_applier

class FusedNovoGrad(torch.optim.Optimizer):

    """Implements NovoGrad algorithm.

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused NovoGrad implements 2 fusions.

      * Fusion of the NovoGrad update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.FusedNovoGrad`'s usage is identical to any Pytorch optimizer::

        opt = apex.optimizers.FusedNovoGrad(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedNovoGrad` may be used with or without Amp.  If you wish to use :class:`FusedNovoGrad` with Amp,
    you may choose any ``opt_level``::

        opt = apex.optimizers.FusedNovoGrad(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()

    In general, ``opt_level="O1"`` is recommended.

    It has been proposed in `Jasper: An End-to-End Convolutional Neural Acoustic Model`_.
    More info: https://nvidia.github.io/OpenSeq2Seq/html/optimizers.html#novograd

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            NOT SUPPORTED now! (default: False)
        reg_inside_moment (bool, optional): whether do regularization (norm and L2)
            in momentum calculation. True for include, False for not include and
            only do it on update term. (default: False)
        grad_averaging (bool, optional): whether apply (1-beta1) to grad when
            calculating running averages of gradient. (default: True)
        norm_type (int, optional): which norm to calculate for each layer.
            2 for L2 norm, and 0 for infinite norm. These 2 are only supported
            type now. (default: 2)
        init_zero (bool, optional): whether init norm with 0 (start averaging on
            1st step) or first step norm (start averaging on 2nd step). True for
            init with 0. (default: False)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)

    .. _Jasper - An End-to-End Convolutional Neural Acoustic Model:
        https://arxiv.org/abs/1904.03288
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.,
                 amsgrad=False, reg_inside_moment=False,
                 grad_averaging=True, norm_type=2, init_zero=False,
                 set_grad_none=True):
        if amsgrad:
            raise RuntimeError('FusedNovoGrad does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        grad_averaging=grad_averaging, norm_type=norm_type,
                        init_zero=init_zero)
        super(FusedNovoGrad, self).__init__(params, defaults)
        if multi_tensor_applier.available:
            import amp_C
            # Skip buffer
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_novograd = amp_C.multi_tensor_novograd
        else:
            raise RuntimeError('apex.optimizers.FusedNovoGrad requires cuda extensions')

        self.moment_mode = 0 if reg_inside_moment else 1
        self.set_grad_none = set_grad_none

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedNovoGrad, self).zero_grad()

    def load_state_dict(self, state_dict):
        super(FusedNovoGrad, self).load_state_dict(state_dict)
        # in case exp_avg_sq is not on the same device as params, move it there
        for group in self.param_groups:
            if len(group['params']) > 0 and "exp_avg_sq" in group:
                group['exp_avg_sq'][0] = group['exp_avg_sq'][0].to(group['params'][0].device)
                group['exp_avg_sq'][1] = group['exp_avg_sq'][1].to(group['params'][0].device)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            # create lists for multi-tensor apply
            g_16, p_16, m_16 = [], [], []
            g_32, p_32, m_32 = [], [], []

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('FusedNovoGrad does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                else:
                    raise RuntimeError('FusedNovoGrad only support fp16 and fp32.')

            # we store per weight norm as one tensor for one group/precision combination
            # different from optim.Adam, we store norm here(not ^2) so we can unify calculation for norm types
            if 'exp_avg_sq' not in group:
                group['exp_avg_sq'] = [None, None]
                if group['init_zero']:
                    group['exp_avg_sq'][0] = torch.cuda.FloatTensor(len(g_16)).contiguous().fill_(0)
                    group['exp_avg_sq'][1] = torch.cuda.FloatTensor(len(g_32)).contiguous().fill_(0)
                else: # init with first step norm, so first blend have no effect
                    if group['norm_type'] == 0:
                        v_16 = [torch.max(torch.abs(g.to(torch.float32))).item() for g in g_16]
                        v_32 = [torch.max(torch.abs(g)).item() for g in g_32]
                    elif group['norm_type'] == 2:
                        v_16 = [torch.sum(torch.pow(g.to(torch.float32), 2)).sqrt().item() for g in g_16]
                        v_32 = [torch.sum(torch.pow(g, 2)).sqrt().item() for g in g_32]
                    else:
                        raise RuntimeError('FusedNovoGrad only support l2/inf norm now.')
                    group['exp_avg_sq'][0] = torch.cuda.FloatTensor(v_16)
                    group['exp_avg_sq'][1] = torch.cuda.FloatTensor(v_32)
            else:
                assert(len(g_16) == group['exp_avg_sq'][0].numel())
                assert(len(g_32) == group['exp_avg_sq'][1].numel())

            if(len(g_16) > 0):
                # print("Group LR:", group['lr'])
                multi_tensor_applier(self.multi_tensor_novograd,
                                     self._dummy_overflow_buf,
                                     [g_16, p_16, m_16],
                                     group['exp_avg_sq'][0],
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     group['step'],
                                     bias_correction,
                                     group['weight_decay'],
                                     grad_averaging,
                                     self.moment_mode,
                                     group['norm_type'])
            if(len(g_32) > 0):
                multi_tensor_applier(self.multi_tensor_novograd,
                                     self._dummy_overflow_buf,
                                     [g_32, p_32, m_32],
                                     group['exp_avg_sq'][1],
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     group['step'],
                                     bias_correction,
                                     group['weight_decay'],
                                     grad_averaging,
                                     self.moment_mode,
                                     group['norm_type'])


        return loss

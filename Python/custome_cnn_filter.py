import numpy as np
import torch
from torch.autograd import Function
import mahalanobis_transformation

# Inherit from Function
class MDT_Function(Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, a1, a2):
        trans = np.array([[a1, 0], [0, a2]])
        output = mahalanobis_transformation.MDT_connectivity(input.detach().cpu().numpy(), trans, "8-connectivity", 0)
        return torch.tensor(output)

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, a1, a2 = inputs
        ctx.save_for_backward(input, a1, a2)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, a1, a2 = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

a = np.array([[1, 2], [3, 4]])
a_tensor = torch.tensor(a)
print(a_tensor.detach().cpu().numpy())
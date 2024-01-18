from hqq.core.common.quant_utils import Quantizer, HQQBackend
import torch
from hqq.core.common.utils import opition


# Main linear layer
class HQQLinear(torch.nn.Module):
    backend = HQQBackend.PYTORCH  # Default

    def __init__(self, linear_layer, quant_config, del_orig=True):
        super().__init__()
        self.ready = False
        self.in_gpu = False
        self.device = None
        self.bias = None
        self.quant_config = quant_config
        self.set_backend(HQQLinear.backend)  # Default backend

        if linear_layer is not None:
            if linear_layer.bias:
                if opition.use_half:
                    self.bias = linear_layer.bias.half().cuda()
                else:
                    self.bias = linear_layer.bias
            self.quantize(linear_layer.weight.data, **quant_config)

        if del_orig:
            del linear_layer
        if opition.use_cuda():
            torch.cuda.empty_cache()

    # Set backends
    @classmethod
    def set_backend(cls, backend: HQQBackend):
        HQQLinear.backend = backend
        cls.forward = getattr(cls, backend.value)

    def cuda(self, device_n=0):
        if self.in_gpu:
            return
        self.W_q, self.meta = Quantizer.cuda(self.W_q, self.meta, device_n)
        if self.meta["quant_scale"]:
            self.meta["scale_q"], self.meta["meta_scale"] = Quantizer.cuda(
                self.meta["scale_q"], self.meta["meta_scale"], device_n
            )
        if self.meta["quant_zero"]:
            self.meta["zero_q"], self.meta["meta_zero"] = Quantizer.cuda(
                self.meta["zero_q"], self.meta["meta_zero"], device_n
            )

        if self.bias is not None:
            if opition.use_half:
                self.bias.half()
            self.bias = self.bias.cuda(device_n)

        self.W_q = torch.nn.Parameter(self.W_q, requires_grad=False)
        self.device = self.W_q.device
        self.in_gpu = True

    def to(self, *args, **kwargs):
        pass

    def half(self, *args, **kwargs):
        return self

    def state_dict(self):
        return {"W_q": self.W_q, "meta": self.meta, "bias": self.bias}

    def load_state_dict(self, state_dict):
        self.W_q = state_dict["W_q"]
        self.meta = state_dict["meta"]
        self.bias = state_dict["bias"] if ("bias" in state_dict) else None
        self.in_gpu = self.W_q.device.type == "cuda"
        if self.in_gpu == False:
            if opition.use_cuda:
                self.cuda()
            else:
                self.cpu()
        self.ready = True

    def quantize(self, W, weight_quant_params, scale_quant_params, zero_quant_params):
        quant_scale = scale_quant_params is not None
        quant_zero = zero_quant_params is not None

        self.in_features, self.out_features = W.t().shape

        # Quantize
        W_q, meta = Quantizer.quantize(W, **weight_quant_params)
        meta.update({"quant_scale": quant_scale, "quant_zero": quant_zero})
        if meta["quant_scale"]:
            meta["scale_q"], meta["meta_scale"] = Quantizer.quantize(
                meta["scale"], **scale_quant_params
            )
            del meta["scale"]
        if meta["quant_zero"]:
            meta["zero_q"], meta["meta_zero"] = Quantizer.quantize(
                meta["zero"], **zero_quant_params
            )
            del meta["zero"]

        self.W_q = W_q
        self.meta = meta
        if opition.use_cuda:
            self.cuda()
        else:
            self.cpu()
        self.ready = True

    def dequantize(self):
        assert self.ready, "model was not quantized"
        W_q, meta = self.W_q, self.meta

        del_keys = []
        if meta["quant_scale"]:
            meta["scale"] = Quantizer.dequantize(meta["scale_q"], meta["meta_scale"])
            del_keys.append("scale")
        if meta["quant_zero"]:
            meta["zero"] = Quantizer.dequantize(meta["zero_q"], meta["meta_zero"])
            del_keys.append("zero")

        W_est = Quantizer.dequantize(W_q, meta)

        # Cleanup
        for key in del_keys:
            del meta[key]
        return W_est

    def matmul(self, x, transpose=True):
        weight = self.dequantize()
        return torch.matmul(x, weight.t() if (transpose) else weight)

    @torch.compile()
    def matmul_compile(self, *args, **kwargs):
        return self.matmul(*args, **kwargs)

    # def forward_pytorch_backprop(self, x):
    # 	return HQQMatmulNoCacheMul.apply(x, self.matmul, self.bias)

    # def forward_pytorch_backprop_compile(self, x):
    # 	return HQQMatmulNoCacheMul.apply(x, self.matmul_compile, self.bias)

    def forward_pytorch(self, x):
        out = torch.matmul(x, self.dequantize().t())
        if self.bias is not None:
            out += self.bias
        return out

    @torch.compile()
    def forward_pytorch_compile(self, x):
        return self.forward_pytorch(x)

    ##############################################
    # Experimental
    #############################################
    # Requires building the aten backend
    # def forward_aten(self, x):
    # 	empt = torch.empty([0])
    # 	W_q  = self.W_q
    # 	meta = self.meta
    # 	bias = self.bias

    # 	W_q, W_s, W_z              = W_q,  empt if (meta['quant_scale']) else meta['scale'], empt if (meta['quant_zero']) else meta['zero']
    # 	W_shape,  W_group_size     = meta['shape'], meta['group_size']
    # 	W_nbits, W_axis, W_packing = meta['nbits'], meta['axis'], meta['packing']

    # 	if(meta['quant_scale']):
    # 		S_q, S_s, S_z              = meta['scale_q'],             meta['meta_scale']['scale'], meta['meta_scale']['zero']
    # 		S_shape, S_group_size      = meta['meta_scale']['shape'], meta['meta_scale']['group_size']
    # 		S_nbits, S_axis, S_packing = meta['meta_scale']['nbits'], meta['meta_scale']['axis'],  meta['meta_scale']['packing']
    # 	else:
    # 		S_q, S_s, S_z              = empt, empt, empt
    # 		S_shape, S_group_size      = meta['shape'], -1
    # 		S_nbits, S_axis, S_packing = -1, 0, ""

    # 	if(meta['quant_zero']):
    # 		Z_q, Z_s, Z_z              = meta['zero_q'],             meta['meta_zero']['scale'], meta['meta_zero']['zero']
    # 		Z_shape, Z_group_size      = meta['meta_zero']['shape'], meta['meta_zero']['group_size']
    # 		Z_nbits, Z_axis, Z_packing = meta['meta_zero']['nbits'], meta['meta_zero']['axis'],  meta['meta_zero']['packing']
    # 	else:
    # 		S_q, S_s, S_z              = empt, empt, empt
    # 		S_shape, S_group_size      = meta['shape'], -1
    # 		S_nbits, S_axis, S_packing = -1, 0, ""

    # 	S_group_size = 0 if (S_group_size==None) else S_group_size
    # 	Z_group_size = 0 if (Z_group_size==None) else Z_group_size

    # 	args = [x, bias if (bias is not None) else empt,
    # 			W_q, W_s, W_z, W_shape, W_group_size, W_nbits, W_axis, W_packing,
    # 			S_q, S_s, S_z, S_shape, S_group_size, S_nbits, S_axis, S_packing,
    # 			Z_q, Z_s, Z_z, Z_shape, Z_group_size, Z_nbits, Z_axis, Z_packing]

    # 	return hqq_aten.forward_with_quant(*args)

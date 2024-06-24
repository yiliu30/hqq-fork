# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#####################################################
import torch

try:
    import marlin
except Exception:
    marlin = None
from ..core.quantize import Quantizer



try:
    from intel_extension_for_transformers import qbits  # with QBits kernels ()

    QBITS_INSTALLED = True
except:
    QBITS_INSTALLED = False


BITS_DTYPE_MAPPING = {
    4: "int4_clip",
    8: "int8",
}

def convert_dtype_torch2str(dtype):
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif isinstance(dtype, str) and dtype in ["int8", "fp32", "fp16", "bf16"]:
        return dtype
    else:
        assert False, "Unsupported pytorch dtype {} to str dtype".format(dtype)


class QBitsLinear(torch.nn.Module):
    def __init__(
        self, W: torch.Tensor, scales: torch.Tensor, u=None, bias=None, groupsize=-1
    ):
        super().__init__()
        self._W = W.clone()
        self._scales = scales.clone()
        in_feats, out_feats = W.shape
        self.in_features = in_feats
        self.out_features = out_feats
        # device = W.device
        # _linear = torch.nn.Linear(m, n)
        # _linear.weight.data = W.half().t()

        # effective_groupsize = m if (groupsize == -1) else groupsize

        # _layer = marlin.Layer(m, n, groupsize=groupsize)
        # _layer.k = m
        # _layer.n = n
        # _layer.groupsize = effective_groupsize
        # _layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=device)
        # _layer.s = torch.empty(
        #     (m // effective_groupsize, n), dtype=torch.half, device=device
        # )
        # _layer.pack(_linear, scales.t())

        # self.bias = bias.half() if (bias is not None) else None
        # self.Wq_packed = _layer.B.clone()
        # self.scales = _layer.s.clone()
        # self.workspace_fp = torch.zeros(n // 128 * 16, device=device)
        # self.in_features = m
        # self.out_features = n
        # self.group_size = effective_groupsize
        # self.axis = 1
        # self.device = device
        # self.compute_dtype = torch.float16
        self.u = torch.nn.Parameter(u, requires_grad=False) if (u is not None) else None

        # del _linear, _layer
        # torch.cuda.empty_cache()
        
        # ---------------------------------------------------------------------
        # For qbits
        device = "cpu"
        self.scales = scales.clone().float().to(device)
        self.use_bf16 = qbits.check_isa_supported("AMX")
        self.scale_dtype = torch.float32
        self.zero_point = None
        self.w_bit = 4
        
        self.bias = bias
        self.group_size = groupsize if groupsize != -1 else in_feats
        self.qweight = self._pack_weight_for_qbit(W, scales)
        print(self)
    
    def _pack_weight_for_qbit(self, W, scales):
        # For Qbist.repack_quantized_weight
        #                 shape,    dtype
        # int_weight:    (K, N),    int8
        # scale:         (K//GS, N), fp32
        # zero:          (K//GS, N), int8
        assert W.shape == (self.in_features, self.out_features)
        assert scales.shape == (self.in_features//self.group_size, self.out_features), f"Expected {(self.in_features//self.group_size, self.out_features)} but got {scales.shape}"
        # TODO: What's the shape of W?
        intweight = W.to("cpu")
        scales = scales.to("cpu")
        zeros = torch.empty(0, dtype=torch.int8)
        g_idx = torch.empty(0, dtype=torch.int32)
        weight_type = BITS_DTYPE_MAPPING[self.w_bit]
        compute_type = convert_dtype_torch2str(self.scale_dtype)
        scale_type = convert_dtype_torch2str(self.scales.dtype)
        qbits_packed_qweight = qbits.repack_quantized_weight(
            intweight.contiguous(),
            scales.float().contiguous(),
            zeros,
            g_idx,
            weight_type,
            scale_type,
            compute_type,
            # BITS_DTYPE_MAPPING[self.w_bit],
            # convert_dtype_torch2str(self.scale_dtype),
            # convert_dtype_torch2str(self.scales.dtype),
            False, # self.zero_point,
            self.group_size
            )
        # revert_wei = torch.zeros(k, n, dtype=torch.float)
        # qbits.dequantize_packed_weight(qbits_packed_qweight, revert_wei, False, compute_type, weight_type, scale_type)
        # print(f"The range of revert weight: {revert_wei.min()}, {revert_wei.max()}")
        # import pdb; pdb.set_trace()
        # print(f"packed result: {qbits_packed_qweight}")
        return qbits_packed_qweight

    
    @torch.no_grad()
    def qbits_woq_linear(self, x):
        x = x.to("cpu")
        assert QBITS_INSTALLED, (
            "QBits kernels could not be loaded. "
            "Please install with `pip install intel-extension-for-transformers` and "
            "refer to the detial https://github.com/intel/intel-extension-for-transformers/blob/main/docs/qbits.md")
        input_dtype = x.dtype
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])  # convert xd to 2d
        out_2d_shape = x.shape[:-1] + (self.out_features,)

        outputs = torch.zeros(out_2d_shape, dtype=torch.float)
        bias = self.bias if self.bias is not None else torch.empty(
            0, dtype=torch.bfloat16 if self.use_bf16 else torch.float32)
        # bias =  torch.empty(0, dtype=torch.bfloat16 if self.use_bf16 else torch.float32)

        qbits.woq_linear(x,
                         self.qweight,
                         bias, outputs,
                         convert_dtype_torch2str(input_dtype),
                         BITS_DTYPE_MAPPING[self.w_bit],
                         convert_dtype_torch2str(self.scale_dtype),
                         self.zero_point
                         )
        # # check Nan
        # if torch.isnan(outputs).any():
        #     print("Nan detected in QBitsLinear forward")
        #     import pdb; pdb.set_trace()
        return outputs.view(out_shape)

    
    def __repr__(self) -> str:
        return f"QBitsLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, group_size={self.group_size})"


    @torch.jit.ignore
    def forward(self, x):
        # x (W - float_u)
        # [M, K] [K, N] -> [M, N]
        out = self.qbits_woq_linear(x)
        assert out.shape[-1] == self.out_features
        if self.u is not None:
            # [M, K] @ [1, N] -> [M, 1]
            addtion = torch.matmul(x.sum(axis=-1, keepdim=True), self.u)
            out = out + addtion
            
        # if self.bias is not None:
        #     out += self.bias
        # print(f"qbits out range: {out.min()}, {out.max()}, {out.mean()}")

        return out


def check_range(tensor, dtype):
    if dtype == torch.int8:
        assert tensor.min() >= -128, "Expected all values to be greater than -128"
        assert tensor.max() <= 127, "Expected all values to be less than 127"
    elif dtype == torch.uint8:
        assert tensor.min() >= 0, "Expected all values to be greater than 0"
        assert tensor.max() <= 255, "Expected all values to be less than 255"
    elif dtype == "int4":
        assert tensor.min() >= -8, "Expected all values to be greater than -8"
        assert tensor.max() <= 7, "Expected all values to be less than 7"
    elif dtype == "uint4":
        assert tensor.min() >= 0, "Expected all values to be greater than 0"
        assert tensor.max() <= 15, "Expected all values to be less than 15"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

# ONLY WORKS WITH AXIS=1, group_size= - 1
def patch_hqq_to_qbits(layer, patch_params=None):
    if marlin is None:
        return layer

    z_shift = 8.0
    hqq_layer = layer.linear_layer if hasattr(layer, "linear_layer") else layer

    # Check config suppport
    # if (
    #     (hqq_layer.meta["axis"] == 0)
    #     or (hqq_layer.meta["group_size"] is not None)
    #     or (hqq_layer.meta["nbits"] != 4)
    # ):
    #     print("Skipping marlin conversion for", hqq_layer.name)
    #     return layer
    bits = hqq_layer.meta["nbits"]
    unpack_func = Quantizer.unpack[hqq_layer.meta["packing"]]
    W_q_u8 = unpack_func(hqq_layer.W_q, dtype=torch.uint8)
    # assert W_q_u8.dtype == torch.uint8, f"Expected uint8, got {W_q_u8.dtype}"
    assert W_q_u8.min() >= 0, "Expected all values to be positive"
    check_range(W_q_u8, "uint4")
    W_q_s8 =  W_q_u8 - z_shift
    W_q_s8 = W_q_s8.to(torch.int8)
    assert (W_q_s8 + 8 - W_q_u8).max() == 0, "Expected W_q_s8 + 8 == W_q_u8"
    W_q_s8 = W_q_s8.t_()
    check_range(W_q_s8, "int4")
    # W_r = Quantizer.unpack[hqq_layer.meta["packing"]](
    #     hqq_layer.W_q, dtype=hqq_layer.compute_dtype
    # ).t()
    z = hqq_layer.meta["zero"]
    s = hqq_layer.meta["scale"].t()
    
    # float_zp, float_scale
    # Int_w
    # float_tensor: [K, N]
    # [K, N] = [K, N] - [1, N]   * [1, N]
    # dq = (Int_w - float_zp) * float_scale
    # [1, N] = [1, N]
    # Int_zp = round(float_zp)
    # [K, N] = [K, N] - [1, N]   * [1, N] 
    # dq = (Int_w - Int_zp) * new_float_scale

    # new_float_scale =  (Int_w - float_zp) * float_scale / (Int_w - Int_zp)
    #                    ([K, N] - [1, N] ) * [1, N]      / ([K, N]- [1, N])  ==> [K, N]
    
    
    # dq = (Int_w - float_zp) * float_scale
    # Int_zp = round(float_zp)
    # dq = (Int_w - Int_zp) * new_float_scale
    
    # W_r = (W_r - z_shift) * s
    # W_r = (W_r - z_shift) * s
    
    #          [K, N] - [K, N]    [K, N]    [1, N]     [1, N]
    # y = X @ ((W_int - z_shift + z_shift - float_z) * scale)
    # y = X @ (W_int - z_shift) * scale + X @ ((z_shift - float_z) * scale)
    #     woq kernel + small matmul
    # y = qbits.woq_linear(W_int, scale, X) + torch.matmul(X.sum(dim=-1, keepdim=True), (z_shift - float_z) * scale)

    if type(z) in [torch.Tensor, torch.nn.Parameter]:
        z = z.t()
        u = (s * (-z + z_shift)).view([1, -1])
    else:
        u = None
    # print(f"W_q_s8 shape: {W_q_s8.shape}, s shape: {s.shape} u shape: {u.shape if u is not None else None}")
    
    qbits_layer = QBitsLinear(W_q_s8, s, u=u, bias=hqq_layer.bias)
    if hasattr(layer, "linear_layer"):
        del layer.linear_layer.W_q
        del layer.linear_layer.meta
        del layer.linear_layer
        layer.linear_layer = qbits_layer
    else:
        del hqq_layer.W_q
        del hqq_layer.meta
        del hqq_layer
        layer = qbits_layer

    torch.cuda.empty_cache()

    return layer

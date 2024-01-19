from hqq.core.common.config import opition
from hqq.core.bitpack import BitPack

# from hqq.core.optimize import optimize_weights_proximal
from hqq.core.common.optim_utils import optimize_weights_proximal
from hqq.core.utils import is_divisible
import torch

from enum import Enum


class HQQBackend(Enum):
    # Name of the forward functions
    PYTORCH = "forward_pytorch"
    PYTORCH_COMPILE = "forward_pytorch_compile"
    # PYTORCH_BACKPROP         = "forward_pytorch_backprop"
    # PYTORCH_BACKPROP_COMPILE = "forward_pytorch_backprop_compile"
    # ATEN                     = "forward_aten"


class Quantizer:
    SUPPORTED_BITS = [8, 4, 3, 2]
    optimize_weights = optimize_weights_proximal

    bit_to_packing = {8: "8bit_u8", 4: "4bit_u8", 3: "3bit_32", 2: "2bit_u8"}

    pack = {
        "8bit_u8": BitPack.pack_8bit_u8,
        "4bit_u8": BitPack.pack_4bit_u8,
        "3bit_32": BitPack.pack_3bit_32,
        "2bit_u8": BitPack.pack_2bit_u8,
    }

    unpack = {
        "8bit_u8": BitPack.unpack_8bit_u8,
        "4bit_u8": BitPack.unpack_4bit_u8,
        "3bit_32": BitPack.unpack_3bit_32,
        "2bit_u8": BitPack.unpack_2bit_u8,
    }

    @classmethod
    def quantize(
        cls,
        tensor,
        nbits=4,
        channel_wise=True,
        group_size=64,
        optimize=False,
        round_zero=False,
        axis=0,
        bitpack=True,
    ):
        print(f"[Quantizer][quantize] start quantize ..... ")
        assert nbits in Quantizer.SUPPORTED_BITS, (
            "nbits=" + str(nbits) + " not supported."
        )
        assert axis in [0, 1], "axis should be either 0 or 1"
        if group_size is not None:
            assert is_divisible(tensor.numel(), group_size), (
                "group_size should be divisble by the total tensor dimensions. shape: "
                + str(tensor.shape)
                + ", group_size: "
                + str(group_size)
            )

        W = tensor.float()
        shape = W.shape

        # Reshape for grouping
        if (group_size is not None) and channel_wise:
            W = (
                W.reshape([-1, group_size])
                if (axis == 1)
                else W.reshape([group_size, -1])
            )

        # Get min/max values
        if channel_wise == False:
            _min, _max = W.min(), W.max()
            optimize = False
        else:
            _min = W.min(axis=axis, keepdim=True)[0]
            _max = W.max(axis=axis, keepdim=True)[0]

        max_v = 2**nbits - 1
        min_v = 0
        min_max = [min_v, max_v]

        # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
        scale = (max_v / (_max - _min)).clamp(
            max=2e4
        )  # clamp to avoid half-precision problems
        zero = -_min * scale

        # Round zero as in: https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py#L42C9-L42C14
        if round_zero:
            zero = torch.round(zero)

        # Fine-tune weights
        if optimize:
            scale, zero = Quantizer.optimize_weights(
                tensor=W, scale=scale, zero=zero, min_max=min_max, axis=axis
            )

        # Quantize
        scale, zero = (
            scale.clone(),
            zero.clone(),
        )  # Necessary for fake quantization backprop
        W_q = torch.round(W * scale + zero).clamp(min_max[0], min_max[1])

        # Store meta-data (we invert the scale for dequantization)
        meta = {
            "nbits": nbits,
            "group_size": group_size,
            "shape": shape,
            "scale": 1.0 / scale,
            "zero": zero,
            "axis": axis,
            "packing": Quantizer.bit_to_packing[nbits],
        }

        # Pack bits
        if bitpack:
            W_q = Quantizer.pack[meta["packing"]](W_q)
        else:
            W_q = W_q.to(tensor.dtype)
            meta["packing"] = None

        # cleanup
        del W, _min, _max
        if opition.use_cuda:
            torch.cuda.empty_cache()

        return W_q, meta

    # Main dequantization: bit_unpacking > (W_q - z)*s > reshape
    @classmethod
    def dequantize(cls, W_q, meta):
        if meta["packing"]:
            W_r = Quantizer.unpack[meta["packing"]](W_q)
            if opition.use_half:
                W_r.half()
            if (meta["group_size"] is not None) and (meta["nbits"] == 3):
                W_r = (
                    W_r[: meta["group_size"]]
                    if (meta["axis"] == 0)
                    else W_r[:, : meta["group_size"]]
                )
        else:
            if opition.use_half:
                W_r = W_q.half()
        W_r = ((W_r - meta["zero"]) * meta["scale"]).reshape(meta["shape"])
        return W_r

    @classmethod
    def to_inplace(cls, W_q, meta, device):
        W_q = W_q.to(device).contiguous()
        for key in meta:
            if type(meta[key]) == torch.Tensor:
                meta[key] = (
                    (
                        meta[key].half()
                        if meta[key].dtype == torch.float32 and opition.use_half
                        else meta[key]
                    )
                    .to(device)
                    .contiguous()
                )
        return W_q, meta

    @classmethod
    def to_ooplace(cls, W_q, meta, device):
        W_q_c = W_q.to(device).contiguous()
        meta_c = {}
        for key in meta:
            if type(meta[key]) == torch.Tensor:
                meta_c[key] = (
                    (
                        meta[key].half()
                        if meta[key].dtype == torch.float32 and opition.use_half
                        else meta[key]
                    )
                    .to(device)
                    .contiguous()
                )
            else:
                meta_c[key] = meta[key]
        return W_q_c, meta_c

    @classmethod
    def cuda(cls, W_q, meta, device_n=0):
        return Quantizer.to_inplace(W_q, meta, device="cuda:" + str(device_n))

    @classmethod
    def cpu(cls, W_q, meta):
        return Quantizer.to_ooplace(W_q, meta, device="cpu")

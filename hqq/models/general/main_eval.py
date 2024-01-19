from hqq.core.quantize import BaseQuantizeConfig
from hqq.core.common.config import opition
from hqq.core.common.utils import dump_elapsed_time
from hqq.core.common.modules import HQQLinear


######################################
# If use GPU
opition.use_cuda = True
if opition.use_half:
    assert opition.use_cuda, "Please use `cuda` if use `half`."
######################################


# HQQLinear(linear_layer, quant_config)


# def replace_linear_weight_only_int8_per_channel(module):
#     for name, child in module.named_children():
#         if isinstance(child, nn.Linear):
#             setattr(module, name, WeightOnlyInt8Linear(child.in_features, child.out_features))
#         else:
#             replace_linear_weight_only_int8_per_channel(child)
from torch import nn


def patch_func(module, name, quant_config):
    print(f"patch for module : {name}")

    def one_of_tag(name):
        for tag in opition.linear_tags:
            if tag in name:
                return True
        return False

    if one_of_tag(name):
        print(f"[patch_func] Replace {name} with HQQ")
        return HQQLinear(module, quant_config)
    else:
        print(f"[patch_func] {name} is a linear module but not replace it")
        if opition.use_half:
            return module.half().cuda()
        else:
            return module


def replace_linear_with_hqq_linear(module, quant_config, prefix=[]):
    for name, child in module.named_children():
        cur_child_name = ".".join(prefix + [name])
        if isinstance(child, nn.Linear):
            print(f"{cur_child_name} is a instance of linear")
            setattr(module, name, patch_func(child, cur_child_name, quant_config))
        elif name in opition.nonlinear_tags:
            print(f"{cur_child_name} is one of non linear tags")
            setattr(module, name, child.half().cuda() if opition.use_half else child)
        else:
            replace_linear_with_hqq_linear(child, quant_config, prefix=prefix + [name])


@dump_elapsed_time("Quantize model...")
def quant_model(model, quant_config):
    replace_linear_with_hqq_linear(model, quant_config)


def get_float_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


LLAMA_2_7B_HF_NAME = "llama_2_7b_hf"
GPT_J_6B = "gpt-j-6B"

mode_id_mapping = {
    LLAMA_2_7B_HF_NAME: "/models/Llama-2-7b-hf",
    GPT_J_6B: "/models/gpt-j-6B",
}

linear_tags_mapping = {
    LLAMA_2_7B_HF_NAME: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    GPT_J_6B: [
        "k_proj",
        "v_proj",
        "q_proj",
        "out_proj",
        "fc_in",
        "fc_out",
    ],
}

nonlinear_tags_mapping = {
    LLAMA_2_7B_HF_NAME: [
        "lm_head",
        "embed_tokens",
        "norm",
        "rotary_emb",
        "act_fn",
        "input_layernorm",
        "post_attention_layernorm",
    ],
    GPT_J_6B: [
        "lm_head",
        "wte",
        "ln_1",
        "act",
        "ln_f",
    ],
}


def main(args):
    model_name = args.model_name

    model_id = mode_id_mapping[model_name]
    opition.linear_tags = linear_tags_mapping[model_name]
    opition.nonlinear_tags = nonlinear_tags_mapping[model_name]
    model, tokenizer = get_float_model(model_id)
    quant_config = BaseQuantizeConfig(
        nbits=args.nbits, group_size=args.group_size, quant_scale=args.quant_scale
    )
    print(quant_config)
    print(model)
    # import pdb; pdb.set_trace()
    if opition.use_cuda:
        model.to("cuda")
    from hqq.core.common.cuda_utils import see_memory_usage

    see_memory_usage("After loading float model")
    if args.quant:
        quant_model(model, quant_config)
    see_memory_usage("After replace the HQQ")
    ##########################################################################################################
    ###  Eval code
    ##########################################################################################################
    from hqq.core.common.eval_ppl import eval_wikitext2

    eval_wikitext2(model=model, tokenizer=tokenizer, verbose=True)
    see_memory_usage("After Evaluation ... ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize a model.")
    parser.add_argument(
        "--quant", action="store_false", default=True, help="quant or not"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=LLAMA_2_7B_HF_NAME,
        choices=[LLAMA_2_7B_HF_NAME, GPT_J_6B],
        help="Model name to be quantized.",
    )
    parser.add_argument("--nbits", type=int, default=4, choices=[8, 4, 3, 2], help="")
    parser.add_argument("--group_size", type=int, default=64, help="")
    parser.add_argument(
        "--quant_scale", action="store_true", default=False, help="quant_scale or not"
    )
    args = parser.parse_args()
    main(args)


"""
test cmds:

p hqq/models/general/main_eval.py --model_name "gpt-j-6B" --nbits 2 --group_size 16 --quant_scale &> ./test/gpt-j-6B-2bits-g16-default_quant_scale-ppl


"""

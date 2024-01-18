from hqq.core.quantize import BaseQuantizeConfig
from hqq.core.common.utils import opition
from hqq.core.common.modules import HQQLinear

quant_config = BaseQuantizeConfig(nbits=4, group_size=64)


######################################
# If use GPU
opition.use_cuda = False
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

linear_tags = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

nonlinear_tags = [
    "lm_head",
    "embed_tokens",
    "norm",
    "rotary_emb",
    "act_fn",
    "input_layernorm",
    "post_attention_layernorm",
]


def one_of_tag(name):
    for tag in linear_tags:
        if tag in name:
            return True
    return False


def patch_func(module, name):
    print(f"patch for module : {name}")
    if one_of_tag(name):
        return HQQLinear(module, quant_config)
    else:
        if opition.use_half:
            return module.half().cuda()
        else:
            return module


def replace_linear_with_hqq_linear(module, quant_config, prefix=[]):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, patch_func(child, ".".join(prefix)))
        elif name in nonlinear_tags:
            setattr(module, name, child.half().cuda() if opition.use_half else child)
        else:
            replace_linear_with_hqq_linear(child, quant_config, prefix=prefix + [name])


def get_float_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


model_id = (
    "/home/yliu7/workspace/gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/tmp"
)
model, tokenizer = get_float_model(model_id)
if opition.use_cuda:
    model.to("cuda")
replace_linear_with_hqq_linear(model, quant_config)

##########################################################################################################
###
###  Chat Code
###
##########################################################################################################
import transformers
from threading import Thread

from sys import stdout


def print_flush(data):
    stdout.write("\r" + data)
    stdout.flush()


# Adapted from https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/app.py
def process_conversation(chat):
    system_prompt = chat["system_prompt"]
    chat_history = chat["chat_history"]
    message = chat["message"]

    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )
    conversation.append({"role": "user", "content": message})

    result = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if opition.use_cuda:
        result = result.to("cuda")
    return result


def chat_processor(chat, max_new_tokens=100, do_sample=True):
    tokenizer.use_default_system_prompt = False
    streamer = transformers.TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_params = dict(
        {"input_ids": process_conversation(chat)},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=0.90,
        top_k=50,
        temperature=0.6,
        num_beams=1,
        repetition_penalty=1.2,
    )

    t = Thread(target=model.generate, kwargs=generate_params)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        print_flush("".join(outputs))

    return outputs


###################################################################################################

outputs = chat_processor(
    {
        "system_prompt": "You are a helpful assistant.",
        "chat_history": [],
        "message": "How can I build a car?",
    },
    max_new_tokens=1000,
    do_sample=False,
)

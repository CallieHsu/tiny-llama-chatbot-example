from transformers import AutoConfig, AutoTokenizer, TextStreamer
from optimum.intel import OVQuantizer
from optimum.intel.openvino import OVModelForCausalLM
import openvino as ov
from pathlib import Path
import shutil
import nncf
import gc
import sys
import time
import argparse

############################
## Read your private model
############################
# from huggingface_hub import login
# access_token_read = "Your_huggingface_token"
# login(token = access_token_read)

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-m',
                    '--model_id',
                    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    required=False,
                    type=str,
                    help='Huggingface model name')
parser.add_argument('-o',
                    '--output',
                    default='compressed_model',
                    required=False,
                    type=str,
                    help='Path to save the Openvino IR model')
parser.add_argument('-p',
                    '--precision',
                    required=False,
                    default="INT8",
                    type=str,
                    choices=["FP16", "INT8", "INT4"],
                    help='FP16, INT8 or INT4')
parser.add_argument('-d',
                    '--device',
                    required=False,
                    default="CPU",
                    type=str,
                    choices=["CPU", "GPU"],
                    help='CPU or GPU')
args = parser.parse_args()

def convert_to_fp16():
    if (fp16_model_dir / "openvino_model.xml").exists():
        return
    ov_model = OVModelForCausalLM.from_pretrained(
        model_id, export=True
    )
    ov_model.half()
    ov_model.save_pretrained(fp16_model_dir)
    del ov_model
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(fp16_model_dir)

def convert_to_int8():
    if (int8_model_dir / "openvino_model.xml").exists():
        return
    int8_model_dir.mkdir(parents=True, exist_ok=True)
    if fp16_model_dir.exists():
        ov_model = OVModelForCausalLM.from_pretrained(fp16_model_dir, compile=False)
    else:
        ov_model = OVModelForCausalLM.from_pretrained(
            model_id, export=True, compile=False
        )
        ov_model.half()
    quantizer = OVQuantizer.from_pretrained(ov_model)
    quantizer.quantize(save_directory=int8_model_dir, weights_only=True)
    del quantizer
    del ov_model
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(int8_model_dir)

def convert_to_int4():
    model_compression_params = {
        "mode": nncf.CompressWeightsMode.INT4_ASYM,
        "group_size": 128,
        "ratio": 0.8,
    }
    if (int4_model_dir / "openvino_model.xml").exists():
        return
    int4_model_dir.mkdir(parents=True, exist_ok=True)
    if not fp16_model_dir.exists():
        model = OVModelForCausalLM.from_pretrained(
            model_id, export=True, compile=False
        ).half()
        model.config.save_pretrained(int4_model_dir)
        ov_model = model._original_model
        del model
        gc.collect()
    else:
        ov_model = ov.Core().read_model(fp16_model_dir / "openvino_model.xml")
        shutil.copy(fp16_model_dir / "config.json", int4_model_dir / "config.json")

    compressed_model = nncf.compress_weights(ov_model, **model_compression_params)
    ov.save_model(compressed_model, int4_model_dir / "openvino_model.xml")
    del ov_model
    del compressed_model
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(int4_model_dir)

model_id = args.model_id
output_dir = args.output

fp16_model_dir = Path(output_dir) / "FP16"
int8_model_dir = Path(output_dir) / "INT8"
int4_model_dir = Path(output_dir) / "INT4"

convert_to_fp16()
convert_to_int8()
convert_to_int4()

model_to_run = args.precision
device = args.device

if model_to_run == "INT4":
    model_dir = int4_model_dir
elif model_to_run == "INT8":
    model_dir = int8_model_dir
else:
    model_dir = fp16_model_dir
print(f"Loading model from {model_dir}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
except:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(model_dir)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}  # To disable model caching, set CACHE_DIR to an empty string.
try:
    print(" --- use local model --- ")
    model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device=device,
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir),
    )
except:
    print(" --- use remote model --- ")
    model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device=device,
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir),
        export=True
    )

max_new_tokens = 128
temperature, top_p, top_k, repetition_penalty = 0.1, 1, 50, 1.1
streamer = TextStreamer(tokenizer,
                        timeout=30.0,
                        skip_prompt=True,
                        skip_special_tokens=True
                        )

generate_kwargs = dict(
    max_new_tokens=max_new_tokens,
    pad_token_id=tokenizer.pad_token_id,
    temperature=temperature,
    do_sample=temperature > 0.0,
    top_p=top_p,
    top_k=top_k,
    repetition_penalty=repetition_penalty,
    streamer=streamer,
)

def chat_prompt(text):
    return [
        {"role": "system", "content": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.",},
        {"role": "user", "content": text}
    ]

while True:
    user_input = input("You: ").strip()
    input_text = tokenizer.apply_chat_template(chat_prompt(user_input), tokenize=True, add_generation_prompt=True, return_tensors="pt")
    print("Model:")
    start = time.perf_counter()
    response = model.generate(input_ids=input_text, **generate_kwargs)
    end = time.perf_counter()
    new_tokens = len(response[0]) - len(input_text[0])
    print(f"[info] Generation of {new_tokens} tokens took {end - start:.3f} s on {device} ({(new_tokens / (end - start)):.3f} token/s) \n")


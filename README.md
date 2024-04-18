# TinyLlama chatbot example
- openvino
- Large Language Model (LLM): [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

## Environment setup
### 1. Create a Virtual Environment
```
python3 -m venv openvino_env
```
### 2. Activate the Environment
```
source openvino_env/bin/activate
```
### 3. Clone the Repository
```
git clone https://github.com/CallieHsu/tiny-llama-chatbot-example.git
cd tiny-llama-chatbot-example
```
### 4. Install the Packages
```
python3 -m pip install --upgrade pip
pip install wheel setuptools
pip install -r requirements.txt
```
## Convert LLM to Openvino IR format & inference
### - openvino_IR_chat.py
1. Download LLM from Hugging Face.
2. Convert LLM to openvino format and quantize to `FP16`, `INT8` and `INT4`.
3. Inference with the device `CPU` or `GPU` and precision `FP16`, `INT8` or `INT4`.

##### Arguments
- `-m`: model id of Hugging Face.
- `-o`: output folder name of quantization model.
- `-p`: precision of model, option: `FP16`, `INT8`, `INT4`
- `-d`: device, option: `CPU`, `GPU`.

##### Usage
```
python openvino_IR_chat.py -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 -o compressed_model -p INT8 -d CPU
```

### - openvino_IR_chat_dialogue_record.py
- `openvino_IR_chat.py` with dialogue record.

##### Arguments
same as `openvino_IR_chat.py`

##### Usage
```
python openvino_IR_chat_dialogue_record.py -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 -o compressed_model -p INT8 -d CPU
```

## Gradio Chatbot demo
##### Arguments
- `-m`: quantized openvino format model folder

```
python gradio_chatbot_tiny-llama-shuttle_demo.py -m compressed_model/INT8
```

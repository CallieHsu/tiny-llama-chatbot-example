# TinyLlama chatbot example
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
```
python openvino_IR_chat.py -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 -o compressed_model -p INT8 -d CPU
```
```
python openvino_IR_chat_dialogue_record.py -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 -o compressed_model -p INT8 -d CPU
```

## Gradio Chatbot demo
```
python gradio_chatbot_tiny-llama-shuttle_demo.py -m compressed_model/INT8
```

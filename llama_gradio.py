import os
from llama_cpp import Llama
import gradio as gr

# 从环境变量读取
MODEL_PATH = os.environ.get("MODEL", "./models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf")
N_THREADS = int(os.environ.get("THREADS", "4"))
PORT = int(os.environ.get("PORT", "8000"))

# 加载模型
model = Llama(model_path=MODEL_PATH, n_ctx=1024, n_threads=N_THREADS)

conversation_history = []

def chat(user_input, chat_history):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})

    prompt_text = "你是一个中文助手，请用中文回答用户的问题。\n"
    for msg in conversation_history:
        if msg["role"] == "user":
            prompt_text += f"用户: {msg['content']}\n"
        else:
            prompt_text += f"助手: {msg['content']}\n"

    output = model(prompt_text, max_tokens=200, stop=["用户:", "助手:"])
    answer = output["choices"][0]["text"].strip()
    conversation_history.append({"role": "assistant", "content": answer})
    
    # 更新聊天历史记录用于界面显示
    chat_history.append((user_input, answer))
    return "", chat_history

def reset_chat():
    global conversation_history
    conversation_history = []
    return []

# Gradio 前端
with gr.Blocks() as demo:
    gr.Markdown("## TinyLlama 中文助手（CPU 版）")
    chatbot = gr.Chatbot()
    txt = gr.Textbox(show_label=False, placeholder="输入中文问题，回车发送")
    with gr.Row():
        send_btn = gr.Button("发送")
        reset_btn = gr.Button("重置")
    
    txt.submit(chat, [txt, chatbot], [txt, chatbot])
    send_btn.click(chat, [txt, chatbot], [txt, chatbot])
    reset_btn.click(reset_chat, None, chatbot)

demo.launch(server_name="0.0.0.0", server_port=PORT)
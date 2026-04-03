import gradio as gr
import requests
import json

URL = "https://detker-roberta-qa.hf.space/predict"

def predict(question, context):
    params = {
        "question": question,
        "context": context
    }
    response = requests.get(URL, params=params)
    response_json = json.loads(response.text)
    return response_json

demo = gr.Interface(
    fn=predict,
    fn=predict,
    inputs=[
        gr.Textbox(label='Question', placeholder='Enter your question here'),
        gr.Textbox(label='Context', placeholder='Enter the context here')
    ],
    outputs=gr.Textbox(label='Answer'),
    title='Question Answering with RoBERTa',
    description='Provide a question and context to get an answer using the RoBERTa model.',
    flagging_mode='never'
)

demo.launch()

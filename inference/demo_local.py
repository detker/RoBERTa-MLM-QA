import gradio as gr
from inference import Inference

inference = Inference()
inference.load_model()

demo = gr.Interface(
    fn=inference.predict,
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

import gradio as gr
import torch
from transformers import RobertaTokenizerFast
from hf_pretrained_model import RobertaConfigHF, RobertaForQAHF
from transformers import AutoModel, AutoConfig

AutoConfig.register('roberta-qa', RobertaConfigHF)
AutoModel.register(RobertaConfigHF, RobertaForQAHF)
config = AutoConfig.from_pretrained('detker/roberta-qa-125M')
model = AutoModel.from_pretrained('detker/roberta-qa-125M',
                                  trust_remote_code=True,
                                  use_safetensors=True,
                                  config=config)
tokenizer = RobertaTokenizerFast.from_pretrained(config.hf_model_name)
model.eval()

def inference(question, context):
    inputs = tokenizer(
        text=question,
        text_pair=context,
        max_length=config.context_length,
        truncation='only_second',
        return_tensors='pt'
    )

    with torch.no_grad():
        start_logits, end_logits = model(inputs)

    start_token_idx = start_logits.squeeze().argmax().item()
    end_token_idx = end_logits.squeeze().argmax().item()

    tokens = inputs['input_ids'].squeeze()[start_token_idx:end_token_idx+1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()

    return answer

demo = gr.Interface(
    fn=inference,
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

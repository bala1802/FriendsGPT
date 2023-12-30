import torch
import gradio as gr

import data_utils
from gpt_language_model import GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

inference_model = GPTLanguageModel()

inference_model.load_state_dict(torch.load('model/friendsGPT.pth', map_location=torch.device('cpu')))
inferenceModel = inference_model.to(device)

def generate():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output = data_utils.decode(inferenceModel.generate(context, max_new_tokens=500)[0].tolist())
    return output
    

demo = gr.Interface(fn=generate, inputs=None, outputs="text", title="F.R.I.E.N.D.S GPT", thumbnail="FRIENDS.jpg")
    
if __name__ == "__main__":
    demo.launch(show_api=False, share=True)
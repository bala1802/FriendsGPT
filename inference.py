import torch

import data_utils
from gpt_language_model import GPTLanguageModel

inference_model = GPTLanguageModel()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "mps"

inference_model.load_state_dict(torch.load('model/friendsGPT.pth', 
                                           map_location=torch.device('cpu')))
inference_model = inference_model.to(device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(data_utils.decode(inference_model.generate(context, max_new_tokens=500)[0].tolist()))
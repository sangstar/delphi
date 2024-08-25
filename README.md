# delphi
An implementation of an inference service that runs GPT-NeoX models, focusing on simplicity and readability.

## Quickstart
Install the package:

```bash
pip install .
```

Loading a model and generating is straightforward:

```python
from delphi import GPTNeoX

model_id = "EleutherAI/pythia-1b"
model = GPTNeoX.from_pretrained(model_id)

prompt = "Pythia, high priestess of the Temple"
output = model.generate(prompt, max_new_tokens=50, temperature=1, top_k=50)

print(output)

'''
Pythia, high priestess of the Temple of Apollo at Delphi. She would be the 
object of a Roman poet's dream.

He saw her as an old, wizened woman, bent almost double. 
She leaned heavily on a pair of sticks. The only part of her body that was
'''
```

## Additional Features
The `scripts/` directory has a few example use cases of the `delphi` module, such as a
chatbot in `chatbot.py`, and a planned API server. 

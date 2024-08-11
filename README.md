# delphi
An implementation of an inference service that runs GPT-NeoX models.

## Quickstart
Install the package:

```bash
pip install .
```

Loading a model and generating is as easy as this:

```python
from delphi import GPTNeoX

model_id = "EleutherAI/pythia-410m"
model = GPTNeoX.from_pretrained(model_id)

prompt = "It began to rain heavily"
output = model.generate(prompt, max_new_tokens=50, temperature=1, top_k=50)

print(output)

'''
It began to rain heavily.  A gust of wind swept across their
faces, and for a moment the two looked like a pair of demons, but then
came a momentary pause of perfect stillness.

The first light was breaking in upon this dreamlike
'''
```


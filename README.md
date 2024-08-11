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

prompt = "The knight drew his sword"
output = model.generate(prompt, max_new_tokens=50, temperature=1, top_k=50)

print(output)

'''
The knight drew his sword from his scabbard. "Good,"
he said, and the giant's hand moved away from him.

"Let's not fight. I cannot be forced to fight. Will you give me your
garrison now that he is
'''
```


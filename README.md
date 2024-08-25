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
chatbot in `chatbot.py`, and a planned API server. An example chat is here with
`EleutherAI/pythia-1b` (which clearly needs to refresh itself on the history of
men's tennis!). The chatbot streams tokens as they're generated.

In this case, this model isn't a dedicated chat model, so the flow of 
conversation works best as if the user and itself are communicating "together":

```
Warming up... Ready!

-> Roger Federer is the greatest

GPTNeoX:
 and highest level tennis player ever and is the first male
 tennis player to win Wimbledon on three consecutive occasions. He won two 
 Grand Slam titles at the event and one additional Wimbledon title, the 
 French Open. Federer

-> has a rivalry with Nadal

GPTNeoX:
 that dates back to their first year together in 1997.
 Both Federer and Roger Federer are considered to be at the top 
 of men's professional tennis.

History

Early years (2000–2003)
Federer was

-> 
```

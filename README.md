## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:

  In modern natural language processing (NLP), the identification of key entities such as names, organizations, locations, and temporal expressions from unstructured text is essential for downstream applications like information retrieval, document classification, and knowledge graph construction.
However, traditional rule-based or shallow machine learning approaches often lack the contextual understanding required for accurate entity recognition in complex sentences.
This project aims to build a prototype NER system using a fine-tuned BART (Bidirectional and Auto-Regressive Transformer) model, which effectively captures both contextual dependencies and semantic relationships. The model output is integrated with an interactive Gradio interface that allows real-time user testing, visualization, and performance assessment.

### DESIGN STEPS:

#### STEP 1:

Choose a fine-tuned BART model for Named Entity Recognition and set up the development environment with required libraries like transformers, torch, and gradio.

#### STEP 2:

Connect the model via the Hugging Face API or local pipeline, implement a function to process text inputs and retrieve entity predictions, and refine the token outputs for accurate labeling.

#### STEP 3:

Design an interactive Gradio interface with text input and highlighted output for entity visualization, add example prompts for testing, and deploy the prototype for real-time user evaluation.

### PROGRAM:
```
import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']
API_URL = os.environ['HF_API_NER_BASE']

def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters:
        data["parameters"] = parameters
    response = requests.post(ENDPOINT_URL, headers=headers, json=data)
    if response.status_code != 200:
        raise ValueError(f"Model API returned {response.status_code} error: {response.text}")
    text = response.text.strip()
    # Try single valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try line-separated JSON chunks
        for line in text.splitlines():
            try:
                return json.loads(line)
            except Exception:
                continue
        raise ValueError(f"Unable to parse model output: {text}")

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)
    return merged_tokens

def ner(input_text):
    output = get_completion(input_text, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    
    # Convert to format expected by gr.HighlightedText
    # Format: {"text": str, "entities": [{"entity": label, "start": int, "end": int}]}
    entities = []
    for token in merged_tokens:
        # Extract clean entity label (remove B- or I- prefix)
        entity_label = token['entity'].replace('B-', '').replace('I-', '')
        entities.append({
            "entity": entity_label,
            "start": token['start'],
            "end": token['end']
        })
    
    return {"text": input_text, "entities": entities}

gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with dslim/bert-base-NER",
    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
    allow_flagging="never",
    examples=[
            "Barack Obama was born in Hawaii and served as the President of the United States.",
            "Elon Musk founded SpaceX and Tesla in California.",
            "Amazon was established in 1994 by Jeff Bezos in Seattle."
    ]
)
demo.launch(share=True, server_port=int(os.environ['PORT4']))
```

### OUTPUT:

<img width="1175" height="626" alt="exp 5 gen" src="https://github.com/user-attachments/assets/31283283-c97a-4262-a0be-a6374b9b269b" />

### RESULT:

The NER prototype using a fine-tuned BART model and Gradio was successfully developed to identify and highlight entities like names, places, and organizations.
It performs real-time text analysis with accurate and interactive visualization of recognized entities.

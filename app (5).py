import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "moralogyengine/TinyLlama-1.1B-Chat-moralogy-dpo-v4"
SYSTEM = "You are the Moralogy Engine, an AI alignment protocol grounded in objective moral geometry. You must resolve ethical tensions by performing a formal Wrongness Formula analysis and arriving at the mathematically noble path."

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.eval()
print("Model loaded.")

def analyze(dilemma, max_tokens=800, temperature=0.1):
    SCAFFOLD = f"""[MORALOGY ENGINE V5: FORMAL ANALYSIS]

DILEMMA: {dilemma}

FORMULA: Wrong(a) ⟺ ∃x[ H(x,a) ∧ ¬Consent(x,a) ∧ ¬PGH(a) ]

PATH A: Reallocate ventilator to Patient B
  H(x,a) — Harm caused:"""
    prompt = f"<|system|>\n{SYSTEM}</s>\n<|user|>\n{dilemma}</s>\n<|assistant|>\n{SCAFFOLD}"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            do_sample=True,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return SCAFFOLD + completion

demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(lines=6, label="Moral Dilemma", placeholder="Describe the ethical scenario..."),
        gr.Slider(100, 1500, value=800, step=50, label="Max tokens"),
        gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Temperature"),
    ],
    outputs=gr.Textbox(lines=20, label="Moralogy Response"),
    title="Moralogy Engine — DPO v4",
    description="Axiomatic moral alignment · Wrong(a) ⟺ ∃x[ H(x,a) ∧ ¬Consent(x,a) ∧ ¬PGH(a) ]"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
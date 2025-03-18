# Fine-Tuning DeepSeek-R1-Distill-Qwen-1.5B with KTO  

This repository provides a step-by-step guide for fine-tuning **DeepSeek-R1-Distill-Qwen-1.5B** using **Kahneman-Tversky Optimization (KTO)**. KTO enhances model alignment with human preferences without relying on expensive preference datasets. This implementation leverages **Unsloth**, **LoRA**, and **Flash Attention 2** for efficient and scalable training.

---

## **Table of Contents**  
1. [Installation](#installation)  
2. [Import Necessary Libraries](#import-necessary-libraries)  
3. [Model Loading and Configuration](#model-loading-and-configuration)  
4. [Dataset Preparation](#dataset-preparation)  
5. [Model Training Setup](#model-training-setup)  
6. [Model Saving and Export](#model-saving-and-export)  
7. [Inference and Response Generation](#inference-and-response-generation)  
8. [Conclusion](#conclusion)  

---

## **1. Installation**  

Install and upgrade the necessary libraries for model training and inference.  

```bash
# Install package manager
pip install pip3-autoremove  

# Remove existing PyTorch versions
pip-autoremove torch torchvision torchaudio -y  

# Install latest PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121  

# Install Unsloth for fast fine-tuning
pip install unsloth  
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git  
```

### **Install Flash Attention 2** (for GPUs with CUDA capability **≥ 8**, e.g., A100, H100)
```python
import torch
if torch.cuda.get_device_capability()[0] >= 8:
    !pip install --no-deps packaging ninja einops "flash-attn>=2.6.3"
```

---

## **2. Import Necessary Libraries**  

```python
import torch
from unsloth import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import KTOTrainer
```

---

## **3. Model Loading and Configuration**  

- Load the **DeepSeek-R1-Distill-Qwen-1.5B** model.  
- Use **4-bit quantization** to reduce memory usage.  
- Apply a **default chat template** if missing.

```python
model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
```

---

## **4. Dataset Preparation**  

- Load the **KTO dataset**.  
- Apply the **chat template** to ensure consistent formatting.  
- Select a subset for **faster training**.  

```python
def format_chat_template(example):
    return {"formatted_text": f"<system>{example['system']}</system> <user>{example['user']}</user> <assistant>{example['assistant']}</assistant>"}

dataset = load_dataset("your_dataset_name")  # Replace with actual dataset
dataset = dataset.map(format_chat_template)
dataset = dataset.select(range(1000))  # Use a subset for quick fine-tuning
```

---

## **5. Model Training Setup**  

- Configure **LoRA** for parameter-efficient fine-tuning.  
- Set up **KTO training** with hyperparameters.  
- Monitor **GPU memory usage**.  

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

trainer = KTOTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
    logging_dir="./logs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_steps=500,
    learning_rate=2e-5,
)

# Print GPU memory usage
print(torch.cuda.memory_summary())
```

---

## **6. Model Saving and Export**  

- Save the **fine-tuned model** and tokenizer.  
- Provide options for **16-bit and 4-bit merging**.  
- Convert to **GGUF format** for use with **llama.cpp**.  

```python
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

# Convert to GGUF format (optional)
!python -m llama.cpp.convert --model fine_tuned_model --format gguf
```

To upload the model to **Hugging Face Hub**:  
```python
from huggingface_hub import notebook_login
notebook_login()

model.push_to_hub("your-username/deepseek-kto-finetuned")
tokenizer.push_to_hub("your-username/deepseek-kto-finetuned")
```

---

## **7. Inference and Response Generation**  

- Define a function for generating responses from the **fine-tuned model**.  
- Test it on a **set of sample prompts**.  

```python
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example
print(generate_response("What are the benefits of KTO in AI training?"))
```

---

## **8. Conclusion**  

This fine-tuning approach demonstrates the power of **Kahneman-Tversky Optimization (KTO)** in aligning large language models with **human preferences**.  

### **Key Takeaways:**  
✅ **Efficient Fine-Tuning**: Used **4-bit quantization** and **LoRA adapters** to optimize memory and compute usage.  
✅ **Enhanced Human Alignment**: KTO ensures that models **generate human-like responses** without requiring expensive preference data.  
✅ **Scalability**: The method is adaptable to **larger models** and **diverse datasets**.  

### **Future Work:**  
🚀 Apply KTO to **larger-scale models**.  
🚀 Explore its integration with **other alignment techniques**.  
🚀 Optimize training efficiency for **real-world deployment**.  

---

## **🔗 References**  
- [Unsloth](https://github.com/unslothai/unsloth)  
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)  
- [KTO Training Paper](https://arxiv.org/abs/2402.09957)  

Feel free to contribute or raise issues! 🚀

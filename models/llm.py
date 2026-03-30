import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLMWrapper:
    def __init__(
            self,
            model_name: str = "meta-llama/Llama-2-13b-chat-hf",
            load_4bit: bool = True,
            max_seq_len: int = 2048,
            device: str = "cuda"
    ):
        self.device = device
        self.max_seq_len = max_seq_len

        # 4bit量化配置
        if load_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            self.bnb_config = None

        # 加载模型与分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, temperature: float = 0.1, max_new_tokens: int = 512) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def score(self, prompt: str) -> float:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)
        score = torch.mean(log_probs).item()
        return torch.sigmoid(torch.tensor(score)).item()

    def generate_with_score(self, prompt: str) -> tuple[str, float]:
        response = self.generate(prompt)
        score = self.score(prompt + response)
        return response, score


if __name__ == "__main__":
    llm = LLMWrapper(model_name="meta-llama/Llama-2-7b-chat-hf", load_4bit=True)
    test_prompt = "Hello, who are you?"
    response, score = llm.generate_with_score(test_prompt)
    print(f"Response: {response}")
    print(f"Score: {score:.4f}")
    print("✅ LLMWrapper 测试通过！")
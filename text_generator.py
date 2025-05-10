import requests
import json
import os
from typing import Optional, Dict, Any
import time
from datetime import datetime

class AdvancedTextGenerator:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        self.conversation_history = []
        self.system_prompt = """You are a helpful, knowledgeable, and friendly AI assistant. 
        Provide detailed, accurate, and well-structured responses. 
        When asked about topics, include relevant facts, examples, and context."""
        
    def generate_text(self, 
                     prompt: str, 
                     max_tokens: int = 2000,
                     temperature: float = 0.7,
                     top_p: float = 0.9) -> Dict[str, Any]:
        """
        Generate text using Ollama's API with advanced parameters
        """
        url = f"{self.base_url}/generate"
        
        # Add system prompt and conversation history
        full_prompt = f"{self.system_prompt}\n\n"
        for msg in self.conversation_history[-5:]:  # Keep last 5 messages for context
            full_prompt += f"{msg['role']}: {msg['content']}\n"
        full_prompt += f"User: {prompt}\nAssistant:"
        
        data = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": result["response"]})
            
            return {
                "response": result["response"],
                "tokens_used": result.get("eval_count", 0),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def save_conversation(self, filename: str = "conversation_history.json"):
        """Save conversation history to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
    
    def load_conversation(self, filename: str = "conversation_history.json"):
        """Load conversation history from a file"""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)

def main():
    print("Welcome to the Advanced Text Generator!")
    print("Available commands:")
    print("- Type your prompt to generate text")
    print("- Type 'save' to save conversation history")
    print("- Type 'load' to load previous conversation")
    print("- Type 'clear' to clear conversation history")
    print("- Type 'quit' to exit")
    
    generator = AdvancedTextGenerator()
    
    while True:
        user_input = input("\nEnter your prompt: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'save':
            generator.save_conversation()
            print("Conversation history saved!")
            continue
        elif user_input.lower() == 'load':
            generator.load_conversation()
            print("Conversation history loaded!")
            continue
        elif user_input.lower() == 'clear':
            generator.conversation_history = []
            print("Conversation history cleared!")
            continue
        
        print("\nGenerating response...")
        start_time = time.time()
        result = generator.generate_text(user_input)
        end_time = time.time()
        
        if "error" in result:
            print(f"\nError: {result['error']}")
        else:
            print("\nGenerated Response:")
            print(result["response"])
            print(f"\nGeneration time: {end_time - start_time:.2f} seconds")
            print(f"Tokens used: {result.get('tokens_used', 'N/A')}")

if __name__ == "__main__":
    main() 
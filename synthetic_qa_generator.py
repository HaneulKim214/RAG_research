#!/usr/bin/env python3
"""
Synthetic QA Dataset Generation for RAG Evaluation

Following HuggingFace methodology: https://huggingface.co/learn/cookbook/en/rag_evaluation

This module implements advanced synthetic QA generation with:
- LLM-based question generation with critique agents
- Multiple question types (factual, analytical, comparative, strategic, risk)
- LLM-as-a-judge evaluation for answer quality
- Comprehensive RAG system evaluation
"""

import numpy as np
import os
import json
import random
import time
from typing import List, Dict, Optional
from datetime import datetime

# LLM and text processing
from huggingface_hub import InferenceClient
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

# Import our clean prompt templates
from prompt_templates import (
    BASE_QA_GENERATION_PROMPT,
    RELEVANCE_CRITIQUE_PROMPT,
    CLARITY_CRITIQUE_PROMPT,
    COMPLEXITY_CRITIQUE_PROMPT,
    LLM_JUDGE_EVALUATION_PROMPT,
    QUESTION_TYPES
)


class LLMClient:
    """Unified LLM client supporting multiple providers"""
    
    def __init__(self, provider="google", model_name=None, api_key=None):
        self.provider = provider
        
        if provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model_name = model_name or "gpt-3.5-turbo"
        elif provider == "huggingface":
            self.client = InferenceClient(
                model=model_name or "mistralai/Mixtral-8x7B-Instruct-v0.1",
                timeout=120,
            )
            self.model_name = model_name or "mistralai/Mixtral-8x7B-Instruct-v0.1"
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.client = ChatGoogleGenerativeAI(
                model=model_name or "gemini-2.5-flash",
                google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
                temperature=0.1
            )
            self.model_name = model_name or "gemini-2.5-flash"
        else:
            raise ValueError("Provider must be 'openai', 'huggingface', or 'google'")
    
    def call_llm(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Call LLM with error handling and retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response.choices[0].message.content.strip()
                
                elif self.provider == "huggingface":
                    response = self.client.post(
                        json={
                            "inputs": prompt,
                            "parameters": {"max_new_tokens": max_tokens, "temperature": temperature},
                            "task": "text-generation",
                        },
                    )
                    result = json.loads(response.decode())[0]["generated_text"]
                    # Remove the prompt from the response
                    if result.startswith(prompt):
                        result = result[len(prompt):].strip()
                    return result
                
                elif self.provider == "google":
                    response = self.client.invoke(prompt)
                    return response.content.strip()
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e


class SyntheticQAGenerator:
    """Generate synthetic QA pairs using LLM with quality filtering"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    def generate_qa_pair(self, context: str, question_type: str) -> Dict:
        """Generate a single QA pair from context"""
        prompt = BASE_QA_GENERATION_PROMPT.format(
            question_type=question_type,
            context=context
        )
        
        try:
            response = self.llm_client.call_llm(prompt, max_tokens=500, temperature=0.8)
            
            # Parse the response
            if "Question:" in response and "Answer:" in response:
                question_part = response.split("Question:")[1].split("Answer:")[0].strip()
                answer_part = response.split("Answer:")[1].strip()
                
                return {
                    "question": question_part,
                    "answer": answer_part,
                    "question_type": question_type,
                    "context": context,
                    "generation_successful": True
                }
            else:
                return {"generation_successful": False, "error": "Failed to parse response"}
                
        except Exception as e:
            return {"generation_successful": False, "error": str(e)}
    
    def critique_qa_pair(self, qa_pair: Dict, critique_type: str) -> Dict:
        """Apply quality critique to QA pair"""
        if not qa_pair.get("generation_successful", False):
            return {"passed": False, "reason": "Generation failed"}
        
        # Select appropriate critique prompt from imported templates
        critique_prompts = {
            "relevance": RELEVANCE_CRITIQUE_PROMPT,
            "clarity": CLARITY_CRITIQUE_PROMPT,
            "complexity": COMPLEXITY_CRITIQUE_PROMPT
        }
        
        prompt = critique_prompts[critique_type].format(
            question=qa_pair["question"],
            answer=qa_pair["answer"],
            context=qa_pair.get("context", ""),
            question_type=qa_pair.get("question_type", "")
        )
        
        try:
            response = self.llm_client.call_llm(prompt, max_tokens=200, temperature=0.3)
            
            # Parse response
            passed = "PASS" in response.upper()
            reason = ""
            if "Reason:" in response:
                reason = response.split("Reason:")[1].strip()
            
            return {"passed": passed, "reason": reason, "critique_type": critique_type}
            
        except Exception as e:
            return {"passed": False, "reason": f"Critique failed: {str(e)}", "critique_type": critique_type}
    
    def generate_filtered_qa_pairs(self, chunks: List, n_generations_per_type: int = 5, 
                                 apply_critiques: bool = True) -> List[Dict]:
        """Generate QA pairs with quality filtering"""
        
        all_qa_pairs = []
        
        for question_type in QUESTION_TYPES.keys():
            print(f"\n🔄 Generating {question_type} questions...")
            
            successful_pairs = []
            attempts = 0
            max_attempts = n_generations_per_type * 3  # Allow more attempts than needed
            
            while len(successful_pairs) < n_generations_per_type and attempts < max_attempts:
                attempts += 1
                
                # Sample random chunk
                chunk = random.choice(chunks)
                context = chunk.page_content
                
                # Generate QA pair
                qa_pair = self.generate_qa_pair(context, question_type)
                
                if not qa_pair.get("generation_successful", False):
                    print(f"  ❌ Generation failed: {qa_pair.get('error', 'Unknown error')}")
                    continue
                
                # Apply critiques if enabled
                if apply_critiques:
                    critiques_passed = []
                    for critique_type in ["relevance", "clarity", "complexity"]:
                        critique_result = self.critique_qa_pair(qa_pair, critique_type)
                        critiques_passed.append(critique_result["passed"])
                        
                        if not critique_result["passed"]:
                            print(f"  ⚠️  Failed {critique_type}: {critique_result['reason']}")
                            break
                    
                    if not all(critiques_passed):
                        continue
                
                # Add metadata
                qa_pair.update({
                    "chunk_id": chunks.index(chunk),
                    "generation_attempt": attempts,
                    "critiques_applied": apply_critiques
                })
                
                successful_pairs.append(qa_pair)
                print(f"  ✅ Generated {question_type} question {len(successful_pairs)}/{n_generations_per_type}")
            
            all_qa_pairs.extend(successful_pairs)
            print(f"📊 Completed {question_type}: {len(successful_pairs)}/{n_generations_per_type} successful")
        
        return all_qa_pairs


class LLMJudgeEvaluator:
    """LLM-as-a-judge evaluator following HuggingFace methodology"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def evaluate_answer(self, question: str, generated_answer: str, reference_answer: str) -> Dict:
        """Evaluate a generated answer against reference using LLM-as-a-judge"""
        prompt = LLM_JUDGE_EVALUATION_PROMPT.format(
            instruction=question,
            response=generated_answer,
            reference_answer=reference_answer
        )
        
        try:
            response = self.llm_client.call_llm(prompt, max_tokens=300, temperature=0.1)
            
            # Parse feedback and score
            if "[RESULT]" in response:
                parts = response.split("[RESULT]")
                feedback = parts[0].replace("Feedback:", "").strip()
                score_text = parts[1].strip()
                
                # Extract numeric score
                score = None
                for char in score_text:
                    if char.isdigit():
                        score = int(char)
                        break
                
                if score is None:
                    score = 1  # Default to lowest score if parsing fails
                    
                # Normalize score to 0-1 range
                normalized_score = (score - 1) / 4
                
                return {
                    "feedback": feedback,
                    "raw_score": score,
                    "normalized_score": normalized_score,
                    "evaluation_successful": True
                }
            else:
                return {
                    "feedback": "Failed to parse evaluation",
                    "raw_score": 1,
                    "normalized_score": 0.0,
                    "evaluation_successful": False
                }
                
        except Exception as e:
            return {
                "feedback": f"Evaluation failed: {str(e)}",
                "raw_score": 1,
                "normalized_score": 0.0,
                "evaluation_successful": False
            }


def save_synthetic_dataset(qa_dataset: List[Dict], chunks: List, filename: str = "data/synthetic_qa_lges_advanced.json"):
    """Save the generated synthetic dataset with metadata"""
    
    if not qa_dataset:
        print("⚠️  No QA dataset to save")
        return
        
    # Create dataset metadata
    type_counts = {}
    for qa in qa_dataset:
        qtype = qa["question_type"]
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    dataset_metadata = {
        "name": "LGES_Advanced_Synthetic_QA_Dataset",
        "description": "Advanced synthetic QA dataset generated using LLM with critique filtering",
        "created_date": datetime.now().isoformat(),
        "generation_method": "LLM-based with critique agents",
        "total_questions": len(qa_dataset),
        "question_types": type_counts,
        "source_text": "all_about_lges_text (2Q audit report + news summary)",
        "chunk_settings": {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "total_chunks": len(chunks)
        },
        "evaluation_ready": True
    }
    
    # Prepare complete dataset
    complete_dataset = {
        "metadata": dataset_metadata,
        "qa_pairs": qa_dataset,
        "chunks": [
            {
                "chunk_id": i, 
                "content": chunk.page_content, 
                "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
            } 
            for i, chunk in enumerate(chunks)
        ]
    }
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(complete_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Advanced synthetic dataset saved to: {filename}")
    print(f"📊 Contains {len(qa_dataset)} questions across {len(type_counts)} types")
    return filename


def main():
    """Main function to demonstrate usage"""
    print("🚀 Synthetic QA Dataset Generation")
    print("=" * 50)
    
    # Load data
    print("📄 Loading LGES data...")
    with open("data/processed/2025_2Q_LGES_audit_report_summary.txt", 'r') as f:
        lges_2025_2q_summ = f.read()
    
    with open("data/processed/2025_09_11_LGES_news_summary.txt", "r") as f:
        lges_2025_09_01_news_summary = f.read()
    
    all_about_lges_text = lges_2025_2q_summ + "\n"+ lges_2025_09_01_news_summary
    
    # Create chunks
    print("📦 Creating chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.create_documents([all_about_lges_text])
    
    # Initialize LLM client and generator
    print("🤖 Initializing LLM client...")
    llm_client = LLMClient(provider="google")  # Change as needed
    qa_generator = SyntheticQAGenerator(llm_client)
    
    # Generate synthetic dataset
    print("🎯 Generating synthetic QA dataset...")
    synthetic_qa_dataset = qa_generator.generate_filtered_qa_pairs(
        chunks=chunks,
        n_generations_per_type=3,  # Small number for demo
        apply_critiques=True
    )
    
    # Save dataset
    if synthetic_qa_dataset:
        save_synthetic_dataset(synthetic_qa_dataset, chunks)
        
        # Display summary
        type_counts = {}
        for qa in synthetic_qa_dataset:
            qtype = qa["question_type"]
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        print(f"\n✅ Generation completed!")
        print(f"📈 Total questions: {len(synthetic_qa_dataset)}")
        print(f"📋 By type: {type_counts}")
    else:
        print("❌ No questions generated")


if __name__ == "__main__":
    main()


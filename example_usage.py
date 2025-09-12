#!/usr/bin/env python3
"""
Example usage of the Synthetic QA Generator

This shows how to use the clean, modular synthetic QA generation system
with prompt templates properly organized.
"""

from synthetic_qa_generator import LLMClient, SyntheticQAGenerator, save_synthetic_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from prompt_templates import QUESTION_TYPES

def main():
    print("🎯 Example: Synthetic QA Generation with Clean Architecture")
    print("=" * 60)
    
    # Step 1: Load your data
    print("📄 Loading data...")
    with open("data/processed/2025_2Q_LGES_audit_report_summary.txt", 'r') as f:
        text_data = f.read()
    
    print(f"✅ Loaded {len(text_data):,} characters")
    
    # Step 2: Create chunks
    print("📦 Creating chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.create_documents([text_data])
    print(f"✅ Created {len(chunks)} chunks")
    
    # Step 3: Initialize LLM client
    print("🤖 Initializing LLM...")
    # Change provider as needed: "openai", "huggingface", "google"
    llm_client = LLMClient(provider="google")
    print(f"✅ Using {llm_client.provider} - {llm_client.model_name}")
    
    # Step 4: Initialize QA generator (uses clean prompt templates)
    print("🎯 Initializing QA generator...")
    qa_generator = SyntheticQAGenerator(llm_client)
    print(f"✅ Generator ready with {len(QUESTION_TYPES)} question types")
    
    # Step 5: Generate synthetic QA pairs
    print("🚀 Generating synthetic QA dataset...")
    print("📝 Question types:", list(QUESTION_TYPES.keys()))
    
    synthetic_qa_dataset = qa_generator.generate_filtered_qa_pairs(
        chunks=chunks,
        n_generations_per_type=2,  # Small number for demo
        apply_critiques=True  # Enable quality filtering
    )
    
    # Step 6: Display results
    if synthetic_qa_dataset:
        print(f"\n✅ Successfully generated {len(synthetic_qa_dataset)} QA pairs!")
        
        # Show summary by type
        type_counts = {}
        for qa in synthetic_qa_dataset:
            qtype = qa["question_type"]
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        print("📊 Questions by type:")
        for qtype, count in type_counts.items():
            print(f"  {qtype}: {count}")
        
        # Show sample
        print("\n🔍 Sample generated QA pair:")
        sample_qa = synthetic_qa_dataset[0]
        print(f"Type: {sample_qa['question_type']}")
        print(f"Q: {sample_qa['question']}")
        print(f"A: {sample_qa['answer'][:100]}...")
        
        # Step 7: Save dataset
        print("\n💾 Saving dataset...")
        filename = save_synthetic_dataset(synthetic_qa_dataset, chunks)
        print(f"✅ Dataset saved to: {filename}")
        
    else:
        print("❌ No QA pairs were generated")
    
    print("\n🎉 Example completed!")
    print("💡 Next steps:")
    print("  1. Adjust n_generations_per_type for more questions")
    print("  2. Experiment with different LLM providers")
    print("  3. Customize prompt templates in prompt_templates.py")
    print("  4. Use the dataset to evaluate your RAG system")

if __name__ == "__main__":
    main()


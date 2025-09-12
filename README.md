# RAG Experiments & Research Project

A comprehensive Retrieval-Augmented Generation (RAG) research project focused on experimenting with different RAG components and evaluation methodologies using financial documents.

## 🏗️ Project Architecture

This project implements a complete RAG pipeline with the following components:

```
Raw Documents → Data Extraction → Chunking → Indexing → Retrieval → Generation → Evaluation
     ↓               ↓             ↓          ↓          ↓           ↓           ↓
   PDF/Web       Text/Tables   Text Chunks  Vector DB  Retrieved   LLM Answer  Quality
   Sources      OCR/Docling   Various Sizes  (FAISS)   Contexts    Generation  Metrics
```

### 1. Data Extraction Pipeline (`00_Data_extraction.ipynb`)
- **PDF Processing**: PyMuPDF, PyPDF2 for text extraction
- **OCR Integration**: Pytesseract for image-based text extraction
- **Table Extraction**: Docling integration for structured data
- **Web Scraping**: YouTube transcript extraction for financial news
- **Multi-modal Support**: Handles text, images, and tables from financial reports

**Key Features:**
- Automated text extraction from LG Energy Solution audit reports
- OCR fallback for image-heavy pages
- YouTube financial news analysis with transcript processing
- Clean text preprocessing and formatting

###  2. Advanced Chunking Strategies (`01_Chunking.ipynb`)
- **Recursive Character Text Splitter**
- [ ] **Sentence-Aware Chunking**: Grammar-preserving text segmentation
- **Semantic Chunking**: Similarity-based intelligent splitting
- [ ] **Late Chunking**: Long-context embedding optimization
- [ ] **Contextual Chunking**: Context-aware chunk boundaries
 

### 3. Vector Indexing & Embedding (`03_Indexing.ipynb`)
- **Multiple Index Types**:
  - **Flat Index**: Basic L2 similarity search
  - **IVF Flat**: Inverted file index with clustering
  - **HNSW**: Hierarchical Navigable Small World graphs
- **Vector Storage**: FAISS integration with LangChain
- **Persistent Storage**: Serialized indexes for reuse

**Performance Metrics:**
- Vector dimensions: 768
- Index types compared for speed vs accuracy trade-offs
- Scalable storage solutions implemented

### 4. Retrieval & Generation (`04_retrieval_and_generation.ipynb`)
- **Multi-Vector Store Support**: Seamless switching between index types
- **Similarity Search**: Configurable k-parameter retrieval
- **Context Assembly**: Retrieved chunks formatting for LLM input
- **Generation Pipeline**: Google Gemini integration
- **Response Quality**: Structured answer generation

### 5. Comprehensive Evaluation Framework (`99_evaluation.ipynb`)
- **Synthetic QA Generation**
- **Multi-Criteria Evaluation**:
  - **Groundedness**: Answer accuracy vs context
  - **Relevance**: Question quality assessment
  - **QA Pair Quality**: Overall coherence evaluation
- **LLM-as-Judge**: Automated quality scoring (1-5 scale)
- **Retrieval Metrics**: Hit rate, precision, recall analysis
- **End-to-End RAG Evaluation**: Complete pipeline assessment

**Evaluation Components:**
- Pydantic models for robust data validation
- Multiple LLM provider support (OpenAI, Google, HuggingFace)
- Comprehensive scoring with detailed feedback
- Integration guides for existing RAG systems


## 🛠️ Technology Stack

### Core Libraries:
- **LangChain**
- **FAISS**
- **LLMs**: OpenAI, Google, HuggingFace
- **Pydantic**: Data validation and type safety
- **PyMuPDF/PyPDF2**: PDF processing
- **Pytesseract**: OCR capabilities
- **Docling**: Advanced document processing

### Data & ML:
- **NumPy/Pandas**: Data manipulation
- **Scikit-learn**: Evaluation metrics
- **Pickle**: Model serialization
- **Jupyter**: Interactive development

## 🎯 Key Experiments Completed

### 1. ✅ Chunking Strategy Analysis
- Compared grammar-aware vs character-based chunking
- Analyzed impact on retrieval quality
- **Finding**: Character-based chunking with overlap performs better for most use cases

### 2. ✅ Vector Index Comparison
- Benchmarked Flat, IVF, and HNSW indexes
- Speed vs accuracy trade-off analysis
- **Finding**: HNSW optimal for production, Flat for development

### 3. ✅ Synthetic QA Quality Assessment
- Implemented HuggingFace evaluation methodology
- Multi-criteria filtering with LLM judges
- **Finding**: Quality filtering essential for reliable evaluation datasets

### 4. ✅ End-to-End RAG Evaluation
- Complete pipeline performance measurement
- Retrieval and generation quality metrics
- **Finding**: Context quality more important than retrieval quantity

## 📈 Current Status & Metrics

- **Documents Processed**: 3 financial reports + news content
- **Chunks Generated**: ~500 text segments
- **Vector Embeddings**: 768-dimensional Google embeddings
- **QA Pairs Created**: High-quality synthetic evaluation dataset
- **Evaluation Coverage**: Complete RAG pipeline assessment
- **Code Quality**: Production-ready with error handling

## 🚀 Next Steps & Future Work

### Planned Experiments:
- [ ] **Needle in Haystack**: Long-context retrieval effectiveness
- [ ] **Multi-lingual RAG**: Cross-language retrieval performance
- [ ] **Query Routing**: Intelligent query classification and routing
- [ ] **Hybrid Search**: Combining dense and sparse retrieval
- [ ] **Advanced Generation**: RAG with reasoning capabilities

### Technical Improvements:
- [ ] **Streaming Responses**: Real-time answer generation
- [ ] **Caching Layer**: Response caching for common queries
- [ ] **A/B Testing**: Systematic component comparison
- [ ] **Production Deployment**: Scalable inference pipeline
- [ ] **Monitoring**: Performance and quality tracking

### Research Directions:
- [ ] **Contextual Embeddings**: Document-aware chunk encoding
- [ ] **Multi-modal RAG**: Image and table integration
- [ ] **Adaptive Chunking**: Dynamic chunk size optimization
- [ ] **Query Understanding**: Intent classification and expansion
- [ ] **Feedback Learning**: Human preference integration

## 📁 Project Structure

```
RAG_experiments/
├── 00_Data_extraction.ipynb    
├── 01_Chunking.ipynb           
├── 02_query_routing_and_translation.ipynb 
├── 03_Indexing.ipynb         
├── 04_retrieval_and_generation.ipynb    
├── 99_evaluation.ipynb      
├── prompt_templates.py        
├── requirements.txt         
├── data/                     
│   ├── raw/                  
│   ├── processed/       
│   ├── chunks/              
│   ├── vector_stores/    
│   
└── venv_rag/               
```

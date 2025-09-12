from langchain.prompts import ChatPromptTemplate


# SYNTHETIC QA GENERATION PROMPTS
class QAGenerationPrompts:
    """Class containing all prompts for synthetic QA generation following HuggingFace RAG evaluation methodology."""
    
    # Base QA Generation Prompt
    BASE_QA_GENERATION = ChatPromptTemplate.from_template("""
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::""")

    GROUNDEDNESS_CRITIQUE_PROMPT = ChatPromptTemplate.from_template("""
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """)

    RELEVANCE_CRITIQUE_PROMPT = ChatPromptTemplate.from_template("""
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """)

    QA_PAIR_CRITIQUE_PROMPT = ChatPromptTemplate.from_template("""
You will be given a question and an answer.
Your task is to provide a 'total rating' representing how well the question and answer pair is answerable from the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question and answer pair is not answerable at all from the context, and 5 means that the question and answer pair is clearly and unambiguously answerable from the context.

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

Question: {question}\n
Context: {context}\n
Answer: {answer}\n

Answer:::
""")


    # Question Types Configuration
    QUESTION_TYPES = {
        "factual": "Direct facts, dates, numbers, or specific information",
        "analytical": "Analysis, interpretation, or synthesis requiring reasoning", 
        "comparative": "Comparisons between different aspects, periods, or entities",
        "strategic": "Business strategy, plans, or strategic positioning",
        "risk": "Challenges, risks, or potential issues and uncertainties"
    }


financial_report_analysis_prompt = ChatPromptTemplate.from_template("""
You are a senior sell-side equity research analyst and valuation specialist with deep accounting knowledge (IFRS) and 10+ years' experience.
Produce a rigorous, data-backed, step-by-step financial analysis and valuation of the uploaded financial report.
Leverage 

Required deliverables & analysis structure (follow exactly; show work, cite pages):
1) Executive Snapshot (one page)
   - Short business description, FY revenue, EBIT, Net income, Net debt, Market cap (if available).
   - 2–3 key risks from MD&A/notes with exact PDF page citations.

2) Peer / relative valuation
   - Find comptetitors to {company_name}.
   - Produce peer table with EV/EBITDA, P/E, EV/Sales (latest reported and TTM where available). If external market data is used, label it as external and cite source + date.

3) Future prospect of company, its research area, etc...

4) Risk, such as regulations it will be facing

Please provide a detailed, accurate answer based on given context. Be sure to provide strong reasoning
that backs your answer. Also provide reference if you retrieved data from external sources.
""")

youtube_news_summary_prompt = ChatPromptTemplate.from_template("""
You are a financial news analyst and political analyst for individual equity investors.

Inputs:
- Company (optional): {company_name}
- Output language: {target_language}  # "en" for English, "ko" for Korean
- Videos (one or more entries with URL, PublishedAt, Transcript text; Title optional):
{videos_block}

Instructions:
- Read the provided transcripts carefully; base your summary primarily on transcript content rather than only titles.
- Weigh recency: prefer newer information when there are conflicts; call out outdated items explicitly with dates.
- Summarize key topics in the videos, then analyze how those topics may impact the company (near-term and mid-term).
- Extract investor-relevant details when present: catalysts, guidance/outlook, management quotes, numbers (revenue, EPS, margin, units),
  regulatory/legal items, macro impacts, competitive dynamics, product/tech updates, valuation/price targets.
- Deduplicate across videos; consolidate overlapping facts. If claims conflict, present both and mark uncertainty.
- If a video is unavailable or lacks usable content/transcript, list it under Unprocessed with a short reason.
- If target_language is "ko", write entirely in Korean; otherwise, write in English.

Deliverable (use these sections):
1) Executive Summary
2) Dated Key Points 
3) Impact on Company (near-term vs long-term)
4) Risks & Uncertainties
5) Actionable Takeaways
6) Sources (URL with date; include title if provided)
""")
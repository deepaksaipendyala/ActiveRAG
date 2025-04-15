class Prompt:
    template = {
        'cot':
"""
Efficiently solve the following question step-by-step, clearly breaking down your reasoning.

Question: {question}

Generation Format:
Reasoning process:
1. [Concise critical reasoning step]
2. [Next critical step]
...
Answer: [Integrated, concise, and reliable response leveraging multi-agent assimilation and accommodation, focusing on accuracy and reduced hallucinations]

Additionally, explicitly outline:
- Essential unfamiliar concepts identified:
- Relevant foundational and advanced insights integrated:
- Essential logical elements utilized:
- Essential factual corrections applied:
""",

'anchoring': [
"""
You're enhancing an AI's knowledge efficiently. Quickly identify:
Question: {question}

Relevant Passages:
{passages}

Tasks:
1. Selectively highlight crucial concepts or terms potentially unfamiliar to the AI.
2. Provide succinct explanations, focusing on information essential for comprehension.

Your objective: Fill critical knowledge gaps efficiently to ensure accuracy and minimize hallucinations.
""",
""" 
Here's a language model's initial reasoning:

Question: {question}
Answer: {reply}

Identified unfamiliar concepts:
{unknown_knowledge_reply}

Tasks:
1. Refine the original reasoning by integrating critical concepts selectively labeled for maximum impact.
2. Provide a precise revised answer enhancing accuracy and reducing hallucinations.

Generation Format:
Enhanced reasoning:
1. [Essential step]
2. [Next essential step]
...
Answer: [Optimized concise answer with minimized hallucinations]
"""
        ],

'associate': [
"""
You're rapidly deepening an AI's understanding:
Question: {question}

Relevant Passages:
{passages}

Tasks:
1. Select foundational and relevant advanced information selectively labeled for clarity and impact.
2. Efficiently link concepts to deepen understanding without redundancy.

Your goal: Enhance comprehension and accuracy swiftly, minimizing hallucinations.
""",
"""
Here's an initial AI-generated reasoning:

Question: {question}
Answer: {reply}

Relevant foundational and advanced insights:
{recite_knowledge_reply}

Tasks:
1. Strengthen the original reasoning by connecting selectively labeled foundational and advanced concepts.
2. Provide a refined, accurate answer reducing hallucinations.

Generation Format:
Enhanced reasoning:
1. [Concise critical reasoning step]
2. [Next critical step]
...
Answer: [Reliable and accurate answer with reduced hallucinations]
"""
        ],

'logician': [
"""
You specialize in efficient logical reasoning:
Question: {question}

Relevant Passages:
{passages}

Tasks:
1. Selectively identify vital logical structures or causal relationships.
2. Extract strictly necessary information for robust reasoning.

Goal: Enhance logical reasoning capabilities efficiently, ensuring reliability and minimizing hallucinations.
""",
""" 
Here's initial reasoning from a language model:

Question: {question}
Answer: {reply}

Essential logical elements identified:
{logic_knowledge_reply}

Tasks:
1. Refine the reasoning precisely using selectively labeled logical connections.
2. Provide a robustly reasoned revised answer with reduced hallucinations.

Generation Format:
Enhanced logical reasoning:
1. [Critical logical step]
2. [Next essential step]
...
Answer: [Reliable logically consistent answer with minimized hallucinations]
"""
        ],
'cognition': [
"""
You're optimizing factual accuracy and minimizing hallucinations:
Question: {question}

Relevant Authoritative Passages:
{passages}

Tasks:
1. Quickly identify directly relevant factual details selectively labeled for correcting misconceptions.
2. Provide clear, concise corrections to prevent inaccuracies.

Goal: Rapidly ensure accurate answers, enhancing reliability and minimizing hallucinations.
""",
""" 
Here's a language model's original reasoning:

Question: {question}
Answer: {reply}

Identified essential factual corrections:
{fact_knowledge_reply}

Tasks:
1. Revise the original reasoning briefly, using selectively labeled essential corrections.
2. Provide an accurate, concise revised answer to enhance reliability and reduce hallucinations.

Generation Format:
Factually optimized reasoning:
1. [Key corrected reasoning step]
2. [Next essential correction]
...
Answer: [Factually concise, accurate, and reliable answer with minimized hallucinations]
"""
        ]
    }

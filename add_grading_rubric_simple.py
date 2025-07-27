#!/usr/bin/env python3
"""
Add Grading Rubric to Pinecone Index (Simplified)
Adds the detailed Ethics in the Workplace Midterm Grading Rubric to the RAG system
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Get environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    print("❌ Error: Please set PINECONE_API_KEY and OPENAI_API_KEY environment variables")
    print("   You can set them in your deployment platform or create a .env file locally")
    exit(1)

def generate_embedding(text, openai_client):
    """Generate embedding using OpenAI"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Failed to generate embedding: {e}")
        return None

def add_midterm_rubric():
    """Add the detailed midterm grading rubric to the RAG system"""
    
    print("📝 Adding Ethics in the Workplace Midterm Grading Rubric")
    print("="*70)
    
    # Initialize Pinecone and OpenAI
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        index = pc.Index("professorjames-experiment-grading")
        print("✅ Connected to Pinecone and OpenAI")
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        return
    
    # Define the rubric content in structured chunks
    rubric_chunks = [
        {
            'title': 'Ethics Midterm - General Instructions and Penalties',
            'content': '''
ETHICS IN THE WORKPLACE MIDTERM GRADING RUBRIC

GENERAL INSTRUCTIONS & PENALTIES:

Failure to use theories from both Chapter 2 (Principles) and Chapter 3 (Consequences) in the essay section: If a student attempts to use theories from only one chapter for both essays, the penalty is applying the full grade for both essay answers, but then halving the score of the lower-scoring essay.

OVERALL ESSAY STRUCTURE & CLARITY (Applies to both Part 1 and Part 2):

Defining Theories: For every theory used, the student must define it clearly at the start (e.g., "Utilitarianism is the ethical theory that states..."). This demonstrates understanding for someone unfamiliar with the subject (like a roommate).

Paragraph Structure: Clear, clean paragraphs for each part of the answer are expected. Good paragraph structure helps organize thoughts and leads to more fully and coherently developed answers. Essays with one or two long paragraphs tend to be disorganized and unclear.

Considering Multiple Viewpoints/Arguments: For long answer questions, well-developed and persuasive arguments are expected. Considering more than one way, arguing in favor of a position, or considering opposing viewpoints is helpful.

Coherence and Development: Answers should be well-developed, coherent, and avoid "gabbledegook".
''',
            'content_type': 'grading_rubric',
            'source': 'Ethics Midterm Rubric',
            'course_id': 1,
            'assignment_id': 1
        },
        {
            'title': 'Ethics Midterm - Part 1 Short Answer Questions',
            'content': '''
PART 1: SHORT ANSWER QUESTIONS (Choose 4 of 6)

Each question is worth 5 points.

GRADING CRITERIA FOR EACH QUESTION (0-5 points):

5 points (Excellent): Provides a clear, concise, and accurate answer directly addressing the prompt. Demonstrates a strong understanding of the relevant ethical concepts and their distinctions/applications.

3-4 points (Good): Provides a mostly accurate answer but might lack some clarity, conciseness, or depth. Shows a general understanding of the concept.

1-2 points (Developing): Attempts to answer the question but demonstrates significant misunderstanding or provides an irrelevant/incomplete response.

0 points (Unsatisfactory): No answer, or the answer is entirely incorrect/off-topic.

SPECIFIC EXPECTATIONS PER QUESTION:

1. What's the difference between honesty and fidelity? 
Expected Answer: Honesty is being true to others (giving a correct representation of the world). Fidelity is being true to oneself and one's own oaths/contracts. A failure of honesty is misrepresenting facts to others; a failure of fidelity is not upholding a commitment to oneself or one's agreement.

2. Why might a libertarian disobey zoning laws? 
Expected Answer: Libertarianism is a subset of Rights Theory, where the key idea is freedom maximization as long as it doesn't interfere with others' freedom. Libertarians believe property is an extension of themselves. They would dislike zoning laws as "capricious or exterior restrictions" on their land use, believing they should be free to do what they wish with their property up to the point of interfering with others.

3. Cite a quick example from your own life where you used the "veil of ignorance" theory of fairness. 
Expected Answer: Student provides a personal example (e.g., dividing resources like cookies or pizza) where they made a decision without knowing their own position to ensure fairness. The key idea is that fairness is a product of the process when one is ignorant of whether the decision will benefit them.

4. What is a global ethics, and why is it important for the consequentialist? 
Expected Answer: For a consequentialist, Global Ethics means considering all happiness and suffering an act will cause as far into the future as possible and for as many people as possible. It doesn't necessarily mean "the whole world," but rather covering everyone affected into the future.

5. What is the invisible hand argument justifying egoism as preferable to utilitarianism? 
Expected Answer: The Invisible Hand argument suggests that the best way to help others (even if you're a utilitarian or altruist) is to try to help yourself (be an egoist). By pursuing self-interest (e.g., a pizza restaurant making better pizza at a lower price to get rich), you inadvertently benefit others.

6. According to monetized utilitarianism, how much happiness does a cup of Starbucks coffee provide? 
Expected Answer: According to monetized utilitarianism, the happiness derived from an item is measured by the amount of money one is willing to pay for it. So, if a person pays $5 for coffee, it provides $5 worth of happiness for them. The problem is that it is difficult to put a price on things like friendship and love.
''',
            'content_type': 'grading_rubric',
            'source': 'Ethics Midterm Rubric',
            'course_id': 1,
            'assignment_id': 1
        },
        {
            'title': 'Ethics Midterm - Part 2 Essay Overview',
            'content': '''
PART 2: ESSAY QUESTIONS (Respond to Case 1, then Choose Case 2 or Case 3)

Each case is worth 40 points. Score depends on effective use of ethical theories and a variety of theories.

One essay must core around a Chapter 2 theory (Principles); the other must core around a Chapter 3 theory (Consequences). 

GRADING CRITERIA FOR EACH ESSAY (0-40 points):

A. Theory Definition & Application (15 points):
13-15 points (Excellent): Clearly and accurately defines the core ethical theories used. Applies theories directly and appropriately to the case, demonstrating a sophisticated understanding of how they inform the ethical justification. Effectively distinguishes between hard and soft utilitarianism if applicable.

9-12 points (Good): Defines theories with minor inaccuracies or omissions. Applies theories reasonably well but may lack some depth or precision in application.

5-8 points (Developing): Attempts to define and apply theories but shows significant misunderstanding or misapplication.

0-4 points (Unsatisfactory): Theories are not defined, or their application is irrelevant/incorrect.

B. Argument Development & Persuasiveness (15 points):
13-15 points (Excellent): Presents a well-developed, persuasive argument for the chosen position, clearly justifying answers in terms of ethical theories. Demonstrates coherent thought process, potentially setting theories against each other or exploring nuances within a single theory (e.g., Rights Theory arguing against itself in Case 1, or the long-term vs. short-term utilitarian view in Case 3). Considers opposing viewpoints or multiple angles of the issue.

9-12 points (Good): Argument is generally well-structured but may lack some depth, persuasiveness, or consistency. May touch on multiple viewpoints but not fully develop them.

5-8 points (Developing): Argument is unclear, poorly structured, or not well-supported by ethical reasoning. May fail to address opposing viewpoints adequately.

0-4 points (Unsatisfactory): No discernible argument, or argument is illogical/irrelevant.

C. Variety & Integration of Theories (10 points):
9-10 points (Excellent): Effectively uses a variety of different theories (as appropriate to the chosen case and the prompt's requirement for Chapter 2/3 theories). Seamlessly integrates multiple theories to build a comprehensive ethical analysis, even if they sometimes contradict.

6-8 points (Good): Uses more than one theory, but integration might be clunky, or the selection of theories could be more appropriate.

3-5 points (Developing): Primarily relies on one theory, or attempts to use multiple theories but does so in a superficial or confused manner.

0-2 points (Unsatisfactory): Uses only one theory, or theories are haphazardly thrown in without clear purpose.
''',
            'content_type': 'grading_rubric',
            'source': 'Ethics Midterm Rubric',
            'course_id': 1,
            'assignment_id': 1
        },
        {
            'title': 'Ethics Midterm - Case 1 Data and Algorithm Predictions',
            'content': '''
SPECIFIC CASE EXPECTATIONS:

Case 1: Data and Algorithm Predictions Thought Experiment (Netflix, LinkedIn, Match) 

Question: Justify your "Yes" or "No" answers in terms of the ethical theories studied.

Key Theories Expected:

Rights Theory (Chapter 2 / Principles): Often the central discussion point. Students can argue both "Yes" (free to choose to enter the system, even if it limits future choice – a "kind of freedom") and "No" (loss of sequential freedom, fails to maximize freedom). The "Rights Theory argues against itself" perspective is a sophisticated approach.

Utilitarianism (Chapter 3 / Consequences): Can be used to argue "Yes" (taking the job/partner/entertainment makes me happy, so happiness is worth more than freedom). This sets utilitarianism against rights theory.

Fairness (Chapter 2 / Principles) / Kantian Dignity (Chapter 2 / Principles): Less directly applicable as core arguments, but might be brought in for supporting points.

Professor's Insights: Most students said "No". Discussion of whether one can freely choose not to be free, or if forced happiness is a kind of freedom. The interplay of rights theory with itself (yes vs. no) is a strong line of argument. Setting rights theory against utilitarianism ("happiness is worth more than freedom") is also a good approach.
''',
            'content_type': 'grading_rubric',
            'source': 'Ethics Midterm Rubric',
            'course_id': 1,
            'assignment_id': 1
        },
        {
            'title': 'Ethics Midterm - Case 2 Two at the Same Time',
            'content': '''
Case 2: Two at the Same Time (Real Estate Offers) 

Question: What would your advice be? Justify in ethical terms.

Key Theories Expected:

Fairness (Aristotle - Chapter 2 / Principles): This was the most common and strong answer. Focus on "equals should be treated equally and unequals unequally." Argue that buyers and sellers are fundamentally equals in the real estate market, and since sellers routinely make multiple offers (advertising on StreetEasy), buyers should be able to do the same.

Rights Theory (Chapter 2 / Principles): Can argue both "Yes" (buyer acting freely, not impinging on seller's freedom) and "No" (buyer's multiple offers "tying up" seller's time, thus limiting seller's freedom). Both directions can work depending on the justification.

Kantian Universalization (Chapter 2 / Principles): Analyze if a world where everyone makes multiple offers is conceivable without melting down. The professor notes it is conceivable, suggesting Kant might support it.

Kantian Dignity Argument (Chapter 2 / Principles): Argue that making multiple offers might treat sellers as mere "means to an end" (objects in a transaction) and not respect their dignity. The professor notes this is a stronger argument against multiple offers, but also points out that sellers do the same.

Professor's Insights: Fairness (Aristotle) was the strongest argument here, noting the equality of buyers and sellers and the reciprocal nature of multiple offers. Rights theory could go either way. Kantian universalization is potentially applicable but might be harder to apply. Dignity argument is a good approach against, but also has counter-points.
''',
            'content_type': 'grading_rubric',
            'source': 'Ethics Midterm Rubric',
            'course_id': 1,
            'assignment_id': 1
        },
        {
            'title': 'Ethics Midterm - Case 3 UFC',
            'content': '''
Case 3: UFC (Ultimate Fighting Championship) 

Question: Present an ethical argument for or against the senator's actions.

Key Theories Expected:

Rights Theory (Chapter 2 / Principles): A very strong argument against the senator. Focus on freedom: participants are not coerced, spectators are not forced. The senator is trying to limit others' freedom to pursue entertainment or a career. Also, the "perennial duties" argument about developing one's talents applies here (fighters have a talent they have a responsibility to develop).

Utilitarianism (Chapter 3 / Consequences): Can be argued against the senator (short-term happiness of many spectators and those making money outweighs Matua's suffering). Can also be argued for the senator (if it promotes violence in culture, leading to long-term unhappiness, a "Global Ethics" perspective could find the senator on the right side).

Monetized Utilitarianism (Chapter 3 / Consequences): Discuss how happiness/suffering can be measured by money (e.g., ticket sales vs. hospital bills). If money for happiness outweighs costs of suffering, it could support UFC.

Kantian Dignity (Chapter 2 / Principles): A strong argument for the senator's actions. If UFC treats fighters as "human form of cockfighting" or "tools/instruments for our happiness," it violates their dignity. Can be countered by the "dignity of developing talents" argument.

Professor's Insights: Freedom (Rights Theory) was a very strong argument against the senator. Perennial duties also support against the senator. Utilitarianism offers a rich discussion for both sides (short-term vs. long-term, Global Ethics). Monetized utilitarianism is a specific way to approach the utilitarian argument. Kantian dignity is a strong argument for the senator, but can be countered by the dignity of developing talents.
''',
            'content_type': 'grading_rubric',
            'source': 'Ethics Midterm Rubric',
            'course_id': 1,
            'assignment_id': 1
        }
    ]
    
    # Add each chunk to Pinecone
    successful_uploads = 0
    total_chunks = len(rubric_chunks)
    
    for i, chunk in enumerate(rubric_chunks, 1):
        print(f"📝 Processing chunk {i}/{total_chunks}: {chunk['title']}")
        
        # Generate embedding
        embedding = generate_embedding(chunk['content'], openai_client)
        if not embedding:
            print(f"   ❌ Failed to generate embedding for chunk {i}")
            continue
        
        # Prepare metadata
        metadata = {
            'title': chunk['title'],
            'content_type': chunk['content_type'],
            'source': chunk['source'],
            'course_id': chunk['course_id'],
            'assignment_id': chunk['assignment_id'],
            'text': chunk['content'][:1000]  # First 1000 chars for search
        }
        
        # Upload to Pinecone
        try:
            index.upsert(
                vectors=[{
                    'id': f"rubric_chunk_{i}_{int(time.time())}",
                    'values': embedding,
                    'metadata': metadata
                }],
                namespace='grading_rubric'
            )
            print(f"   ✅ Uploaded chunk {i} successfully")
            successful_uploads += 1
            time.sleep(0.1)  # Small delay to avoid rate limits
        except Exception as e:
            print(f"   ❌ Failed to upload chunk {i}: {e}")
    
    print(f"\n🎉 Grading rubric upload completed!")
    print(f"📊 Successfully uploaded: {successful_uploads}/{total_chunks} chunks")
    print("="*70)

if __name__ == "__main__":
    add_midterm_rubric() 
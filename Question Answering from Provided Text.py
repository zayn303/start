from sentence_transformers import SentenceTransformer
from transformers import pipeline

import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_collection")

# Список невеликих документів на різні тематики
documents = [
    "Meeting Minutes - Project Alpha\nDate: 2024-09-05\nAttendees: John Doe, Jane Smith, Michael Brown\nSummary: Discussed project timelines, assigned tasks for next sprint, and reviewed budget allocations.\nAction Items: John to finalize design by 09-10, Jane to review the financial plan, Michael to coordinate with the client on feature requests.",

    "Q3 Financial Report\nTotal Revenue: $2,500,000\nExpenses: $1,800,000\nNet Profit: $700,000\nKey insights: Revenue increased by 15% compared to Q2, largely driven by new client acquisitions. Operational expenses remained stable.",

    "Employee Feedback Survey\nResults: 85% of employees are satisfied with the current work-from-home policy. Key areas of improvement include internal communication and clearer career progression paths.",

    "Product Launch Plan\nProduct: X-200 Wireless Headphones\nLaunch Date: 2024-10-01\nMarketing Strategy: Digital campaign with influencer partnerships. Targeting tech enthusiasts aged 18-35.\nSales Goal: 50,000 units sold by the end of Q4.",

    "Annual IT Security Audit Report\nDate: 2024-08-20\nKey Findings: No major security breaches detected. Recommended updates to firewall and VPN protocols. All employees must complete cybersecurity training by 2024-09-15.",

    "Office Relocation Notice\nWe are pleased to inform you that our office is relocating to a new space on 2024-11-01. The new address is 1234 Corporate Avenue, Suite 5678. The relocation process will take place over the weekend to minimize business disruption.",

    "Client Proposal\nClient: GreenTech Innovations\nProject: Solar Panel Installation\nEstimated Cost: $500,000\nProject Duration: 6 months\nKey Deliverables: Site assessment, installation of 500 solar panels, and system optimization.",

    "HR Policy Update\nEffective Date: 2024-09-15\nChanges: The company will now offer extended parental leave of 16 weeks for all employees. Additionally, the annual leave policy has been updated to include 5 additional paid personal days.",

    "Quarterly Marketing Performance Review\nCampaign: Summer 2024\nBudget: $150,000\nTotal Impressions: 10 million\nConversions: 25,000\nReturn on Investment (ROI): 35%\nKey Takeaways: Video ads performed 40% better than banner ads, with higher engagement rates on social media platforms.",

    "Weekly Team Update\nWeek: 2024-09-01 to 2024-09-07\nProgress: The team completed 85% of the planned tasks. The new CRM system is fully integrated, and the first round of user testing was successful.\nChallenges: Minor delays in the mobile app update due to API issues.\nNext Steps: Finalize app updates by 2024-09-10 and prepare for product demo."
]


# Унікальні ідентифікатори для кожного документа
ids = [f"id{i+1}" for i in range(len(documents))]

# Додавання документів у колекцію
collection.add(
    documents=documents,
    ids=ids
)

# Запитання
question = "what is location adddres"

model = SentenceTransformer('all-MiniLM-L6-v2')
query_vector = model.encode(question)


retrieved_docs = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=1  # Кількість результатів
)

retrieved_docs

context = retrieved_docs['documents'][0][0]

# Модель для відповіді на питання
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
#qa_model = pipeline("question-answering", model="gpt2")
result = qa_model(question=question, context=context)

# Виведення відповіді
print('---***--- ', result['answer'], ' ---***--- ')
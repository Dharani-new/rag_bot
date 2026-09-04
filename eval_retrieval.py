"""
eval_retrieval.py
Retrieval evaluation: precision, recall, hit-rate@k
against manually-verified ground truth chunk IDs.
"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="retrieval_query"   # note: "query" not "document" here, since we're searching
)
vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)


def my_retrieve(question, k):
    results = vector_store.similarity_search(question, k=k)
    return [doc.metadata.get("chunk_id") for doc in results]
# ---- Ground truth: (question, list of correct chunk indices, 1-indexed) ----
# Empty list = out-of-scope question, scored separately (should retrieve NOTHING relevant / bot should say "not found")
GROUND_TRUTH = [
    ("What are the four moment terms included in the portfolio optimization objective?", [2, 12, 13, 38]),
    ("What is the formula used to determine the number of assets k to select based on the risk tolerance parameter r?", [37]),
    ("What method is used to convert higher-order terms in HUBO into a QUBO formulation?", [44, 45]),
    ("What three market regimes are identified by the Gaussian Hidden Markov Model?", [3, 11, 12]),
    ("What Sharpe ratio did the QAOA-based MVSK portfolio achieve under the moderate risk configuration?", [5, 63, 64]),
    ("How does detecting a bear market regime modify objective function weights compared to a bull regime?", [40]),
    ("Why does the paper use downside covariance instead of standard sample covariance?", [38]),
    ("How are continuous asset weights calculated after QAOA determines the binary selection vector x?", [49, 50]),
    ("What two classical baselines are used to benchmark the QAOA-based MVSK system?", [5, 63, 64]),
    ("Why are only fully diagonal tensor elements retained in the HUBO objective?", [42]),
    ("What real-time automated broker execution API is used to place trades in live markets?", []),
    ("How does the system dynamically rebalance portfolio weights across multiple periods?", []),
    ("What quantum random access memory architecture does the system use to load historical price data?", []),
    ("How does the QAOA solver account for asset transaction costs during optimization?", []),
    ("Why did the authors choose the QAOA solution with the best realized Sharpe ratio rather than the lowest raw QUBO energy?", [56, 57]),
]

K = 5  # top-k retrieved chunks to evaluate against


def precision_at_k(retrieved_ids, correct_ids):
    if not retrieved_ids:
        return 0.0
    hits = len(set(retrieved_ids) & set(correct_ids))
    return hits / len(retrieved_ids)


def recall_at_k(retrieved_ids, correct_ids):
    if not correct_ids:
        return None  # undefined for out-of-scope questions
    hits = len(set(retrieved_ids) & set(correct_ids))
    return hits / len(correct_ids)


def hit_rate_at_k(retrieved_ids, correct_ids):
    if not correct_ids:
        return None  # undefined for out-of-scope questions
    return 1.0 if set(retrieved_ids) & set(correct_ids) else 0.0


def run_eval(retrieve_fn):
    """
    retrieve_fn: a function(question: str, k: int) -> list[int]
                 must return the 1-indexed chunk IDs your retriever pulled back,
                 in rank order. Wire this to your actual FAISS + reranker pipeline.
    """
    results = []
    for question, correct_ids in GROUND_TRUTH:
        retrieved_ids = retrieve_fn(question, K)
        is_out_of_scope = len(correct_ids) == 0

        p = precision_at_k(retrieved_ids, correct_ids)
        r = recall_at_k(retrieved_ids, correct_ids)
        hr = hit_rate_at_k(retrieved_ids, correct_ids)

        results.append({
            "question": question,
            "retrieved": retrieved_ids,
            "correct": correct_ids,
            "precision": p,
            "recall": r,
            "hit_rate": hr,
            "out_of_scope": is_out_of_scope,
        })
    return results


def summarize(results):
    in_scope = [r for r in results if not r["out_of_scope"]]
    out_scope = [r for r in results if r["out_of_scope"]]

    print("=" * 60)
    print("IN-SCOPE QUESTIONS (retrieval quality)")
    print("=" * 60)
    for r in in_scope:
        print(f"\nQ: {r['question'][:70]}...")
        print(f"  Retrieved: {r['retrieved']}")
        print(f"  Correct:   {r['correct']}")
        print(f"  Precision: {r['precision']:.2f} | Recall: {r['recall']:.2f} | Hit: {r['hit_rate']:.0f}")

    avg_p = sum(r["precision"] for r in in_scope) / len(in_scope)
    avg_r = sum(r["recall"] for r in in_scope) / len(in_scope)
    avg_hr = sum(r["hit_rate"] for r in in_scope) / len(in_scope)

    print(f"\n--- AVERAGES (in-scope, n={len(in_scope)}) ---")
    print(f"Precision@{K}: {avg_p:.2f}")
    print(f"Recall@{K}:    {avg_r:.2f}")
    print(f"Hit-rate@{K}:  {avg_hr:.2f}")

    print("\n" + "=" * 60)
    print("OUT-OF-SCOPE QUESTIONS (should retrieve nothing relevant)")
    print("=" * 60)
    for r in out_scope:
        print(f"\nQ: {r['question'][:70]}...")
        print(f"  Retrieved: {r['retrieved']} (any result here should be low-confidence / bot should say 'not found')")


if __name__ == "__main__":
    results = run_eval(my_retrieve)
    summarize(results)


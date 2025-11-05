from flask import Flask, request, jsonify,Response
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app)
# =====================
# Algorithm Code Strings
# =====================
# server.py
algo_codes = {
    "KMeans": '''
import random, math

def kmeans(data, k=2, max_iter=100):
    centroids = random.sample(data, k)
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        # assign each point to nearest centroid
        for point in data:
            dists = [math.dist(point, c) for c in centroids]
            clusters[dists.index(min(dists))].append(point)
        # recompute centroids
        new_centroids = [
            [sum(x)/len(x) for x in zip(*cluster)] if cluster else c
            for cluster, c in zip(clusters, centroids)
        ]
        if new_centroids == centroids: break
        centroids = new_centroids
    return clusters, centroids

# --- Example usage ---
data = [[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]]
clusters, centroids = kmeans(data, k=2)
print("Clusters:", clusters)
print("Centroids:", centroids)

''',
    "NB": '''
import math
from collections import Counter

X = ["love python", "hate bugs", "love code", "hate errors"]
y = ["pos", "neg", "pos", "neg"]

# --- Training ---
classes = set(y)
priors = {c: y.count(c)/len(y) for c in classes}
word_count = {c: Counter() for c in classes}
total_words = {c: 0 for c in classes}
for text, c in zip(X, y):
    for w in text.split():
        word_count[c][w] += 1
        total_words[c] += 1
vocab = {w for c in word_count for w in word_count[c]}

print("Priors:", priors)
print("Word counts:", {c: dict(word_count[c]) for c in classes})

# --- Prediction ---
def predict(text):
    words = text.split()
    scores = {}
    V = len(vocab)
    for c in classes:
        logp = math.log(priors[c])
        for w in words:
            p = (word_count[c][w] + 1) / (total_words[c] + V)
            logp += math.log(p)
        scores[c] = logp
    print(f"\nScores for '{text}':", {c: round(scores[c], 4) for c in scores})
    print("Predicted:", max(scores, key=scores.get))

predict("love bugs")
predict("hate code")
''',
    "Apriori": '''
# ----- Apriori Algorithm (from scratch) -----
from itertools import combinations

def apriori(transactions, min_support=0.5):
    def support(itemset):
        return sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)

    C1 = {frozenset([item]) for t in transactions for item in t}
    L1 = {c for c in C1 if support(c) >= min_support}
    L = [L1]

    k = 2
    while L[-1]:
        candidates = set()
        prev = list(L[-1])
        for i in range(len(prev)):
            for j in range(i+1, len(prev)):
                union = prev[i].union(prev[j])
                if len(union) == k:
                    candidates.add(union)
        Lk = {c for c in candidates if support(c) >= min_support}
        if not Lk:
            break
        L.append(Lk)
        k += 1

    for level in L:
        for item in level:
            print(set(item), "Support:", support(item))

transactions = [
    {"milk", "bread", "butter"},
    {"beer", "bread"},
    {"milk", "bread", "beer", "butter"},
    {"bread", "butter"}
]
apriori(transactions, 0.5)
''',
    "PageRank": '''
def pagerank(graph, d=0.85, iterations=100):
    nodes = list(graph.keys())
    N = len(nodes)
    rank = {n: 1/N for n in nodes}  # initial rank

    for _ in range(iterations):
        new_rank = {}
        for n in nodes:
            # find inbound links (who points to n)
            inbound = [j for j in nodes if n in graph[j]]
            # apply formula: (1-d) + d * Σ(PR(j)/L(j))
            new_rank[n] = (1 - d) + d * sum(rank[j]/len(graph[j]) for j in inbound)
        rank = new_rank

    return rank

# --- Example graph ---
graph = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A"],
    "D": ["C"]
}

ranks = pagerank(graph)
for node, score in sorted(ranks.items()):
    print(f"{node}: {score:.4f}")

'''
}

@app.route('/', methods=['GET', 'HEAD'])
def health():
    # respond 200 for health checks (HEAD or GET)
    return ('OK', 200)

@app.route('/run', methods=['POST'])
def run_code():
    data = request.get_json(silent=True) or {}
    code_name = data.get('code_name')
    if not code_name:
        return jsonify({"error": "Missing 'code_name' in JSON payload"}), 400

    if code_name not in algo_codes:
        return jsonify({"error": f"Invalid code_name. Choose one of: {list(algo_codes.keys())}"}), 400

    code_text = algo_codes[code_name]
    print(f"\n🔹 Running {code_name} algorithm (printed below):\n")
    print(code_text)

    # Return raw code as plain text (for easy copy-paste)
    return Response(
        code_text,
        mimetype='text/plain',
        headers={"Content-Disposition": f"inline; filename={code_name}.py"}
    )
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # For local debugging you can set DEBUG=True; on Render use gunicorn
    app.run(host='0.0.0.0', port=port, debug=False)

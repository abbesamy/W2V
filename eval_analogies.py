import sys
import numpy as np
from scipy.spatial import KDTree

def load_embeddings(path):
    words, vectors = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().split()
        dim = int(header[1])
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) < dim + 1:
                continue
            words.append(parts[0])
            vectors.append([float(x) for x in parts[1:dim + 1]])
    return words, np.array(vectors, dtype=np.float32)

def main():
    emb_path, ana_path = sys.argv[1], sys.argv[2]
    
    print("Chargement et construction du KD-Tree...")
    words, vectors = load_embeddings(emb_path)
    word2idx = {w: i for i, w in enumerate(words)}
    tree = KDTree(vectors)

    results = {}
    current_cat = "Général"

    with open(ana_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(("#", "-")):
                current_cat = line.lstrip("#- :").strip()
                results.setdefault(current_cat, [0, 0])  # [réussies, total]
                continue

            a, b, c, d = line.split()
            if any(w not in word2idx for w in (a, b, c, d)):
                continue

            # Calcul vectoriel : b - a + c (ex: paris - france + espagne)
            target = vectors[word2idx[b]] - vectors[word2idx[a]] + vectors[word2idx[c]]

            # Recherche des 4 plus proches voisins pour pouvoir exclure a, b, c
            _, idxs = tree.query(target, k=4)
            pred = next(words[i] for i in idxs if words[i] not in (a, b, c))

            results[current_cat][1] += 1
            if pred == d:
                results[current_cat][0] += 1

    # Affichage des résultats par catégorie
    tot_ok, tot_all = 0, 0
    for cat, (ok, total) in results.items():
        if total > 0:
            print(f"{cat:<25} : {ok}/{total} ({ok / total * 100:.1f}%)")
            tot_ok += ok
            tot_all += total
    print(f"{'TOTAL GLOBAL':<25} : {tot_ok}/{tot_all} ({tot_ok / tot_all * 100:.1f}%)")

if __name__ == "__main__":
    main()
import argparse
import os
import sys
import numpy as np


def load_embeddings(path):
    """Lit un fichier d'embeddings
    Ligne 1 : "nb_mots dimension". Lignes suivantes : "mot v1 v2 ... vd".
    Retourne un dictionnaire mot -> vecteur numpy."""
    emb = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        if not header:
            return emb
        nb_words, dim = int(header[0]), int(header[1])
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) < dim + 1:
                continue
            word = parts[0]
            emb[word] = np.array(parts[1:dim + 1], dtype=np.float64)
    return emb


def cosine(a, b):
    """cos(a, b) = a.b / (||a|| * ||b||)"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_file(emb_path, sim_path):
    """Évalue un fichier d'embeddings sur le fichier de triplets et retourne le taux de réussite."""
    emb = load_embeddings(emb_path)
    evaluated = 0
    passed = 0

    with open(sim_path, "r", encoding="utf-8") as f:
        for line in f:
            words = line.split()
            if len(words) != 3:
                continue
            m, m_pos, m_neg = words
            if m not in emb or m_pos not in emb or m_neg not in emb:
                continue  # mot inconnu : triplet non évaluable
            evaluated += 1
            if cosine(emb[m], emb[m_pos]) > cosine(emb[m], emb[m_neg]):
                passed += 1

    if evaluated > 0:
        return passed / evaluated
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation d'embeddings sur tous les fichiers d'un répertoire"
    )
    parser.add_argument("-d", "--dir", default="embeddings",
                        help="Répertoire contenant les fichiers d'embeddings (défaut: embeddings)")
    parser.add_argument("-s", "--sim", required=True,
                        help="Fichier d'évaluation (un triplet par ligne)")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Erreur : le répertoire '{args.dir}' n'existe pas.", file=sys.stderr)
        sys.exit(1)

    # Récupération de tous les fichiers du dossier
    emb_files = [
        os.path.join(args.dir, f) 
        for f in os.listdir(args.dir) 
        if os.path.isfile(os.path.join(args.dir, f))
    ]

    if not emb_files:
        print(f"Aucun fichier trouvé dans le répertoire '{args.dir}'.", file=sys.stderr)
        sys.exit(1)

    rates = []
    print(f"--- Évaluation des fichiers du dossier '{args.dir}' ---")
    for path in sorted(emb_files):
        rate = evaluate_file(path, args.sim)
        rates.append(rate)
        print(f"{os.path.basename(path)} -> Taux de réussite : {rate:.3f}")

    # Calcul de la moyenne et de l'écart-type
    mean_rate = np.mean(rates)
    std_rate = np.std(rates)

    print("\n--- Résultats globaux ---")
    print(f"Nombre de fichiers évalués : {len(rates)}")
    print(f"Taux de réussite moyen : {mean_rate:.3f}")
    print(f"Écart-type : {std_rate:.3f}")


if __name__ == "__main__":
    main()
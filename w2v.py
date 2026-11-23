import numpy as np
import argparse
import sys
import os

# ==========================================
# Fonctions utilitaires
# ==========================================

# **** Calcul de la sigmoïde *******

def sigmoid(x):
    # on clip pour eviter les inf
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# **** Calcul de L (Loss) *******

def loss_positive(m, c_pos):
    # -log(sigma(m . c_pos))
    score = np.dot(m, c_pos)
    return -np.log(sigmoid(score) + 1e-15)


def loss_negative(m, c_neg):
    # -log(1 - sigma(m . c_neg))
    score = np.dot(m, c_neg)
    return -np.log(1 - sigmoid(score) + 1e-15)

def total_loss(m, c_pos, c_neg_list):
    # Perte positive
    loss = loss_positive(m, c_pos)
    # Ajout des pertes pour chaque exemple négatif
    for c_neg in c_neg_list:
        loss += loss_negative(m, c_neg)
    return loss


# **** Calcul des gradients *******

# Par rapport à Cpos
def grad_c_pos(m, c_pos):
    # Calcule le gradient de la perte par rapport au vecteur c_pos
    sig = sigmoid(np.dot(m, c_pos))
    return (sig - 1) * m


# Par rapport à Cneg
def grad_c_neg(m, c_neg):
    # Calcule le gradient de la perte par rapport au vecteur c_neg
    sig = sigmoid(np.dot(m, c_neg))
    return sig * m


# Par rapport au vecteur cible m : 
# Partie positive :
def grad_m_positive(m, c_pos):
    # Contribution du contexte positif au calcul de ce gradient
    sig = sigmoid(np.dot(m, c_pos))
    return (sig - 1) * c_pos

# Partie négative : 
def grad_m_negative(m, c_neg):
    # Contribution d'un contexte (exemple) négatif au calcul de ce gradient
    sig = sigmoid(np.dot(m, c_neg))
    return sig * c_neg

def grad_m_total(m, c_pos, c_neg_list):
    # Gradient apporté par le contexte positif
    g_m = grad_m_positive(m, c_pos)
    # + l'impact de chaque contexte négatif (k contextes)
    for c_neg in c_neg_list:
        g_m += grad_m_negative(m, c_neg)
    return g_m


def compute_unigram_probabilities(word_counts, alpha=0.75):
    # tableau des frequences de chaque mot
    counts = np.array(list(word_counts.values()), dtype=np.float64)
    # on applique (** alpha)
    weighted_counts = np.power(counts, alpha)
    probs = weighted_counts / np.sum(weighted_counts)
    return probs


# ==========================================
# Génération du fichier texte avec les mots
# ==========================================

def generate_training_file(corpus_lines, word_to_idx, idx_to_word, vocab_size, unigram_probs, L, k, filename):
    """Parcourt le corpus, applique la fenêtre glissante, tire les k exemples négatifs 
    et écrit chaque paire sous forme de mots dans un fichier texte intermédiaire.
    """
    print(f"--- Génération du fichier texte d'entraînement : {filename} ---")
    with open(filename, "w", encoding="utf-8") as f_out:
        for line in corpus_lines:
            tokens = [w for w in line if w in word_to_idx]
            line_len = len(tokens)

            # Parcours de chaque mot de la ligne comme mot cible
            for i in range(line_len):
                target_word = tokens[i]

                # Définition de la fenêtre glissante [i - L, i + L] bornée par les bords de la phrase
                start = max(0, i - L)
                end = min(line_len, i + L + 1)

                # Contexte positif : tous les mots de la fenêtre sauf le mot cible lui-même
                for j in range(start, end):
                    if j == i:
                        continue

                    c_pos_word = tokens[j]

                    # TIRAGE NÉGATIF (k exemples)
                    neg_indices = np.random.choice(
                        vocab_size, size=k, p=unigram_probs, replace=True
                    )
                    
                    # Conversion des indices négatifs en mots
                    neg_words = [idx_to_word[idx] for idx in neg_indices]

                    # Écriture dans le fichier : [mot_cible] [mot_positif] [mots_négatifs...]
                    line_data = [target_word, c_pos_word] + neg_words
                    f_out.write(" ".join(line_data) + "\n")
                    
    print("--- Fichier texte généré avec succès ---")


# ==========================================
# Fonction d'apprentissage depuis le fichier
# ==========================================

def train_from_file(training_file_path, M, C, word_to_idx, eta, epochs):
    """Lit le fichier texte ligne par ligne pour chaque époque, convertit les mots 
    en indices et applique la mise à jour des gradients (SGD).
    """
    print("--- Début de l'apprentissage ---")
    
    for epoch in range(epochs):
        with open(training_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                target_word = parts[0]
                c_pos_word = parts[1]
                neg_words = parts[2:]

                target_idx = word_to_idx[target_word]
                c_pos_idx = word_to_idx[c_pos_word]
                neg_indices = [word_to_idx[w] for w in neg_words]

                m = M[target_idx]
                c_pos = C[c_pos_idx]
                c_neg_list = [C[neg_idx] for neg_idx in neg_indices]

                # CALCUL DES GRADIENTS
                g_m = grad_m_total(m, c_pos, c_neg_list)
                g_c_pos = grad_c_pos(m, c_pos)
                g_c_neg_list = [grad_c_neg(m, C[neg_idx]) for neg_idx in neg_indices]

                # MISE À JOUR DES VECTEURS
                M[target_idx] -= eta * g_m
                C[c_pos_idx] -= eta * g_c_pos

                for neg_idx, g_c_neg in zip(neg_indices, g_c_neg_list):
                    C[neg_idx] -= eta * g_c_neg

        print(f"Époque {epoch + 1}/{epochs} terminée.")


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Entraînement Word2Vec (SGNS)")
    parser.add_argument("-i", "--input", required=True, help="Chemin vers le fichier texte segmenté")
    parser.add_argument("-o", "--output", required=True, help="Chemin du fichier de sortie pour les embeddings")
    parser.add_argument("-n", type=int, default=50, help="Dimension des embeddings")
    parser.add_argument("-L", type=int, default=2, help="Taille des contextes gauche et droit")
    parser.add_argument("-k", type=int, default=5, help="Nombre de contextes négatifs")
    parser.add_argument("-eta", type=float, default=0.025, help="Taux d'apprentissage")
    parser.add_argument("-e", type=int, default=5, help="Nombre d'itérations")
    parser.add_argument("-minc", type=int, default=5, help="Nombre minimal d'occurrences pour garder un mot")
    parser.add_argument("-ef", "--example_file", required=True, help="chemin vers le fichier intermédiaire d'exemples pos/neg")

    args = parser.parse_args()

    file_path = args.input
    output_path = args.output
    n = args.n
    L = args.L
    k = args.k
    eta = args.eta
    epochs = args.e
    minc = args.minc
    alpha = 0.75
    temp_pairs_file = os.path.join("train_pairs_sub", args.example_file)

    print(f"Paramètres chargés : n={n}, L={L}, k={k}, eta={eta}, e={epochs}, minc={minc}")
    print("--- Chargement et traitement du corpus ---")

    # Éléments indésirables à exclure (balises et ponctuation pure)
    ignored_tokens = {"<s>", "</s>", ",", ".", ";", ":", "!", "?", "«", "»", "—", "-", "'"}

    corpus_lines = []
    word_counts = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            # On filtre les balises et la ponctuation de chaque ligne
            clean_words = [w for w in words if w not in ignored_tokens]
            if clean_words:
                corpus_lines.append(clean_words)
                for word in clean_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Ajout du sous echantillonnage des mots fréquents
        total_words = sum(word_counts.values())
        t = 1e-2  # Ajusté pour petit corpus
        subsampled_lines = []
        for line in corpus_lines:
            new_line = []
            for w in line:
                f_w = word_counts[w] / total_words
                prob_keep = np.sqrt(t / f_w)
                if prob_keep >= 1.0 or np.random.rand() < prob_keep:
                    new_line.append(w)
            if new_line:
                subsampled_lines.append(new_line)
        corpus_lines = subsampled_lines

    word_to_idx = {}
    idx_to_word = {}
    filtered_word_counts = {}
    idx = 0

    for word, count in word_counts.items():
        if count >= minc:
            word_to_idx[word] = idx
            idx_to_word[idx] = word
            filtered_word_counts[word] = count
            idx += 1

    vocab_size = len(word_to_idx)
    print(f"Vocabulaire filtré : {vocab_size} mots uniques.")

    unigram_probs = compute_unigram_probabilities(filtered_word_counts, alpha=alpha)

    generate_training_file(
        corpus_lines, word_to_idx, idx_to_word, vocab_size, unigram_probs, L, k, temp_pairs_file
    )

    np.random.seed(42)
    M = np.random.uniform(-0.5 / n, 0.5 / n, (vocab_size, n))
    C = np.random.uniform(-0.5 / n, 0.5 / n, (vocab_size, n))

    train_from_file(temp_pairs_file, M, C, word_to_idx, eta, epochs)

    print(f"--- Sauvegarde des embeddings dans {output_path} ---")
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"{vocab_size} {n}\n")
        for i in range(vocab_size):
            word = idx_to_word[i]
            vector_str = " ".join(map(str, M[i]))
            out.write(f"{word} {vector_str}\n")

    print("--- Terminé avec succès ! ---")


if __name__ == "__main__":
    main()
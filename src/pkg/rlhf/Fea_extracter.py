import math
from collections import Counter
from pathlib import Path

import contractions
import numpy as np
import pandas as pd
import spacy
import torch
from grakel import Graph
from grakel.kernels import WeisfeilerLehman
from nltk.util import ngrams
from RL.setting import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker

##########################
# preprocess text
##########################

spell = SpellChecker()


def is_word(word: str):
    """Check if a word is a legitimate English word."""
    return len(spell.unknown([word])) == 0


##########################
# extract all features
##########################


class FeatureExtractor:
    def __init__(
        self,
        word_info=None,
        func_info=None,
        embedding_model=None,
        sem_fea=["sem_align", "sem_div", "sem_ent", "sem_repe"],
        doc_fea=[
            "tree_depth",
            "clause",
            "POS_count",
            "pp_den",
            "lex_den",
            "func_den",
            "dep_align",
            "dep_div",
            "lemma_sem",
            "lemma_div",
            "lemma_sem",
            "dep_repe",
            "lemma_repe",
            "func_den_new",
        ],
    ):
        """Initialize with word information DataFrame and embedding model (optional for coherence)."""
        nlp = spacy.load("en_core_web_trf")
        self.embedding_model = embedding_model
        self.wl_sim = WlTreeSim()
        self.lemma_vector = LemmaVec(func_info, nlp)
        self.syntax_analyzer = SyntaxAnalyzer()
        self.word_analyzer = WordStatAnalyzer(word_info)
        self.nlp = nlp  # Ensure spaCy model is loaded for NLP processing
        self.sem_fea = sem_fea
        self.doc_fea = doc_fea

    def get_turn_fea(
        self, target_sent=None, doc=None, lemma=None, features=None, content_POS=None, target_POS=None, column=None
    ):
        """Compute features from a single sentence

        Parameters
        ----------
        - target_sent: single sentence to analyze.
        - doc: parser related fea, if needed

        Returns
        -------
        - A DataFrame with feature names as columns and their values.
        - lemma vector

        """
        # select the subset based on the inputs

        # Initialize the DataFrame with NaN values;
        if pd.isna(target_sent) or len(str(target_sent)) == 0:
            # handle nan or empty string
            return pd.DataFrame([[np.nan] * len(features)], columns=features)

        # Initialize results dictionary, excluding 'POS_count' from initial results setup
        results = {feature: np.nan for feature in features if feature != "POS_count"}

        ########################
        # word stat
        ########################
        if any(
            f in features
            for f in [
                "freq",
                "conc",
                "word_len",
                "sent_len",
                "ttr",
                "non_word_type_rate",
                "non_word_rate",
                "distinct_2",
                "distinct_3",
            ]
        ):
            word_fea_dict = self.word_analyzer.get_mean_word_stat(target_sent)
            sel_fea_lst = list(set(features) & set(word_fea_dict.keys()))
            for fea in sel_fea_lst:
                results[fea] = word_fea_dict[fea]

        if "tree_depth" in features:
            results["tree_depth"] = self.syntax_analyzer.get_syntax_complexity(doc)

        ########################
        # share the clause_num
        ########################
        if any(f in features for f in ["clause", "pp_den"]):
            clause_count = self.syntax_analyzer.count_clause(doc)
            if "clause" in features:
                results["clause"] = clause_count
            if "pp_den" in features:
                results["pp_den"] = self.syntax_analyzer.get_pp_density(doc, clause_count)

        ########################
        # share the POS_lst
        ########################
        if any(f in features for f in ["POS_count", "lex_den"]):
            POS_lst = self.syntax_analyzer.get_POS(doc)

            if "POS_count" in features:
                POS_count_dict = self.syntax_analyzer.count_pos(POS_lst, target_POS)
                # seperate the results into different POS
                for POS_tag, POS_count in POS_count_dict.items():
                    results[POS_tag] = POS_count

            if "lex_den" in features:
                lex_den = self.syntax_analyzer.get_lex_den(POS_lst, content_POS)
                results["lex_den"] = lex_den
                if "func_den" in features and doc:
                    results["func_den"] = 1 - lex_den

        ########################
        # share lemma
        ########################
        if any(f in features for f in ["func_den_new", "lemma_div"]):
            if "func_den_new" in features:
                results["func_den_new"] = self.lemma_vector.get_func_prop(lemma, column)

        # Create a DataFrame from the results dictionary
        stat_df = pd.DataFrame([results])
        return stat_df

    def get_align_fea(self, features, lemmas: list, dep_graphs: list, word_vectors: list) -> pd.DataFrame:
        """Compute alignment features from input lists
        Alignment features compare adjacent turns by different interlocutors; repe by the same interlocutor

        Parameters
        ----------
        - features: List of features to compute
        - lemmas: List of lemmas
        - dep_graphs: List of dependency graphs
        - word_vectors: List of word vectors
        Returns:
        - A DataFrame with alignment features

        """
        # Initialize an empty dictionary to store alignment features
        align_features = {}

        ################################
        # semantic level alignment
        ################################
        if any(f in features for f in ["sem_align", "sem_repe"]):
            # initialize the class
            sem_div_extractor = PairwiseDiversity(word_vectors, "sem")
            if "sem_align" in features:
                align_features["sem_align"] = sem_div_extractor.compute_nth_sim(1)
            if "sem_repe" in features:
                print("Computing sem fea")
                align_features["sem_repe"] = sem_div_extractor.compute_nth_sim(2)

        ################################
        # lemma level alignment
        ################################
        if any(f in features for f in ["lemma_align", "lemma_repe"]):
            # initialize the class
            lemma_div_extractor = PairwiseDiversity(lemmas, "lemma")
            if "lemma_align" in features:
                align_features["lemma_align"] = lemma_div_extractor.compute_nth_sim(1)
            if "lemma_repe" in features:
                print("Computing lemma fea")
                align_features["lemma_repe"] = lemma_div_extractor.compute_nth_sim(2)

        ################################
        # syntactic level alignment
        ################################
        if any(f in features for f in ["dep_align", "dep_repe"]):
            # initialize the class
            dep_div_extractor = PairwiseDiversity(dep_graphs, "dep")
            if "dep_align" in features:
                align_features["dep_align"] = dep_div_extractor.compute_nth_sim(1)
            if "dep_repe" in features:
                print("Computing dep fea")
                align_features["dep_repe"] = dep_div_extractor.compute_nth_sim(2)
        return align_features

    def get_div_fea(self, features, indices: list, lemmas: list, dep_graphs: list, word_vectors: list) -> pd.DataFrame:
        """Compute features from a single sentence

        Parameters
        ----------
        - target_sent: single sentence to analyze.
        - doc: parser related fea, if needed

        Returns
        -------
        - A dict with the seelcted fea

        """
        div_df = {}
        ################################
        # semantic level align/diveristy
        ################################
        if "sem_div" in features:
            # Slice lists if indices are provided
            if indices is not None:
                word_vectors = [word_vectors[i] for i in indices]
            # initialize the class
            sem_div_extractor = PairwiseDiversity(word_vectors, "sem")
            sem_sim_matrix = sem_div_extractor.compute_pairwise_sim()
            div_df["sem_div"] = sem_div_extractor.compute_div_score(sem_sim_matrix)

        ################################
        # lemma level align/diveristy
        ################################
        if "lemma_div" in features:
            # Slice lists if indices are provided
            if indices is not None:
                lemmas = [lemmas[i] for i in indices]
            # initialize the class
            lemma_div_extractor = PairwiseDiversity(lemmas, "lemma")
            lemma_sim_matrix = lemma_div_extractor.compute_pairwise_sim()
            div_df["lemma_div"] = lemma_div_extractor.compute_div_score(lemma_sim_matrix)

        ################################
        # syntactic level align/diveristy
        ################################
        if "dep_div" in features:
            # Slice lists if indices are provided
            if indices is not None:
                dep_graphs = [dep_graphs[i] for i in indices]
            # initialize the class
            dep_div_extractor = PairwiseDiversity(dep_graphs, "dep")
            dep_sim_matrix = dep_div_extractor.compute_pairwise_sim()
            div_df["dep_div"] = dep_div_extractor.compute_div_score(dep_sim_matrix)

        return div_df

    @staticmethod
    def _convert_to_numpy(item):
        """Helper function to convert an embedding to a NumPy array.
        :param item: A PyTorch tensor, NumPy array, or other supported types.
        :return: A NumPy array or np.nan for unsupported types.
        """
        if isinstance(item, torch.Tensor):
            return item.cpu().numpy()  # Move to CPU and convert to NumPy
        if isinstance(item, np.ndarray):
            return item
        if isinstance(item, float):
            return np.nan  # Return NaN for float types
        raise ValueError(f"Unsupported type {type(item)} for embeddings.")

    # TODO add entropy score
    def extract_vec(self, sentences: list, features: list) -> list:
        """Extract word embeddings from a list of sentences

        Parameters
        ----------
        - sentences: a list of texts
        - features: a list of features

        """
        word_vectors = []
        if any(f in features for f in self.sem_fea):
            for sent in sentences:
                if pd.isna(sent) or len(str(sent)) == 0:
                    word_vectors.append(np.nan)
                else:
                    vector = self.embedding_model.encode(sent, convert_to_tensor=True)
                    word_vectors.append(vector)

        embeddings = [
            self._convert_to_numpy(item) if item is not None and not isinstance(item, str) else np.nan
            for item in word_vectors
        ]

        return embeddings

    def extract_doc(self, sentences: list, features: list) -> list:
        """Compute docs from a list of sentences; to be used for sentence vector and lemma vector

        Parameters
        ----------
        - sentences: a list of texts
        - features: a list of features

        """
        docs = []
        lemmas = []
        dep_graphs = []

        if any(f in features for f in self.doc_fea):
            for sent in sentences:
                if pd.isna(sent) or len(str(sent)) == 0:
                    docs.append(np.nan)
                    lemmas.append(np.nan)
                    dep_graphs.append(np.nan)
                else:
                    doc = self.nlp(sent)
                    docs.append(doc)
                    if any(f in features for f in ["func_den_new", "lemma_div", "lemma_align", "lemma_repe"]):
                        lemma = self.lemma_vector.lemmatize_text(doc)
                        lemmas.append(lemma)
                    if any(f in features for f in ["dep_align", "dep_div", "dep_repe"]):
                        dep_graph = self.wl_sim.dependency_tree_to_graph(doc)
                        dep_graphs.append(dep_graph)
        return docs, lemmas, dep_graphs


##################################
# extract word-level features
##################################


class WordStatAnalyzer:
    def __init__(self, word_info):
        """Initialize the analyzer with a word information DataFrame.
        word_info should contain 'word', 'freq_m', and 'conc' columns.
        """
        self.word_info = word_info

    def get_mean_word_stat(self, sentence: str):
        """Calculate mean frequency, concreteness, word length, type-token ratio (TTR),
        non-word type rate, non-word rate, and distinct-2/3 metrics for a sentence.
        Returns a dictionary with metric names as keys and their values as values.
        """
        if len(sentence) > 0:
            # preprocess data
            sentence = preprocess(sentence, remove_punc=True)
            words = sentence.split()
            sent_len = len(words)

            freq_scores, conc_scores, length_scores = [], [], []
            true_words = []
            non_words = []

            for word in words:
                freq, conc, length = self.get_word_stat(word)
                length_scores.append(length)

                # Only include words with valid frequency and concreteness values
                if not np.isnan(freq) and not np.isnan(conc):
                    freq_scores.append(freq)
                    conc_scores.append(conc)

                # Categorize word as a true word or non-word
                if is_word(word):
                    true_words.append(word)
                else:
                    non_words.append(word)

            # Calculate metrics
            mean_freq = np.nanmean(freq_scores) if freq_scores else np.nan
            mean_conc = np.nanmean(conc_scores) if conc_scores else np.nan
            mean_word_len = np.mean(length_scores) if length_scores else np.nan
            ttr = self.get_ttr(true_words)
            non_word_type_rate = self.get_non_word_type_rate(sentence, non_words)
            non_word_rate = self.get_non_word_rate(sentence, non_words)
            distinct_2 = self.get_distinct_n(words, n=2)
            distinct_3 = self.get_distinct_n(words, n=3)

            # Return results as a dictionary
            return {
                "freq": mean_freq,
                "conc": mean_conc,
                "word_len": mean_word_len,
                "sent_len": sent_len,
                "ttr": ttr,
                "non_word_type_rate": non_word_type_rate,
                "non_word_rate": non_word_rate,
                "distinct_2": distinct_2,
                "distinct_3": distinct_3,
            }
        # Return NaN for all metrics if the sentence is empty
        return {
            "freq": np.nan,
            "conc": np.nan,
            "word_len": np.nan,
            "sent_len": np.nan,
            "ttr": np.nan,
            "non_word_type_rate": np.nan,
            "non_word_rate": np.nan,
            "distinct_2": np.nan,
            "distinct_3": np.nan,
        }

    def replace_single_letter(self, input_string: str):
        """Replace single-letter words in the string (except 'a' and 'i') with 'nword'."""
        words = input_string.split()
        replaced_words = [f"nword_{word}" if len(word) == 1 and word not in {"a", "i"} else word for word in words]
        return replaced_words

    def get_word_stat(self, word: str):
        """Given a cleaned lower-case word, return its frequency, concreteness, and length.
        If the word is not in the word_info DataFrame, checks if it is a legitimate word.
        """
        sel_row = self.word_info[self.word_info["word"] == word]
        if sel_row.shape[0] > 0:
            freq = sel_row["freq_m"].item() / 1000  # Convert to freq_m per-thousand
            conc = sel_row["conc"].item()
        elif is_word(word):
            freq, conc = 0, 0
        else:
            freq, conc = np.nan, np.nan
        return freq, conc, len(word) if word else 0

    def get_ttr(self, true_words: list):
        """Calculate the type-token ratio (TTR) based only on true words in the input list."""
        # Filter the list to include only true words
        if len(true_words) == 0:
            return np.nan
        unique_words = set(true_words)
        return len(unique_words) / len(true_words)

    def get_non_word_type_rate(self, sentence: str, non_words: list):
        """Calculate the non-word type rate (unique non-words / total tokens)."""
        try:
            unique_non_words = set(non_words)
            return len(unique_non_words) / len(set(sentence.split()))
        except:
            return np.nan

    def get_non_word_rate(self, sentence: str, non_words: list):
        """Calculate the non-word rate (total non-words / total tokens)."""
        try:
            return len(non_words) / len(sentence.split())
        except:
            return np.nan

    def get_distinct_n(self, words: list, n: int):
        """Calculate the Distinct-n metric: the ratio of unique n-grams to total n-grams."""
        n_grams = self.extract_ngrams(words, n)
        if not n_grams:
            return np.nan
        unique_n_grams = set(n_grams)
        return len(unique_n_grams) / len(n_grams)

    def extract_ngrams(self, words: list, n: int):
        """Generate n-grams from a list of words."""
        n_grams = list(ngrams(words, n))
        # Convert tuple into a string
        return [" ".join(map(str, t)) for t in n_grams]


##################################
# extract sentence-level features
##################################


class SyntaxAnalyzer:
    def __init__(self):
        # Load the spaCy model for POS tagging and parsing
        pass

    def replace_non_words(self, sentence: str) -> str:
        """Replace non-standard words with a special token."""
        words = sentence.split()
        replaced_words = [word if is_word(word) else "placeholder" for word in words]
        return " ".join(replaced_words)

    def get_POS(self, doc) -> list:
        POS_lst = []
        try:
            for token in doc:
                POS_lst.append(token.pos_)
        except Exception:
            POS_lst = []
        return POS_lst

    def count_pos(self, pos_list, target_tags) -> dict:
        """Count occurrences of each target POS tag in the list of POS tags.

        Parameters
        ----------
        - pos_list: List of POS tags from the document.
        - target_tags: List of POS tags to count.

        Returns
        -------
        - Dictionary with each target tag as the key and its count as the value.

        """
        pos_counts = Counter(pos_list)
        return {tag: pos_counts.get(tag, 0) for tag in target_tags}

    def count_clause(self, doc):
        """Count the number of clauses in a parsed sentence using dependency labels."""
        try:
            clause_count = 0
            for token in doc:
                # Key clause-related dependencies
                if token.dep_ in ["ROOT", "mark", "relcl", "ccomp", "advcl", "xcomp"] or (
                    token.dep_ == "conj" and token.head.dep_ == "ROOT"
                ):
                    if token.pos_ in ["VERB", "AUX"]:  # Ensure it's verbal
                        clause_count += 1

        except Exception as e:
            print(f"Error in count_clause: {e}")
            clause_count = np.nan

        return clause_count

    def get_syntax_complexity(self, doc):
        try:
            # root = [token for token in doc if token.head == token][0]
            def get_depth(token):
                depth = 0
                while token.head != token:
                    token = token.head
                    depth += 1
                return depth

            depths = [get_depth(token) for token in doc]
            mean_depth = sum(depths) / len(depths)
        except Exception as e:
            print(f"Error in get_syntax_complexity: {e}")
            mean_depth = np.nan
        return mean_depth

    def count_pp(self, doc):
        """Count the number of prepositional phrases in the document."""
        try:
            pp_count = sum(1 for token in doc if token.dep_ == "prep")
        except Exception as e:
            print(f"Error in count_prepositional_phrases: {e}")
            pp_count = np.nan
        return pp_count

    def get_pp_density(self, doc, clause_count):
        """Calculate Prepositional Phrase Density (PP Density):
        Number of prepositional phrases per sentence or clause.
        """
        try:
            pp_density = self.count_pp(doc) / max(clause_count, 1)  # Avoid division by zero
        except:
            pp_density = np.nan
        return pp_density

    def get_lex_den(self, pos_list: list, target_tags: list):
        """Calculate lexical density based on two lists A and B."""
        # remove the punctuation and special tags
        nonword_lst = ["PUNCT", "SYM", "X", "SPACE"]
        pos_list = [item for item in pos_list if item not in nonword_lst]

        if len(pos_list) > 0:
            # Count the occurrences of each element in both lists
            counter_A = Counter(pos_list)
            counter_B = Counter(target_tags)

            # Calculate the number of matching elements considering their counts
            matching_elements = 0
            for element in counter_A:
                matching_elements += min(counter_A[element], counter_B.get(element, 0))
            # Calculate the proportion
            proportion = matching_elements / len(pos_list)

        else:
            proportion = np.nan

        return proportion


############################################
# construct vectors & compute similarity
############################################


class LemmaVec:
    def __init__(self, func_info, nlp=None):
        self.func_info = func_info
        self.nlp = nlp

    def lemmatize_text(self, doc):
        """Tokenize and lemmatize the text using spaCy, handling contractions,
        and ignoring punctuation and spaces.
        """
        # Expand contractions in the document text
        expanded_text = contractions.fix(doc.text)
        # Create a new Doc object from the expanded text
        expanded_doc = self.nlp(expanded_text)
        # Lemmatize the text
        return [token.lemma_.lower() for token in expanded_doc if not token.is_punct and not token.is_space]

    def get_lemma_count_vector(self, lemmas):
        """Get the count of each lemma in a list of lemmas."""
        return Counter(lemmas)

    def create_vector(self, counts, all_lemmas):
        """Create a vector from lemma counts based on the set of all unique lemmas."""
        return np.array([counts.get(lemma, 0) for lemma in all_lemmas])

    def compute_cosine_simi(self, utterance_lemmas, response_lemmas):
        """Compute lexical alignment between an utterance and a response."""
        try:
            # Get lemma count vectors
            utterance_counts = self.get_lemma_count_vector(utterance_lemmas)
            response_counts = self.get_lemma_count_vector(response_lemmas)

            # Create a set of all unique lemmas across both utterance and response
            all_lemmas = set(utterance_lemmas).union(set(response_lemmas))

            # Create vectors for the utterance and response based on lemma counts
            # reshape to ensure the vectors are 2D as required by Scikit-learn
            utterance_vector = self.create_vector(utterance_counts, all_lemmas).reshape(1, -1)
            response_vector = self.create_vector(response_counts, all_lemmas).reshape(1, -1)

            # Calculate and return cosine similarity
            return cosine_similarity(utterance_vector, response_vector)[0][0]

        except Exception as e:
            # Log the error and return NaN
            print(f"Error in lexical_alignment: {e}")
            return np.nan

    def get_func_prop(self, lemmas: list, column: str) -> float:
        """Get function word prop from the given list
        words: a list of lemmas
        """
        if len(lemmas) == 0:
            proportion = np.nan
        else:
            column_set = set(self.func_info[column])  # Convert the column to a set for faster lookup
            matching_words = [word for word in lemmas if word in column_set]  # Find matches
            proportion = len(matching_words) / len(lemmas) if lemmas else 0  # Compute proportion
        return proportion


class WlTreeSim:
    # TODO: compare the results between normalized and unnormalized
    """Compute dependency tree similarity using advanced graph representation
    and Weisfeiler-Lehman graph kernel.
    """

    def __init__(self, n_iter=5, normalize=False):
        """Initialize the tree similarity calculator.

        :param n_iter: Number of Weisfeiler-Lehman iterations
        :param normalize: Whether to normalize the kernel
        """
        self.wl_kernel = WeisfeilerLehman(n_iter=n_iter, normalize=normalize)

    def dependency_tree_to_graph(self, doc):
        """Converts a SpaCy dependency tree to a graph with rich node features.

        :param doc: A SpaCy-parsed document
        :return: A Grakel Graph object or np.nan if an error occurs
        """
        # Early exit for invalid input
        if doc is None or (isinstance(doc, float) and np.isnan(doc)):
            return np.nan

        try:
            # Prepare more comprehensive node representation
            n = len(doc)
            adj_matrix = [[0] * n for _ in range(n)]
            node_labels = {}

            for i, token in enumerate(doc):
                # Create a comprehensive node label
                node_labels[i] = self._create_node_label(token)

                # Bidirectional edges to capture full dependency structure
                if token.head.i < n:
                    adj_matrix[i][token.head.i] = 1
                    adj_matrix[token.head.i][i] = 1

            # Create graph with enhanced representation
            graph = Graph(initialization_object=adj_matrix, node_labels=node_labels)

            return graph

        except Exception as e:
            print(f"Error converting dependency tree to graph: {e}")
            return np.nan

    def _create_node_label(self, token):
        """Create a rich, informative node label.

        :param token: SpaCy token
        :return: Comprehensive string representation of the token
        """
        # Combine multiple token attributes for richer representation
        label_parts = [
            token.pos_,  # Part of speech
            token.dep_,  # Dependency relation
            token.text.lower(),  # Lowercase text
            str(token.is_stop),  # Is stop word
            str(token.is_punct),  # Is punctuation
        ]
        return "_".join(label_parts)

    def compute_graph_similarity(self, graph1, graph2):
        """Computes the similarity between two dependency tree graphs.

        :param graph1: First graph representation
        :param graph2: Second graph representation
        :return: Similarity score or np.nan if invalid inputs
        """
        # Validate inputs
        if (graph1 is None or (isinstance(graph1, float) and np.isnan(graph1))) or (
            graph2 is None or (isinstance(graph2, float) and np.isnan(graph2))
        ):
            return np.nan
        try:
            # Compute similarity using Weisfeiler-Lehman kernel
            similarity_matrix = self.wl_kernel.fit_transform([graph1, graph2])
            # add post-normalization
            similarity = similarity_matrix[0, 1]
            self_similarity_1 = similarity_matrix[0, 0]
            self_similarity_2 = similarity_matrix[1, 1]
            normalized_similarity = similarity / np.sqrt(self_similarity_1 * self_similarity_2)
            return normalized_similarity
        except Exception as e:
            print(f"Error computing graph similarity: {e}")
            return np.nan


##################################
# get diversity/similarity scores
##################################
class PairwiseDiversity:
    # TODO: only compute the similarity pairs once
    def __init__(self, items, method=None, func_info=None):
        """Initialize with a list of items (embeddings or SpaCy-parsed documents).
        :param items: List of word embeddings (numpy arrays, PyTorch tensors, or other supported types) or parsed documents.
        :param method: The similarity method to use: "cosine" or "tree".
        """
        # initialize the class
        self.nlp = spacy.load("en_core_web_trf")
        self.method = method
        self.items = items

        if self.method == "dep":
            self.tree_sim = WlTreeSim()
        elif self.method == "lemma":
            self.lemma_vector = LemmaVec(func_info)

    @staticmethod
    def _convert_to_numpy(item):
        """Helper function to convert an embedding to a NumPy array.
        :param item: A PyTorch tensor, NumPy array, or other supported types.
        :return: A NumPy array or np.nan for unsupported types.
        """
        if isinstance(item, torch.Tensor):
            return item.cpu().numpy()  # Move to CPU and convert to NumPy
        if isinstance(item, np.ndarray):
            return item
        if isinstance(item, float):
            return np.nan  # Return NaN for float types
        raise ValueError(f"Unsupported type {type(item)} for embeddings.")

    def compute_pairwise_sim(self):
        """Computes the pairwise similarities based on the selected method.
        :return: A matrix of pairwise similarities, with NaN values for invalid comparisons.
        """
        n = len(self.items)
        similarity_matrix = np.full((n, n), np.nan)  # Initialize similarity matrix with NaN

        # Step 1: Preprocess and filter embeddings
        valid_indices = []
        for i in range(n):
            # Use proper checking for arrays and potential NaN values
            item = self.items[i]
            if (isinstance(item, (np.ndarray, list)) and len(item) > 0) or (pd.notna(item) and item is not None):
                valid_indices.append(i)

        if len(valid_indices) == 0:
            return similarity_matrix  # Return the NaN matrix if no valid sentences

        # Compute pairwise similarities for semantic similarities
        if self.method == "sem":
            # Compute cosine similarity
            valid_embeddings = [self.items[i] for i in valid_indices if not np.isnan(self.items[i]).any()]
            if valid_embeddings:
                cosine_matrix = cosine_similarity(valid_embeddings)
                for i, row_index in enumerate(valid_indices):
                    for j, col_index in enumerate(valid_indices):
                        similarity_matrix[row_index, col_index] = cosine_matrix[i, j]

        # Compute pairwise similarity for the list of dependency graphs
        elif self.method in ["dep", "lemma"]:
            for i in range(len(valid_indices)):
                for j in range(i, len(valid_indices)):  # Include diagonal
                    row_index = valid_indices[i]
                    col_index = valid_indices[j]
                    if self.method == "dep":
                        # Compute tree similarity
                        sim_score = self.tree_sim.compute_graph_similarity(self.items[row_index], self.items[col_index])

                    elif self.method == "lemma":  # compute the lemma simialrity
                        # lemma is looped if we use compute lemma similarity
                        sim_score = self.lemma_vector.compute_cosine_simi(self.items[row_index], self.items[col_index])
                    # Symmetry
                    similarity_matrix[row_index, col_index] = sim_score
                    similarity_matrix[col_index, row_index] = sim_score

        return similarity_matrix

    def compute_div_score(self, similarity_matrix):
        """Computes the diversity score as the inverse of the average pairwise similarity.

        :return: Diversity score (lower is more diverse).
        """
        # Mask diagonal to exclude self-similarity
        np.fill_diagonal(similarity_matrix, np.nan)
        # Compute the average similarity of all valid pairs
        valid_similarities = similarity_matrix[~np.isnan(similarity_matrix)]
        if len(valid_similarities) == 0:
            return np.nan

        avg_similarity = np.mean(valid_similarities)
        diversity_score = 1 - avg_similarity  # 1 means maximum diversity, 0 means no diversity
        return diversity_score

    def compute_adjacent_sim(self):
        """Computes pairwise similarity between adjacent items in the list.

        :return: A list of similarity scores, with NaN at the first index.
        """
        similarity_scores = [np.nan]  # First element has no previous comparison

        for i in range(1, len(self.items)):
            # Check if both current and previous items are valid
            if self.items[i] is None or self.items[i - 1] is None:
                similarity_scores.append(np.nan)
                continue

            # Compute similarity based on the method
            if self.method == "sem":
                # Use semantic embeddings
                # Check if any element in the embedding is NaN
                if np.isnan(self.items[i]).any() or np.isnan(self.items[i - 1]).any():
                    similarity_scores.append(np.nan)
                else:
                    similarity = cosine_similarity([self.items[i - 1]], [self.items[i]])[0][0]
                    similarity_scores.append(similarity)

            elif self.method == "dep":
                # Use dependency tree similarity
                similarity = self.tree_sim.compute_graph_similarity(self.items[i - 1], self.items[i])
                similarity_scores.append(similarity)

            elif self.method == "lemma":
                # Use lemma vector similarity
                similarity = self.lemma_vector.compute_cosine_simi(self.items[i - 1], self.items[i])
                similarity_scores.append(similarity)

            else:
                # Unsupported method
                similarity_scores.append(np.nan)

        return similarity_scores

    def compute_nth_sim(self, n):
        """Computes pairwise similarity between an element and the previous nth element in the list.

        :param n: Number of elements to look back for computing similarity.
        :return: A list of similarity scores, with NaN for elements where the nth previous element does not exist.
        """
        similarity_scores = [np.nan] * n  # First n elements have no prior nth element

        for i in range(n, len(self.items)):
            # Check if both current and nth previous items are valid
            if self.items[i] is None or self.items[i - n] is None:
                similarity_scores.append(np.nan)
                continue

            # Compute similarity based on the method
            if self.method == "sem":
                # Use semantic embeddings
                if np.isnan(self.items[i]).any() or np.isnan(self.items[i - n]).any():
                    similarity_scores.append(np.nan)
                else:
                    similarity = cosine_similarity([self.items[i - n]], [self.items[i]])[0][0]
                    similarity_scores.append(similarity)

            elif self.method == "dep":
                # Use dependency tree similarity
                similarity = self.tree_sim.compute_graph_similarity(self.items[i - n], self.items[i])
                similarity_scores.append(similarity)

            elif self.method == "lemma":
                # Use lemma vector similarity
                similarity = self.lemma_vector.compute_cosine_simi(self.items[i - n], self.items[i])
                similarity_scores.append(similarity)

            else:
                # Unsupported method
                similarity_scores.append(np.nan)

        return similarity_scores


##########################################
# Data preprocessing of the input file
##########################################
class DataMerger:
    def __init__(self, base_dir: Path, role_dict: dict):
        """Initialize the DataMerger class with the base directory and role dictionary.

        :param base_dir: The base directory where CSV files are located.
        :param role_dict: A dictionary mapping speaker roles.
        """
        self.base_dir = base_dir
        self.role_dict = role_dict

    @staticmethod
    def merge_columns(
        df: pd.DataFrame,
        headers_to_merge: list[str],
        speakers: list[str],
        columns_to_preserve: list[str],
        new_header: str,
    ) -> pd.DataFrame:
        """Merge specified columns from a DataFrame into a single column.

        :param df: Input DataFrame.
        :param headers_to_merge: List of column names to merge.
        :param speakers: List of speaker names corresponding to the columns.
        :param columns_to_preserve: List of columns to preserve in the new DataFrame.
        :param new_header: Name of the new merged column.
        :return: A new DataFrame with merged columns.
        """
        merged = []
        duplicated_speakers = []
        preserved_data = {col: [] for col in columns_to_preserve}

        for _, row in df.iterrows():
            for header, speaker in zip(headers_to_merge, speakers, strict=False):
                merged.append(row[header])
                duplicated_speakers.append(speaker)
                for col in columns_to_preserve:
                    preserved_data[col].append(row[col])

        merged_df = pd.DataFrame(preserved_data)
        merged_df["speaker"] = duplicated_speakers
        merged_df[new_header] = merged
        return merged_df

    def concat_single_turn(self, model_lst: list[str], speaker_lst: list[str], history_num: int) -> pd.DataFrame:
        """Concatenate single-turn generations from multiple speakers and models.

        :param model_lst: List of model names.
        :param speaker_lst: List of speakers.
        :param dialogue_num: Dialogue number to load.
        :return: A DataFrame with concatenated single-turn data.
        """
        data_single = pd.DataFrame()
        for speaker in speaker_lst:
            data = pd.read_csv(f"{self.base_dir}/single_turn/gen_few/{speaker}_10.csv")

            """
                if speaker == 'ADULT':
                    model_lst.append(speaker)
            """
            for model in model_lst:
                data_merged = self.merge_columns(
                    data,
                    headers_to_merge=[self.role_dict[speaker], model],
                    speakers=[self.role_dict[speaker], speaker],
                    columns_to_preserve=["month", "path"],
                    new_header="content",
                )
                data_merged.insert(0, "model", model if model != "ADULT" else "CHILDES")
                data_merged.insert(0, "test_role", speaker if model != "ADULT" else "CHILDES")
                data_single = pd.concat([data_single, data_merged])
        data_single.insert(0, "gen_type", "single")
        data_single.insert(0, "history", history_num)
        return data_single

    def concat_multi_turn(self, history_num: int, context_len: str) -> pd.DataFrame:
        """Concatenate multi-turn generations for multiple models.

        :param model_lst: List of model names.
        :param dialogue_num: Dialogue number to load.
        :param context_len: Context length (default is 'all').
        :return: A DataFrame with concatenated multi-turn data.
        """
        filename = f"CHI_{context_len}_ADULT_{context_len}_10.csv"

        data_multi = pd.DataFrame()
        for file in Path(self.base_dir).iterdir():
            if file.is_dir():  # Check if it's a subdirectory
                # loop over folders to load file
                data_all = pd.read_csv(file / filename)
                data = data_all[["month", "path", "speaker", data_all.columns[-1]]]
                data.rename(columns={data_all.columns[-1]: "content"}, inplace=True)
                data.insert(0, "model", data_all.columns[-1])
                data.insert(0, "test_role", "ADULT")
                data_multi = pd.concat([data, data_multi], ignore_index=True)

        data_multi.insert(0, "gen_type", "all")
        data_multi.insert(0, "history", history_num)
        return data_multi


##########################
# Speech act annotation
##########################


class SemEnt:
    def __init__(self, embeddings: list, indices=None, random_state=42, max_k=10):
        """Initialize the semantic diversity calculator with precomputed embeddings.

        Parameters
        ----------
            embeddings (list): List of precomputed sentence embeddings.
            random_state (int): Seed for reproducibility in clustering algorithms.

        """
        self.random_state = random_state  # Set random state for reproducibility
        self.max_k = max_k
        if indices is not None:
            embeddings = [embeddings[i] for i in indices]
            self.embeddings = [x for x in embeddings if not (isinstance(x, float) and math.isnan(x))]
        else:
            self.embeddings = embeddings

    def _optimal_k(self):
        """Determine the optimal number of clusters (k) using the silhouette score.

        Parameters
        ----------
            embeddings (np.ndarray): Semantic embeddings for clustering.
            max_k (int): Maximum number of clusters to consider.
            random_state (int): Seed for reproducibility.

        """
        best_k = 2
        best_score = -1
        for k in range(2, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(self.embeddings)
            score = silhouette_score(self.embeddings, labels)
            if score > best_score:
                best_k = k
                best_score = score
        return best_k

    def _compute_semantic_distribution(self, kmeans):
        """Compute the semantic distribution for the given embeddings and K-means model."""
        cluster_indices = kmeans.predict(self.embeddings)
        cluster_counts = np.bincount(cluster_indices, minlength=kmeans.n_clusters)
        return cluster_counts / len(cluster_indices)

    def _compute_sem_entropy(self, distribution):
        """Compute the semantic entropy for a given distribution."""
        non_zero_probs = distribution[distribution > 0]
        return -np.sum(non_zero_probs * np.log(non_zero_probs))

    def compute_diversity(self):
        """Compute semantic diversity for each dialogue and return the results.

        Parameters
        ----------
            df (pd.DataFrame): DataFrame with two columns: ['dialogue_id', 'dialogue_content'].
            max_k (int): Maximum number of clusters to consider for k-means.
            random_state (int): Seed for reproducibility.

        Returns
        -------
            the original dialogue with sem_ent and k as 2 additional columns

        """
        # Validate that embeddings are provided
        if len(self.embeddings) == 0:
            sem_score = [np.nan, np.nan]
        else:
            try:
                # Find the optimal number of clusters
                k = self._optimal_k()
                # Perform clustering
                kmeans = KMeans(n_clusters=k, random_state=self.random_state).fit(self.embeddings)
                # Compute semantic distribution and entropy
                distribution = self._compute_semantic_distribution(kmeans)
                sem_entropy = self._compute_sem_entropy(distribution)
            except Exception as e:
                print(f"Error processing: {e}")
                sem_entropy = 0
                k = 0
            sem_score = [sem_entropy, k]

        return sem_score


##########################
# Speech act annotation
##########################
'''
class Annotator:
    def __init__(
        self,
        model_path,
        use_bi_grams=False,
        use_pos=False,
        use_past=False,
        use_repetitions=False,
        compare_frequencies_path=None,
    ):
        """
        Initialize the Annotator with model details and feature usage options.
        """
        self.model_path = model_path
        self.use_bi_grams = use_bi_grams
        self.use_pos = use_pos
        self.use_past = use_past
        self.use_repetitions = use_repetitions
        self.compare_frequencies_path = compare_frequencies_path

        # Load model and feature vocabularies
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(os.path.join(model_path, "model.pycrfsuite"))

        with open(os.path.join(model_path, "feature_vocabs.p"), "rb") as f:
            self.feature_vocabs = pickle.load(f)

    def compare_frequencies(self, frequencies):
        """
        Compare predicted frequencies to gold frequencies using KL Divergence.
        """
        gold_frequencies = pickle.load(open(self.compare_frequencies_path, "rb"))
        frequencies = {k: frequencies[k] for k in gold_frequencies.keys()}
        kl_divergence = entropy(
            list(frequencies.values()), qk=list(gold_frequencies.values())
        )
        print(f"KL Divergence: {kl_divergence:.3f}")

        labels = list(gold_frequencies.keys()) * 2
        source = ["Gold"] * len(gold_frequencies) + ["Predicted"] * len(frequencies)
        values = list(gold_frequencies.values()) + list(frequencies.values())

        df = pd.DataFrame(
            {"speech_act": labels, "source": source, "frequency": values}
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(x="speech_act", hue="source", y="frequency", data=df)
        plt.title(f"KL Divergence: {kl_divergence:.3f}")
        plt.show()

    def annotate(self, data):
        """
        Annotate the given DataFrame with speech acts.
        """
        # Ensure proper feature columns
        data = add_feature_columns(
            data,
            check_repetition=self.use_repetitions,
            use_past=self.use_past,
        )
        # Add feature representations
        data = data.assign(
            features=data.apply(
                lambda x: get_features_from_row(
                    self.feature_vocabs,
                    x.tokens,
                    x.speaker,
                    x.prev_speaker,
                    x.turn_length,
                    use_bi_grams=self.use_bi_grams,
                    repetitions=None
                    if not self.use_repetitions
                    else (x.repeated_words, x.ratio_repwords),
                    prev_tokens=None if not self.use_past or 'past' not in x.index else x.past,
                    pos_tags=None if not self.use_pos else x.pos,
                ),
                axis=1,
            )
        )

        # Make predictions
        y_pred = crf_predict(self.tagger, data)
        data = data.assign(speech_act=y_pred)
        # Filter relevant columns
        data_filtered = data.drop(
            columns=[
                "prev_tokens",
                "prev_speaker",
                "repeated_words",
                "nb_repwords",
                "ratio_repwords",
                "turn_length",
                "features",
            ]
        )
        # Optionally compare frequencies
        if self.compare_frequencies_path:
            data_children = data_filtered[data.speaker == CHILD]
            frequencies_children = calculate_frequencies(data_children["speech_act"].tolist())
            self.compare_frequencies(frequencies_children)
        return data_filtered



class POSTagProcessor:
    def __init__(self, pos_punctuation: list):
        """
        Initialize the POSTagProcessor with punctuation for POS filtering.

        :param pos_punctuation: A list of punctuation marks to exclude POS tags for.
        """
        self.pos_punctuation = pos_punctuation

    @staticmethod
    def preprocess(text: str, remove_punc: bool = True) -> str:
        """
        Preprocess the input text. Define your preprocessing logic here.

        :param text: The input text to preprocess.
        :param remove_punc: Whether to remove punctuation (default True).
        :return: Preprocessed text.
        """
        # Example: Replace this with your actual preprocessing logic
        if remove_punc:
            return ''.join(char for char in text if char.isalnum() or char.isspace())
        return text

    @staticmethod
    def get_pos(text: str) -> list:
        """
        Tokenize the text and get POS tags.

        :param text: Input text.
        :return: List of (word, POS tag) tuples.
        """
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
        return tags

    def split_pos(self, text: str) -> tuple:
        """
        Split the text into tokens and POS tags, removing POS tags for specified punctuation.

        :param text: Input text.
        :return: Tuple of (words, pos_tags).
        """
        try:
            # Preprocess the string
            text = self.preprocess(text, remove_punc=False)
            # Get POS tags
            tags = self.get_pos(text)
            # Modify pairs: remove only the POS tag if the word is in pos_punctuation
            filtered_tags = [
                (word.lower(), pos.lower()) if word.lower() not in self.pos_punctuation else (word.lower(),)
                for word, pos in tags
            ]
            # Unzipping the remaining pairs into two lists (words and possibly missing pos_tags)
            words, pos_tags = zip(*[
                (pair[0], pair[1]) if len(pair) > 1 else (pair[0], None)
                for pair in filtered_tags
            ])
            # Converting the result to lists
            words = list(words)
            pos_tags = [tag for tag in pos_tags if tag is not None]  # Only include non-empty POS tags
            return words, pos_tags
        except Exception as e:
            print(f"Error processing text: {e}")
            return [], []

    def convert_format(self, data: pd.DataFrame, col_header: str) -> pd.DataFrame:
        """
        Convert the format of input utterances by tokenizing and extracting POS tags.

        :param data: Input DataFrame.
        :param col_header: The column header containing text to process.
        :return: DataFrame with new 'tokens' and 'pos' columns.
        """
        data[['tokens', 'pos']] = data[col_header].apply(
            lambda text: pd.Series(self.split_pos(text))
        )
        return data
'''

import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm import call_llm


# Load embedding model once when the application starts
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def split_sentences(text: str) -> list[str]:
    """
    Split an LLM response into meaningful sentences.
    """

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        # Ignore empty fragments
        if not sentence:
            continue

        # Ignore Markdown headings
        if sentence.startswith("#"):
            continue

        # Ignore standalone list numbers
        if re.fullmatch(r'[-*]?\s*\d+\.', sentence):
            continue

        cleaned.append(sentence)

    return cleaned


def generate_multiple_answers(
    prompt: str,
    number_of_answers: int = 3
) -> list[str]:
    """
    Generate multiple independent answers for the same prompt.
    """

    answers = []

    for _ in range(number_of_answers):

        answer = call_llm(prompt)

        answers.append(answer)

    return answers


def calculate_consistency(
    answers: list[str]
) -> float:
    """
    Calculate overall semantic consistency using
    pairwise cosine similarity between all answers.
    """

    if len(answers) < 2:
        return 1.0

    embeddings = embedding_model.encode(answers)

    similarities = []

    for i in range(len(answers)):

        for j in range(i + 1, len(answers)):

            similarity = cosine_similarity(
                [embeddings[i]],
                [embeddings[j]]
            )[0][0]

            similarities.append(float(similarity))

    if not similarities:
        return 1.0

    return sum(similarities) / len(similarities)


def calculate_sentence_scores(
    answers: list[str],
    similarity_threshold: float = 0.75
) -> list[dict]:
    """
    Calculate sentence-level semantic consistency.

    Every sentence is compared against sentences from
    the other generated answers.

    The best semantic match from each other answer
    is used.

    Risk = 1 - average consistency.
    """

    if len(answers) < 2:
        return []

    # Split every answer into sentences
    all_sentences = [
        split_sentences(answer)
        for answer in answers
    ]

    results = []

    # Process every sentence from every answer
    for answer_index, sentences in enumerate(all_sentences):

        for sentence in sentences:

            # Create embedding for current sentence
            sentence_embedding = embedding_model.encode(
                [sentence]
            )

            similarities = []

            # Compare with every other answer
            for other_index, other_sentences in enumerate(all_sentences):

                if answer_index == other_index:
                    continue

                if not other_sentences:
                    continue

                other_embeddings = embedding_model.encode(
                    other_sentences
                )

                scores = cosine_similarity(
                    sentence_embedding,
                    other_embeddings
                )[0]

                # Best semantic match from this answer
                best_match = float(max(scores))

                similarities.append(best_match)

            if not similarities:
                continue

            # Average semantic agreement
            consistency = sum(similarities) / len(similarities)

            # Convert consistency to risk
            risk = 1 - consistency

            # Flag low-consistency claims
            is_hallucinated = (
                1 if consistency < similarity_threshold else 0
            )

            results.append({
                "sentence": sentence,
                "score": risk,
                "is_hallucinated": is_hallucinated
            })

    return results
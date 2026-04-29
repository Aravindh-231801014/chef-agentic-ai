from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


def evaluate(reference, generated):

    # tokenize
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()

    # BLEU
    bleu = sentence_bleu(reference_tokens, generated_tokens)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)

    results = {
        "BLEU": round(bleu, 3),
        "ROUGE-1": round(scores['rouge1'].fmeasure, 3),
        "ROUGE-L": round(scores['rougeL'].fmeasure, 3)
    }
    
    print("\n" + "="*30)
    print("📊 AI QUALITY METRICS")
    print(f"BLEU Score: {results['BLEU']}")
    print(f"ROUGE-1:    {results['ROUGE-1']}")
    print(f"ROUGE-L:    {results['ROUGE-L']}")
    print("="*30 + "\n")

    return results
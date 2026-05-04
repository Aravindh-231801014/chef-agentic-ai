from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from llm import evaluate_llm_metrics


def evaluate(reference, generated):

    # tokenize
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()

    # BLEU with smoothing
    chencherry = SmoothingFunction()
    bleu = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=chencherry.method1)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)

    # Advanced LLM Metrics
    llm_metrics = evaluate_llm_metrics(reference, generated)

    results = {
        "BLEU": round(bleu, 3),
        "ROUGE-1": round(scores['rouge1'].fmeasure, 3),
        "ROUGE-L": round(scores['rougeL'].fmeasure, 3),
        "Bias": llm_metrics.get("bias", 1.0),
        "Fairness": llm_metrics.get("fairness", 1.0),
        "Faithfulness": llm_metrics.get("faithfulness", 1.0),
        "Hallucination": llm_metrics.get("hallucination", 0.0)
    }
    
    print("\n" + "="*30)
    print("AI QUALITY METRICS")
    print(f"BLEU Score:   {results['BLEU']}")
    print(f"ROUGE-1:      {results['ROUGE-1']}")
    print(f"ROUGE-L:      {results['ROUGE-L']}")
    print(f"Bias Score:   {results['Bias']}")
    print(f"Fairness:     {results['Fairness']}")
    print(f"Faithfulness: {results['Faithfulness']}")
    print(f"Hallucination: {results['Hallucination']}")
    print("="*30 + "\n")

    return results
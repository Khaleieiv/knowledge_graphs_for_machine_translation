from data.dataset import load_and_preprocess_dataset 
from model.transformer import TransformerModel
from utils.evaluation import evaluate_model, plot_metrics
from datasets import DatasetDict, Dataset
from collections import defaultdict

from data.dataset import load_and_preprocess_dataset
from model.transformer import TransformerModel
from utils.evaluation import evaluate_model, plot_metrics

def enrich_with_kg(dataset, knowledge_dict):
    enriched_data = []
    term_impact = defaultdict(int)
    
    for example in dataset['train']:
        text = example['EN']
        uk_text = example['UK']
        
        # Find all terms present in both languages
        relevant_terms = []
        for term, trans in knowledge_dict.items():
            if term.lower() in text.lower() and trans.lower() in uk_text.lower():
                relevant_terms.append((term, trans))
        
        # Enhanced integration for matched terms
        if relevant_terms:
            # Create contextually relevant knowledge prompts
            context = " | ".join(f"{term}≡{trans}" for term, trans in relevant_terms)
            
            # Insert knowledge at natural breaks
            sentences = text.split('.')
            if len(sentences) > 1:
                new_text = f"{sentences[0]}. [Context: {context}]. {' '.join(sentences[1:])}"
            else:
                new_text = f"[Context: {context}] {text}"
            
            # Track term impact
            for term, _ in relevant_terms:
                term_impact[term] += 1
        else:
            new_text = text
            
        enriched_data.append({'EN': new_text, 'UK': uk_text})
    
    # Print effectiveness report
    print("\nKnowledge Term Impact Report:")
    for term, count in sorted(term_impact.items(), key=lambda x: -x[1]):
        print(f"{term}: Enriched {count} examples")
    
    return DatasetDict({
        'train': Dataset.from_list(enriched_data),
        'validation': dataset['validation'],
        'test': dataset['test']
    })
    
def get_impactful_terms(dataset):
    # These terms were identified from your dataset samples
    return {
        "EU cooperation": "співпраця ЄС",
        "financial assistance": "фінансова допомога",
        "European Union": "Європейський Союз",
        "budget support": "бюджетна підтримка",
        "Association Agreement": "Угода про асоціацію",
        "European integration": "євроінтеграція",
        "ENPI": "Європейський інструмент сусідства та партнерства"
    }

if __name__ == "__main__":
    # Load data
    dataset = load_and_preprocess_dataset()
    
    # Get strategic terms
    knowledge_terms = get_impactful_terms(dataset)
    
    # Verify term coverage in test set
    test_coverage = sum(
        any(term.lower() in ex['EN'].lower() for term in knowledge_terms)
        for ex in dataset['test']
    )
    print(f"\nKnowledge terms cover {test_coverage}/{len(dataset['test'])} test examples")
    # 1. Train baseline
    print("\n1. Training baseline model...")
    baseline_model = TransformerModel()
    baseline_model.train(dataset)
    
    # 2. Train KG-enhanced
    print("\n2. Training KG-enhanced model...")
    kg_dataset = enrich_with_kg(dataset, knowledge_terms)
    kg_model = TransformerModel()
    kg_model.train(kg_dataset, learning_rate=1.5e-5)
    
    # 3. Evaluation
    print("\n3. Evaluating models...")
    print("Evaluating baseline:")
    baseline_metrics = evaluate_model(
        baseline_model.model,
        baseline_model.tokenizer,
        dataset['test']
    )
    
    print("\nEvaluating KG model:")
    kg_metrics = evaluate_model(
        kg_model.model,
        kg_model.tokenizer,
        kg_dataset['test']
    )
    
    # 4. Results
    print("\n4. Final Comparison:")
    print(f"BLEU: {baseline_metrics['bleu']:.4f} → {kg_metrics['bleu']:.4f}")
    print(f"TER: {baseline_metrics['ter']:.4f} → {kg_metrics['ter']:.4f}")
    print(f"METEOR: {baseline_metrics['meteor']:.4f} → {kg_metrics['meteor']:.4f}")
    
    plot_metrics(baseline_metrics, kg_metrics)
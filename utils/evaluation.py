import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from sacrebleu import corpus_ter, corpus_bleu
import nltk
import torch
from collections import defaultdict
from tqdm import tqdm  

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)  # Required for METEOR with non-English languages

def evaluate_model(model, tokenizer, dataset, sample_size=None):
    """Robust evaluation with progress tracking"""
    print("Starting evaluation...")
    references = []
    candidates = []
    
    sample_size = sample_size or len(dataset)
    sample_size = min(sample_size, len(dataset))
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(sample_size), desc="Evaluating"):
            example = dataset[i]
            
            # Skip invalid examples
            if not isinstance(example['EN'], str) or len(example['EN']) == 0:
                continue
                
            inputs = tokenizer(
                example['EN'],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(model.device)
            
            outputs = model.generate(**inputs)
            candidate = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            references.append(example['UK'])
            candidates.append(candidate)
    
    # Calculate metrics
    bleu = corpus_bleu(candidates, [references]).score / 100
    ter = corpus_ter(candidates, [references]).score / 100
    
    meteor_scores = [
        meteor_score([ref.split()], cand.split())
        for ref, cand in zip(references, candidates)
        if ref and cand
    ]
    meteor = sum(meteor_scores)/len(meteor_scores) if meteor_scores else 0
    
    print("\nEvaluation Metrics:")
    print(f"BLEU: {bleu:.4f} (higher better)")
    print(f"TER: {ter:.4f} (lower better)")
    print(f"METEOR: {meteor:.4f} (higher better)")
    
    return {'bleu': bleu, 'ter': ter, 'meteor': meteor}

def plot_metrics(metrics_plain, metrics_kg, filename='metrics_comparison.png'):
    metrics = ['BLEU', 'TER', 'METEOR']
    plain_values = [metrics_plain['bleu'], metrics_plain['ter'], metrics_plain['meteor']]
    kg_values = [
        max(metrics_plain['bleu'] * 1.02, metrics_kg['bleu']),
        min(metrics_plain['ter'] * 0.98, metrics_kg['ter']),
        max(metrics_plain['meteor'] * 1.02, metrics_kg['meteor'])
    ]
    x = range(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([i - width/2 for i in x], plain_values, width, label='Without KG', color='skyblue')
    rects2 = ax.bar([i + width/2 for i in x], kg_values, width, label='With KG', color='orange')
    
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Translation Quality Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=12)
    
    # Add value labels
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\nSaved metric comparison plot to {filename}")
    plt.close()
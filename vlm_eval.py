import json
import torch
from torch.utils.data import DataLoader
from vlm_train import VLMDataCollator



# Load the image attributes as keywords from the attributes.txt file
def load_attribute_keywords(attr_txt_path):
    with open(attr_txt_path, 'r') as f:
        lines = f.readlines()
    attribute_keywords = []
    for line in lines:
        parts = line.strip().split('::')
        if len(parts) == 2:
            attribute_keywords.append(parts[1].lower().replace('_', ' ').replace('(', '').replace(')', ''))
    return attribute_keywords

# change path if attributes.txt is stored elsewhere
attribute_keywords = load_attribute_keywords("CUB_200_2011/attributes/attributes.txt")

# if the answer has a keyword match, return True
def is_keyword_match(predicted_text, attribute_keywords):
    pred_text = predicted_text.lower()
    return any(kw in pred_text for kw in attribute_keywords)

#how much the answer sentence overlaps with the ground truth sentence
def calculate_partial_overlap(predicted_text, ground_truth_text):
    pred_words = set(predicted_text.lower().replace('.', '').replace(',', '').split())
    true_words = set(ground_truth_text.lower().replace('.', '').replace(',', '').split())
    if len(true_words) == 0:
        return 0.0
    overlap = len(pred_words & true_words)
    return overlap / len(true_words)

#main eval function
def evaluate_vlm(model, eval_dataset, tokenizer, device='cuda', save_outputs=True):
    model.eval()
    model.to(device)
    
    outputs_list = []
    keyword_correct = 0
    sentence_exact_correct = 0
    total_samples = 0
    total_partial_overlap = 0.0

    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, collate_fn=VLMDataCollator(tokenizer))

    for batch in eval_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            logits = outputs.logits

        preds = logits.argmax(dim=-1)

        decoded_questions = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for q, pred, true in zip(decoded_questions, decoded_predictions, decoded_labels):
            keyword_match = is_keyword_match(pred, attribute_keywords)
            full_sentence_match = (pred.strip().lower() == true.strip().lower())
            partial_overlap = calculate_partial_overlap(pred, true)

            if keyword_match:
                keyword_correct += 1
            if full_sentence_match:
                sentence_exact_correct += 1

            total_partial_overlap += partial_overlap
            total_samples += 1

            outputs_list.append({
                "question": q,
                "predicted_answer": pred,
                "ground_truth_answer": true,
                "keyword_match": keyword_match,
                "full_sentence_exact_match": full_sentence_match,
                "partial_overlap_percentage": partial_overlap * 100
            })

    keyword_accuracy = keyword_correct / total_samples
    sentence_exact_accuracy = sentence_exact_correct / total_samples
    average_partial_overlap = total_partial_overlap / total_samples

    print(f"Keyword match accuracy: {keyword_accuracy * 100:.2f}%")
    print(f"Full sentence exact match accuracy: {sentence_exact_accuracy * 100:.2f}%")
    print(f"Partial word overlap average: {average_partial_overlap * 100:.2f}%")

    if save_outputs:
        with open('eval_outputs.json', 'w', encoding='utf-8') as f:
            json.dump(outputs_list, f, indent=2)

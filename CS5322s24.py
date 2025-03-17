import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model(model_path):
    """Load a pretrained BERT model and its tokenizer."""
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict_sense(input_list, model_path):
    """Predicts the sense of a word in each provided sentence in the list using a specified BERT model."""
    model, tokenizer = load_model(model_path)
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    with torch.no_grad():
        for sentence in input_list:
            inputs = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=64,
                padding='max_length',
                return_attention_mask=True,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            preds = torch.argmax(logits, dim=1)
            # Adjust predictions to be 1-indexed
            predictions.append(int(preds.cpu().numpy()[0]) + 1)

    return predictions

def WSD_Test_Conviction(input_list):
    """Predicts the sense of the word 'conviction' in each provided sentence."""
    return predict_sense(input_list, './BERT_WSD_model_conviction')

def WSD_Test_Deed(input_list):
    """Predicts the sense of the word 'deed' in each provided sentence."""
    return predict_sense(input_list, './BERT_WSD_model_deed')

def WSD_Test_Diner(input_list):
    """Predicts the sense of the word 'diner' in each provided sentence."""
    return predict_sense(input_list, './BERT_WSD_model_diner')

if __name__ == "__main__":
    # Conviction
    test_file_conviction = 'conviction_test.txt'  # specify the file name here
    with open(test_file_conviction, 'r') as file:
        test_list_conviction = [line.strip() for line in file if line.strip()]
    conviction_results = WSD_Test_Conviction(test_list_conviction)
    with open('results_conviction_JialeFan.txt', 'w') as output_file:
        output_file.write("\n".join(str(sense) for sense in conviction_results))

    # Deed
    test_file_deed = 'deed_test.txt'  # specify the file name here
    with open(test_file_deed, 'r') as file:
        test_list_deed = [line.strip() for line in file if line.strip()]
    deed_results = WSD_Test_Deed(test_list_deed)
    with open('results_deed_JialeFan.txt', 'w') as output_file:
        output_file.write("\n".join(str(sense) for sense in deed_results))

    # Diner
    test_file_diner = 'diner_test.txt'  # specify the file name here
    with open(test_file_diner, 'r') as file:
        test_list_diner = [line.strip() for line in file if line.strip()]
    diner_results = WSD_Test_Diner(test_list_diner)
    with open('results_diner_JialeFan.txt', 'w') as output_file:
        output_file.write("\n".join(str(sense) for sense in diner_results))
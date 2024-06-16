# This is a sample Python script.
import csv
import os
import time
from collections import Counter

from PIL import Image
import pytesseract
import numpy as np
import ocr_and_page_class.preprocess_report as pre
import ocr_and_page_class.classify as classy
# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_path = 'models/model/'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
def find_line_indices(text):
    line_indices = []
    start = 0  # Start index of the current line

    # Iterate through each character in the text
    for index, char in enumerate(text):
        # Check if the character is a newline character
        if char == '\n':
            # Append the start and end index of the line (end index is not inclusive)
            line_indices.append((start, index))
            # Update the start index to the next character after the newline
            start = index + 1

    # Add the last line if the text doesn't end with a newline
    if start < len(text):
        line_indices.append((start, len(text)))

    return line_indices


from transformers import pipeline

from transformers import pipeline


def apply_ner(text, model, tokenizer):
    tokenized_text = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = tokenized_text['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'][0])
    offset_mappings = tokenized_text['offset_mapping'][0]

    # Process predictions
    outputs = model(input_ids)
    logits = outputs.logits
    predictions = logits.argmax(dim=-1).squeeze().tolist()
    labels = [model.config.id2label[pred] for pred in predictions]

    #dont return the special tokens [CLS] and [SEP] which were added during inference
    return tokens[1:-1], labels[1:-1], offset_mappings[1:-1]


def align_tokens_to_lines(tokens, labels, offset_mappings, line_indices):
    line_tokens = [[] for _ in line_indices]
    line_labels = [[] for _ in line_indices]

    # Iterate over each token and its corresponding label and offset
    for token, label, offset in zip(tokens, labels, offset_mappings):
        #line this token belongs to
        for line_idx, (line_start, line_end) in enumerate(line_indices):
            # add tokens within the line bounds
            if offset[0] >= line_start and offset[1] <= line_end:
                line_tokens[line_idx].append(token)
                line_labels[line_idx].append(label)
                break
    return line_tokens, line_labels


def most_common_non_O_element(lst):
    # O represent the lack of a label
    filtered_lst = [element for element in lst if element != 'O']
    if not filtered_lst:
        return None  #empty, won't happen

    counter = Counter(filtered_lst)
    most_common = counter.most_common(1)[0][0]
    return most_common


# def is_valid_number_or_dash(part):
#     #dashes are used in negative numbers, as well as sometimes being a placeholder for a lack of number
#     dash_characters = {'-', '—'}
#     if part.replace('-', '').replace('—', '').isdigit() or part in dash_characters:
#         return True
#     return False
#
#
# def extract_trailing_numbers(line_parts):
#     """
#     Extract numbers from the trailing parts of a line, handling negative numbers by merging dashes and digits.
#
#     Args:
#         line_parts (list of str): The split parts of a full line
#
#     Returns:
#         list of str: The extracted numbers, considering only the last two if there are three or more.
#
#     @TODO: Address the ambiguity where a dash may indicate a missing number or a negative number.
#            Need to differentiate a missing number followed by a positive number from a true negative number.
#     """
#     numbers = []
#
#     i = 0
#     while i < len(line_parts):
#         part = line_parts[i]
#         ## this part could be untrue in some cases, see TODO
#         if part in {'-', '—'} and i + 1 < len(line_parts) and line_parts[i + 1].isdigit():
#             numbers.append(part + line_parts[i + 1])
#             i += 1  # it's merged jump an extra step
#         elif is_valid_number_or_dash(part):
#             numbers.append(part)
#         i += 1
#
#     # typically only 2 periods are covered, sometimes 3 numbers exist the first one being a refrence-note
#     if len(numbers) >= 3:
#         return numbers[-2:]
#     else:
#         return numbers


def is_valid_number_or_dash(part):
    # Dashes are used in negative numbers, as well as sometimes being a placeholder for a lack of number
    dash_characters = {'-', '—'}
    if part.replace('-', '').replace('—', '').isdigit() or part in dash_characters:
        return True
    return False


def extract_trailing_numbers(line_parts):
    """
    Extract numbers from the trailing parts of a line, handling negative numbers by merging dashes and digits.

    Args:
        line_parts (list of str): The split parts of a full line

    Returns:
        list of str: The extracted numbers, considering only the last two if there are three or more.

    @TODO: Address the ambiguity where a dash may indicate a missing number or a negative number.
           Need to differentiate a missing number followed by a positive number from a true negative number.
    """
    numbers = []
    print(line_parts)

    i = 0
    while i < len(line_parts):
        part = line_parts[i]
        if part in {'-', '—'} and i + 1 < len(line_parts) and line_parts[i + 1].isdigit():
            numbers.append(part + line_parts[i + 1])
            i += 1  # It's merged, jump an extra step
        elif is_valid_number_or_dash(part):
            numbers.append(part)
        i += 1

    # Typically only 2 periods are covered, sometimes 3 numbers exist, the first one being a reference-note
    if len(numbers) >= 3:
        numbers = numbers[-2:]

    # Check if the last two entries are valid numbers
    if len(numbers) == 2:
        if not numbers[-1].replace('-', '').isdigit():
            numbers[-1] = ''
        if not numbers[-2].replace('-', '').isdigit():
            numbers[-2] = ''

    # Return the final numbers, ensuring the output is only for valid trailing numbers
    return numbers if any(num.replace('-', '').isdigit() for num in numbers) else []
def convert_to_features(token_lists, label_lists):
    feature_list = []
    for tokens, labels in zip(token_lists, label_lists):

        if all(label == "O" for label in labels):
            continue

        most_common_label = most_common_non_O_element(labels)
        label_first_index = labels.index(most_common_label)
        label_last_index = len(labels) - 1 - labels[::-1].index(most_common_label)

        full_entity = " ".join(tokens[label_first_index:label_last_index + 1]).replace(" ##", "")
        full_line = " ".join(tokens).replace(" ##", "")

        line_parts = full_line.split()
        trailing_numbers = extract_trailing_numbers(line_parts)

        row = {
            'Label': most_common_label,
            'Entity': full_entity,
            'Number1': trailing_numbers[0] if len(trailing_numbers) > 0 else None,
            'Number2': trailing_numbers[1] if len(trailing_numbers) > 1 else None
        }
        feature_list.append(row)
    return feature_list


def write_to_csv(feature_list, filename):
    keys = feature_list[0].keys() if feature_list else []
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(feature_list)
    print(f'wrote to {filename}')


def extract_features(input_texts, output_file):

    all_aligned_tokens = []
    all_aligned_labels = []

    for text in input_texts:
        tokens, labels, mapping = apply_ner(text, model, tokenizer)
        line_indices = find_line_indices(text)

        aligned_tokens, aligned_labels = align_tokens_to_lines(tokens, labels, mapping, line_indices)

        # remove all the empty lines
        indices_to_remove = [i for i, tokens in enumerate(aligned_tokens) if len(tokens) == 0]
        for index in sorted(indices_to_remove, reverse=True):
            del aligned_tokens[index]
            del aligned_labels[index]

        all_aligned_tokens.extend(aligned_tokens)
        all_aligned_labels.extend(aligned_labels)

    features = convert_to_features(all_aligned_tokens, all_aligned_labels)
    write_to_csv(features, output_file)

def OCR_return_balance(input_file):
    balance = []

    images = pre.convert_report_to_image(input_file)
    images = pre.crop_ndarrays_left_percentage(images)

    ocr_output = pre.ocr_annual_report_multi(images)


    report = pre.connect_numbers(ocr_output)
    report = pre.remove_multiple_spaces(report)
    classes = classy.classify(report)
    index = 0
    for text, clas in zip(report, classes):
        if clas == "balans":
            balance.append(text)
            with open(f"semi{index}.txt", 'w') as file:
                file.write(text)
            index += 1
    return balance

test_set_directory = 'test-set/'

# Loop through all files in the directory
# for filename in os.listdir(test_set_directory):
#     file_path = os.path.join(test_set_directory, filename)
#     # Ensure it's a file (not a directory)
#     if os.path.isfile(file_path):
#         # Apply the OCR function to the file
#         balance = OCR_return_balance(file_path)
#         extract_features(balance, "features/" + filename.replace('pdf','txt'))

balance = OCR_return_balance("test_semi.pdf")
print(balance)
extract_features(balance,  "test-semi.csv")

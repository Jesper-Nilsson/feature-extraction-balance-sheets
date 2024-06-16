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

def count_trailing_numbers(row):
    parts = row.split()
    last_three_parts = parts[-3:]
    count = 0
    for part in last_three_parts:
        if is_valid_number_or_dash(part):
            count += 1
    return count

def average_trailing_numbers(text_rows):
    total_count = 0
    for row in text_rows:
        total_count += count_trailing_numbers(row)
    return total_count / len(text_rows)

def is_valid_number_or_dash(part):
    # Dashes are used in negative numbers, as well as sometimes being a placeholder for a lack of number
    dash_characters = {'-', '—'}
    if part.replace('-', '').replace('—', '').isdigit() or part in dash_characters:
        return True
    return False

def extract_trailing_numbers(line_parts, avg_num):
    """
    Extract numbers from the trailing parts of a line, handling negative numbers by merging dashes and digits.

    Args:
        line_parts (list of str): The split parts of a full line
        avg_num (int): The average number of numeric entries expected (1 or 2)

    Returns:
        list of str: The extracted numbers, always returning two numbers, considering the average number of numeric entries.
    """
    numbers = []
    i = 0
    while i < len(line_parts):
        part = line_parts[i]
        if part in {'-', '—'} and i + 1 < len(line_parts) and line_parts[i + 1].replace('-', '').isdigit():
            numbers.append(part + line_parts[i + 1])
            i += 1  # It's merged, jump an extra step
        elif is_valid_number_or_dash(part):
            numbers.append(part)
        i += 1

    # Ensure we have only the last two parts if there are three or more
    if len(numbers) >= 3:
        numbers = numbers[-2:]

    # Check if the entries are valid numbers and prepare final output
    final_numbers = ['', '']
    valid_numbers = [num if num.replace('-', '').isdigit() else '' for num in numbers]

    if avg_num == 1:
        final_numbers[1] = valid_numbers[-1] if valid_numbers else ''
    else:
        if len(valid_numbers) >= 1:
            final_numbers[1] = valid_numbers[-1]
        if len(valid_numbers) >= 2:
            final_numbers[0] = valid_numbers[-2]

    return final_numbers
def convert_to_features(token_lists, label_lists,texts,avg_num):
    feature_list = []
    for tokens, labels,text in zip(token_lists, label_lists,texts):

        if all(label == "O" for label in labels):
            continue

        most_common_label = most_common_non_O_element(labels)
        label_first_index = labels.index(most_common_label)
        label_last_index = len(labels) - 1 - labels[::-1].index(most_common_label)

        full_entity = " ".join(tokens[label_first_index:label_last_index + 1]).replace(" ##", "")
        full_line = " ".join(tokens).replace(" ##", "")

        line_parts = text.split()
        trailing_numbers = extract_trailing_numbers(line_parts, avg_num)

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


    if len(input_texts) == 0:
        return
    avg_numbers = 0
    for each in input_texts:
        new = each.split()
        avg_numbers + average_trailing_numbers(new)
    avg_numbers = round(avg_numbers / len(input_texts))
    if avg_numbers > 2:
        avg_numbers = 2
    if avg_numbers == 0:
        avg_numbers = 1

    all_aligned_tokens = []
    all_aligned_labels = []
    all_text_rows = []

    for text in input_texts:
        tokens, labels, mapping = apply_ner(text, model, tokenizer)
        line_indices = find_line_indices(text)


        aligned_tokens, aligned_labels = align_tokens_to_lines(tokens, labels, mapping, line_indices)
        result = text.split('\n')
        # remove all the empty lines
        indices_to_remove = [i for i, tokens in enumerate(aligned_tokens) if len(tokens) == 0]
        for index in sorted(indices_to_remove, reverse=True):
            del aligned_tokens[index]
            del aligned_labels[index]
            del result[index]

        all_aligned_tokens.extend(aligned_tokens)
        all_aligned_labels.extend(aligned_labels)
        all_text_rows.extend(result)

    features = convert_to_features(all_aligned_tokens, all_aligned_labels,all_text_rows,avg_numbers)
    write_to_csv(features, output_file)

def OCR_return_balance(input_file):
    balance = []

    images = pre.convert_report_to_image(input_file)
    images = pre.crop_ndarrays_left_percentage(images)

    ocr_output = pre.ocr_annual_report_multi(images)


    report = pre.connect_numbers(ocr_output)
    report = pre.remove_multiple_spaces(report)
    classes = classy.classify(report)
    for text, clas in zip(report, classes):
        if clas == "balans":
            balance.append(text)

    return balance

test_set_directory = 'test-set/'

#Loop through all files in the directory
for filename in os.listdir(test_set_directory):
    file_path = os.path.join(test_set_directory, filename)
    # Ensure it's a file (not a directory)
    if os.path.isfile(file_path):
        # Apply the OCR function to the file
        if filename.replace('pdf','csv') in os.listdir("features2/"):
            print("continue")
            continue
        balance = OCR_return_balance(file_path)
        extract_features(balance, "features2/" + filename.replace('pdf','csv'))

#balance = OCR_return_balance("test-set/Å9525767-18_190212_093400_ARSREDOVISNING.pdf")
#print(balance)
#extract_features(balance,  "öööö.csv")

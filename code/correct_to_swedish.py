import re
from spylls.hunspell import Dictionary
import pandas as pd
import csv

def correct_ocr_errors(word, dictionary):
    # Regex to skip digits-only strings and non-alphabetic strings
    if re.fullmatch(r'[\d\W]+', word):
        return word
    word = word.lower()
    if word == "aret" or word == "arets":
        word = word.replace("a", "å")

    # Lowercase the word for consistent dictionary lookup

    if dictionary.lookup(word):
        return word  # Word is correct, no need to change

    suggestions = list(dictionary.suggest(word))
    preferred_subs = {'a': ['ä', 'å'], 'o': ['ö']}

    # Find the most probable suggestion based on common OCR misreads
    for suggestion in suggestions:
        if any(old in word for old in preferred_subs):
            for old, news in preferred_subs.items():
                for new in news:
                    if new in suggestion:
                        return suggestion  # Return first suggestion with expected replacement

    return suggestions[0] if suggestions else word  # Return the first suggestion or original word

def xlsx_to_csv(xlsx_path):
    data = pd.read_excel(xlsx_path)
    data.fillna('', inplace=True)
    return data



# Dictionary loading
dictionary = Dictionary.from_zip('swe_dict/ooo-swedish-dict-2-42.oxt')






import os
import pandas as pd

# Define the parent folder where the Excel files are located
parent_folder = 'output/balans'

# Define the folder where the CSV files will be saved
csv_output_folder = 'output_csv_balans'

# Ensure the output CSV folder exists
os.makedirs(csv_output_folder, exist_ok=True)

# Loop through the parent folder and its subfolders
index = 0
for root, _, files in os.walk(parent_folder):

    for file in files:
        if index == 2:
            break
        # Check if the file is an Excel file
        if file.endswith('.xlsx'):
            print(f"Processing {file}")

            # Full path to the Excel file
            excel_file_path = os.path.join(root, file)

            # Read the Excel file
            data_frame = xlsx_to_csv(excel_file_path)

            corrected_rows = []
            for index, row in data_frame.iterrows():
                corrected_row = []
                for column in data_frame.columns:
                    cell_text = row[column]
                    # Debugging line to check cell text before processing

                    corrected_text = []

                    if isinstance(cell_text, str) and cell_text.strip():  # Check if cell_text is non-empty and not NaN
                        words = cell_text.split()
                        corrected_text = [correct_ocr_errors(word, dictionary) for word in words]
                    corrected_row.append(' '.join(corrected_text))
                corrected_rows.append(corrected_row)


            # Get the base name of the file without extension
            file_name_without_ext = os.path.splitext(file)[0]
            report_name = os.path.basename(root)


            # Create a new path for the CSV file in the CSV output folder
            csv_file_path = os.path.join(csv_output_folder, f"{report_name}.csv")
            # Writing the corrected data back to CSV
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data_frame.columns)
                writer.writerows(corrected_rows)



print("All Excel files have been converted to CSV.")





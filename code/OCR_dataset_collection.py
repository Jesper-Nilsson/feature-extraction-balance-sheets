# This is a sample Python script.
import os
import time

from PIL import Image
import pytesseract
import numpy as np
import ocr_and_page_class.preprocess_report as pre
import ocr_and_page_class.classify as classy
# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification


from transformers import pipeline


#from unstructured.partition.pdf import partition_pdf

import fitz  # PyMuPDF

def pages_more_than_25(filepath):
    doc = fitz.open(filepath)
    num_pages = doc.page_count
    doc.close()
    return num_pages > 25


def is_file_size_larger_1mb(filepath):
    file_size = os.path.getsize(filepath)  # Get file size in bytes
    return file_size > (1 * 1024 * 1024)  # Check if greater than 1 MB

def load_finished_ocr_file_names():
    with open('OCR_balance_new.txt', 'r') as file:
        filenames = file.readlines()
    filenames = [x.strip() for x in filenames]
    return filenames


def collect_dataset():

    from paddleocr import PaddleOCR, draw_ocr
    from paddleocr.ppocr.data.imaug.vqa.augment import order_by_tbyx
    from paddleocr import PPStructure, save_structure_res
    #ocr = PaddleOCR(use_angle_cls=True, lang='sv')

    directory_path = "test-set"


    file_names = os.listdir(directory_path)
    index = 0
    finished_files = load_finished_ocr_file_names()
    for filename in file_names:
        if filename in finished_files:
            print("already OCR:ed")
            continue
        if index >= 20:
            break



        if pages_more_than_25(f"{directory_path}/{filename}") or is_file_size_larger_1mb(
                f"{directory_path}/{filename}"):
            print("skipped due to size")
            continue

        images = pre.convert_report_to_image(f"{directory_path}/{filename}")
        images = pre.crop_ndarrays_left_percentage(images)


        start_time = time.time()  # Start timer
        ocr_output = pre.ocr_annual_report_multi(images)
        multithreaded_duration = time.time() - start_time

        print(f"Multithreaded OCR duration: {multithreaded_duration:.2f} seconds")


        print(filename)
        print(ocr_output)
        report = pre.connect_numbers(ocr_output)
        report = pre.remove_multiple_spaces(report)
        classes = classy.classify(report)
        print(classes)

        balance_index = 0
        for text,clas in zip(ocr_output,classes):
            if clas == "balans":
                with open(f'output/new_balans_test/{filename}{balance_index}.txt', "w") as output_file:
                    output_file.write(text)
                    balance_index += 1
        with open(f'OCR_balance_new.txt', "a") as output_file:
            output_file.write(filename + "\n")
        index += 1


collect_dataset()


















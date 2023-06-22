import PyPDF2
import os
import pandas as pd
import zipfile
from tqdm import tqdm
import nltk

def unzip_file(file_path, destination_path):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(destination_path)

def pdf_to_sentences(file_path):
    sentences = []
    
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        for page_num in tqdm(range(len(pdf_reader.pages) )):
            page =  pdf_reader.pages[page_num]
            text = page.extract_text()
            sentences.extend(nltk.sent_tokenize(text))
    
    return sentences

def convert_pdfs_to_dataframe(directory_path, output_path):
    pdf_files = [file for file in os.listdir(directory_path) if file.endswith(".pdf")]
    dataframes = []

    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        sentences = pdf_to_text(file_path)

        df = pd.DataFrame({"Sentence": sentences, "Source": pdf_file})
        dataframes.append(df)

    concatenated_df = pd.concat(dataframes, ignore_index=True)
    concatenated_df.to_csv(output_path, index=False)
    return concatenated_df


def pdf_pipeline(zip_file_path, zip_destination, directory_path, output_path):

    unzip_file(zip_file_path, zip_destination)
    convert_pdfs_to_dataframe(directory_path, output_path)

pdf_pipeline('/home/ubuntu/git/topic_modeling/data/BTSI.zip', '/home/ubuntu/git/topic_modeling/data', 
             '/home/ubuntu/git/topic_modeling/data/BTSI', '/home/ubuntu/git/topic_modeling/data/extracted_pdf.csv')
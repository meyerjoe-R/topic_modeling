import PyPDF2
import os
import pandas as pd
import zipfile
from tqdm import tqdm
import nltk
import fitz
import textract

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

def textract_extract(file_path):
    text = textract.process(file_path)
    return text

def pdf_to_paragraphs(file_path):
    paragraphs = []
    
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        for page_num in tqdm(range(len(pdf_reader.pages))):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            paragraphs.extend(text.split("\n\n"))  # Split paragraphs using double line breaks
    
    return paragraphs

def pdf_to_text(file_path):
    texts = []
    
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        for page_num in tqdm(range(len(pdf_reader.pages))):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            texts.append(text)
    
    return texts

def fitz_pdf_to_text(file_path):
    texts = []
    
    with fitz.open(file_path) as doc:
        for page in doc:
            text = page.get_text()
            texts.append(text)
    
    return texts

def convert_pdfs_to_dataframe(directory_path, output_path):
    pdf_files = [file for file in os.listdir(directory_path) if file.endswith(".pdf")]
    dataframes = []

    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        sentences = textract_extract(file_path)

        df = pd.DataFrame({"Sentence": sentences, "Source": pdf_file}, index = [0])
        dataframes.append(df)

    concatenated_df = pd.concat(dataframes, ignore_index=True)
    concatenated_df.to_csv(output_path, index=False)
    return concatenated_df


def pdf_pipeline(directory_path, output_path):
    # unzip_file(zip_file_path, zip_destination)
    convert_pdfs_to_dataframe(directory_path, output_path)

def extract_main():
    pdf_pipeline('/Users/josephmeyer/Desktop/git/topic_modeling/data/pdfs', '/Users/josephmeyer/Desktop/git/topic_modeling/data/output/extracted_pdf_textract_extract.csv')

def split_into_sentences(row):
    sentences = nltk.sent_tokenize(row['Sentence'])
    sources = [row['Source']] * len(sentences)
    return pd.DataFrame({'sentence': sentences, 'source': sources})

def sentence_tokenize_data(file_path, output_path):
    df = pd.read_csv(file_path)
    expanded_df = pd.concat(df.apply(split_into_sentences, axis=1).tolist(), ignore_index=True)
    expanded_df.to_csv(output_path)

def split_into_paragraphs(row):
    paragraphs = row['Sentence'].split('\n\n')
    sources = [row['Source']] * len(paragraphs)
    return pd.DataFrame({'sentence': paragraphs, 'source': sources})

def paragraph_tokenize_data(file_path, output_path):
    df = pd.read_csv(file_path)
    expanded_df = pd.concat(df.apply(split_into_paragraphs, axis=1).tolist(), ignore_index=True)
    expanded_df.to_csv(output_path)

# sentence_tokenize_data('/Users/josephmeyer/Desktop/git/topic_modeling/data/output/extracted_pdf_textract_extract.csv', 
#                        '/Users/josephmeyer/Desktop/git/topic_modeling/data/output/extracted_pdf_textract_extract_sentences.csv')


paragraph_tokenize_data('/Users/josephmeyer/Desktop/git/topic_modeling/data/output/extracted_pdf_textract_extract.csv', 
                       '/Users/josephmeyer/Desktop/git/topic_modeling/data/output/extracted_pdf_textract_extract_paragraphs.csv')
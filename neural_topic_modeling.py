import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

def write_to_excel(dfs:list, output_path):
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for id, d in tqdm(enumerate(dfs)):
            d.to_excel(writer, sheet_name=str(id))

def perform_bertopic(file_path, text_column, output_folder, device):
    # Load the data from the file
    df = pd.read_csv(file_path)
    docs = df[text_column].tolist()

    # Initialize a BERT-based sentence transformer
    model = SentenceTransformer('all-mpnet-base-v2', device = device)

    # Create BERTopic instance and fit the model
    topic_model = BERTopic(embedding_model=model, verbose = True)
    topics, probs = topic_model.fit_transform(docs)

    # Get the topic keywords
    topic_info = pd.DataFrame(topic_model.get_topic_info())

    # Create a dataframe with topic information
    df['topic_id'] = topics
    df['topic_probability'] = probs
    df = df.merge(topic_info, how = 'left', left_on = 'topic_id', right_on = 'Topic')
    df.sort_values(by = ['topic_id', 'topic_probability'], ascending= [True, False], inplace = True)

    # ###### Hierarchy ######

    # hierarchical_topics = topic_model.hierarchical_topics(docs)
    # topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

    # Return the dataframes
    dfs = [df, topic_info]
    write_to_excel(dfs, f"{output_folder}/output.xlsx")

    return df, topic_info

perform_bertopic('/Users/josephmeyer/Desktop/git/topic_modeling/data/output/extracted_pdf_textract_extract_sentences.csv',
                 'sentence', '/Users/josephmeyer/Desktop/git/topic_modeling/data/output/modeling', 'mps')
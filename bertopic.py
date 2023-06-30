import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import matplotlib.pyplot as plt
import networkx as nx

def perform_bertopic(file_path, text_column, output_folder):
    # Load the data from the file
    df = pd.read_csv(file_path)
    docs = df[text_column].tolist()

    # Initialize a BERT-based sentence transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create BERTopic instance and fit the model
    topic_model = BERTopic(embedding_model=model, verbose = True)
    topics, topic_freq = topic_model.fit_transform(docs)

    # Get the topic keywords
    topic_keywords = topic_model.get_topic_words()

    # Create a dataframe with topic information
    df['topic_id'] = topics
    df['topic_keywords'] = topic_keywords

    # Create a separate dataframe for topic frequency information
    topic_freq_df = pd.DataFrame({'Frequency': topic_freq,
                                  'TopicKeywords': topic_keywords,
                                  'TopicID': topics})

    # Print the generated topics
    for topic_id, topic_words in topic_model.get_topic_freq().items():
        print(f"Topic {topic_id}: {', '.join(topic_words)}")

    ###### Hierarchy ######

    hierarchical_topics = topic_model.hierarchical_topics(docs)
    topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

    df.to_csv(f"{output_folder}/results.csv")
    topic_freq_df.to_csv(f"{output_folder}/results_summary.csv")
    # Return the dataframes
    return df, topic_freq_df


perform_bertopic('/Users/josephmeyer/Desktop/git/topic_modeling/data/output/extracted_pdf_textract_extract_sentences.csv',
                 'sentence', '/Users/josephmeyer/Desktop/git/topic_modeling/data/output/modeling')
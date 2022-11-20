from .Politician2Vec import Politician2Vec
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from loguru import logger

import re
from collections import defaultdict
import string
import nltk
for dependency in ['punkt', 'wordnet', 'omw-1.4', 'stopwords', 'averaged_perceptron_tagger']:
    nltk.download(dependency)

def preproc_docs(text):
    #Lowercasing words
    text = text.lower()
    
    #Removing HTML tag
    text = re.sub(r'&amp', '', text)

    #Replace "&" with "and"
    text = re.sub(r'&','and', text)
    
    #Removing mentions 
    text = re.sub(r'@\w+ ', '', text)
    
    #Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation.replace('-',''))) #Taking hyphens out of punctuation to remove
    text = re.sub(r' - ','', text) #removing dash lines bounded by whitespace (and therefore not part of a word)
    text = re.sub(r'…', '', text)
    text = re.sub(r'[â€˜â€™â€œâ€â€”]','',text) #removing punctuation that is not captured by string.punctuation
    
    #Removing 'RT' and 'via'
    #text = re.sub(r'(^rt|^via)((?:\\b\\W*@\\w+)+): ', '', text)
    text = re.sub(r'^rt', '', text) 
    
    # Removing mentions
    text = re.sub(r'(@[A-zÃ¦Ã¸Ã¥0-9]{1,15})', '', text)
    
    #Removing odd special characters
    text = re.sub(r"[â”»â”ƒâ”â”³â”“â”â”›â”—]","", text)
    text = re.sub(r"\u202F|\u2069|\u200d|\u2066|\U0001fa86","", text)
    
    #Removing URLs
    text = re.sub(r'http\S+', '', text)
    #text = re.sub(r'https:\/\/t\.co\/[a-zA-Z0-9\-\.]+', '', text) # Shortened "https://t.co/..." Twitter URLs
    #text = re.sub(r'https:\/\/[A-z0-9?\.\/-_=!]+', '', text) # Remove other URLs

    #Removing numbers
    text = re.sub(r'[0-9.]','', text)

    # Removing idiosynchratic characters in our data
    text = re.sub(r'-\n|\n-|\na-|\nb-|â€“|Â«|--|’', '', text)
    text = re.sub(r'- ', ' ', text)

    #Removing separators and superfluous whitespace
    text = text.strip()
    text = re.sub(r' +',' ',text)

    #Tokenizing
    tokenizer = nltk.TweetTokenizer() 
    tokens = tokenizer.tokenize(text)

    #Removing stopwords
    stop_words_list = nltk.corpus.stopwords.words('danish')
    tokens = [i for i in tokens if i not in stop_words_list]
    
    # Removing query-specific words
    stop_words_list = []
    #stop_words_list.extend(['mink'])
    tokens = [i for i in tokens if i not in stop_words_list]

    return tokens

# TODO:
# - IPython progress bars.
# - More extensive error handling.
# – Rewrtie to topics_to_keep
# - Write logging funcitonality for entire package...
# - Move to politician2vec package!

#print('Thank you for using the custom Radar Tool and its utilities! Report issues to mathias.b@adviceagency.com')

def load_politician2vec_from_txt(model_path: 'str') -> 'politician2vecmodel, doc2vec_model':
    '''
    Loads a Politician2Vec embedding model from .txt file.
    
    Returns both the Politician2Vec model and the Doc2Vec
    model which makes up a subset thereof.
    
    -----
    model_path (str): filepath to the Politician2Vec file
    '''
    
    # Load entire Politician2Vec model
    print('Loading Politician2Vec model...')
    politician2vec_model = Politician2Vec.load(model_path)
    
    # Retrieve the Doc2Vec embedding
    print('Retrieving document embedding...')
    doc2vec_model = politician2vec_model.model
    
    print('All done!')

    return politician2vec_model, doc2vec_model

def doc2vec2tensor(
    doc2vec_model,
    temp_w2v_path,
    tsv_prefix = 'TENSOR_v1_',
    output_docvecs = True,
    output_wordvecs = True
):
    '''
    Save a temporary word2vec-formatted version of
    the document embedding, then apply a gensim
    script to transform this file into two separate
    .tsv files comaptible with Tensorflow's TensorBoard:
    a vector file and a metadata file.
    
    -----
    doc2vec_model: document embedding returned by
        load_politician2vec_from_txt()
    
    temp_w2v_path (str): filename to use for temporary
        word2vec-formatted embedding file
    
    tsv_prefix (str): prefix for the .tsv vector and
        metadata files -- default for dev version is
        "TENSOR_v1_"
    
    output_docvecs (bool): whether or not to export
        document vectors -- default is True
    
    output_wordvecs (bool): whether or not to export
        word vectors -- default is True
    '''
    
    if output_docvecs and output_wordvecs:
        print('You have elected to extract both word and document vectors.\nPlease note that this may cause unintended behaviour in TensorBoard visualisations.')
    
    elif output_wordvecs and not output_docvecs:
        print('You have elected to extract only word vectors.\nSince metadata is also extracted for the entire vocab, please note that no\nfurther preprocessing will be strictly necessary to facilitate TensorBoard visualisation.')
    
    elif output_docvecs and not output_wordvecs:
        print('You have elected to extract only document vectors.\nPlease note that further preprocessing -- such as filtering based on topics of\ninterest -- may be desired in order to facilitate TensorBoard visualisation.\nPlease see get_doc_topic_df(), vector_subset2tensor_without_words(), and\nmetadata2tensor()')
    
    else:
        raise ValueError('Both output_ args are False. Please specify whether or not to extract word vectors, document vectors, or both.')
    
    # Save temporary version of the embedding, formatted as w2v
    print('\nSaving temp w2v file and converting to tensor. This may take a while...')
    
    doc2vec_model.save_word2vec_format(
        temp_w2v_path,
        doctag_vec = output_docvecs,
        word_vec = output_wordvecs
    )
    
    # Export tensor version to two separate files using gensim
    os.system(f'python -m gensim.scripts.word2vec2tensor -i {temp_w2v_path} -o {tsv_prefix}')
    
def get_topics_convert_other(politician2vec_model, no_substantive_topics):
    '''
    A list of topics with smaller topics above a certain
    threshold are lumped together as 'Other'.
    
    -----
    politician2vec_model: Politician2Vec model in question
    
    no_substantive_topics (int): number of topics that
        are actually of interest -- topics above this
        number will be lumped together
    '''
    
    # Get the index of the last substantive topic
    max_top_idx = no_substantive_topics - 1
    
    # Get list of topics with topics of index higher than
    # max_top_idx lumped into one group of max_top_idx + 1
    topics_list = np.where(politician2vec_model.doc_top > max_top_idx, max_top_idx+1, politician2vec_model.doc_top)
    
    unique, counts = np.unique(topics_list, return_counts=True)
    
    print(f'Topic sizes before filtering (topic {no_substantive_topics} is "Other"):\n')
    print(np.asarray((unique, counts)).T)
    
    return(topics_list)

def join_topics_to_df(politician2vec_model, top_dict_to_retrieve, orig_df, to_excel_file = None):
    '''
    TODO...
    '''
    doc_dict = dict()
    topic_sizes, topic_nums = politician2vec_model.get_topic_sizes()

    for topic in tqdm(top_dict_to_retrieve.keys(), desc = 'Retrieving topics'):

        documents, document_scores, document_ids = politician2vec_model.search_documents_by_topic(
                topic_num=topic,
                num_docs=topic_sizes[topic]
                )

        doc_dict[topic] = documents

    df_doc_top = pd.concat(pd.DataFrame({'top':k, 'doc':v}) for k, v in doc_dict.items())
    df_out = df_doc_top.merge(
        orig_df,
        left_on = 'doc', right_on = 'fullText', how = 'inner')
    
    for top_idx, top_label in tqdm(top_dict_to_retrieve.items(), desc = 'Writing topic labels'):
        df_out.loc[df_out['top'] == top_idx, ['topic_label']] = top_label
    
    df_out = df_out[[
            'top', 'topic_label', 'doc', 'language', 'date',
            'contentSource', 'author', 'fullname', 'gender',
            'impact', 'sentiment', 'url'
        ]]
        
    if type(to_excel_file) == str:
        writer = pd.ExcelWriter(
            f'{to_excel_file}.xlsx',
            engine = 'xlsxwriter',
            options = {
                'strings_to_urls': False,
                'index': False
            }
        )
        logger.info('Writing to Excel...')
        df_out.to_excel(writer, index = False)
        writer.save()
        logger.info('All done!')

    return(df_out)

def get_doc_topic_df(politician2vec_model, no_substantive_topics = 10, snippets=False, topics_to_remove=None):
    '''
    Extract topic indeces and document ids from the Politician2Vec
    model. Return a DataFrame of these two lists, filtered
    to exclude the "Other" topic as defined by the
    no_substantive_topics argument and executed by the
    get_topics_convert_other function.
    
    It is possible to also include text snippets by
    specifying the "snippets" argument.
    
    -----
    politician2vec_model: Politician2Vec model in question
    
    no_substantive_topics (int): number of topics that
        are actually of interest -- topics above this
        threshold will be lumped together and filtered
        out -- default is 10
        
    snippets (bool): whether or not to include text
        snippets -- default is False
    
    topics_to_remove (list): other topics to remove
        semi-manually (e.g. if topics of interest are
        not consecutive index-wise)
    '''
    
    topics_list = get_topics_convert_other(politician2vec_model, no_substantive_topics)
    doc_ids_list = list(politician2vec_model.document_ids)
    
    topic_df = pd.DataFrame(zip(doc_ids_list, topics_list), columns = ['doc', 'top'])
    
    if type(topics_to_remove) == list:
        topic_df.loc[topic_df['top'].isin(topics_to_remove), ['top']] = no_substantive_topics
    
    topic_df = topic_df.loc[topic_df['top'] < no_substantive_topics]
    
    if snippets:
        snippets = (
            pd.DataFrame(politician2vec_model.documents)
                .reset_index()
                .rename(columns={'index':'snippet_no', 0:'snippet'})
        )

        snippets_to_retrieve = snippets.loc[snippets['snippet_no'].isin(topic_df['doc'])]

        topic_df = topic_df.join(snippets_to_retrieve, how = 'left').drop(columns=['snippet_no'])
        topic_df['snippet'] = topic_df['snippet'].str.replace('\n|\t', ' ', regex = True)
    
    return(topic_df)

def metadata2tensor(topic_df, metadata_path, label_list, word_vecs_included = False, vocab = None):
    '''
    Takes as input an appropriately filtered pandas
    DataFrame of topic ids and document indeces
    (possibly also text snippets).
    
    This lookup table is used to write metadata
    to an existing TensorBoard-compatible .TSV file.
    
    Metadata is either doc id or doc id + text snippet.
    
    If writing metadata for a vector file containing
    both word and document vectors, please sepcify
    word_vecs_included = True and supply the full
    vocab.
    
    -----
    label_list input changed to dict to avoid index mismatch
    due to semi-manual filtering in get_topic_df!
    '''
    
    with open(metadata_path,'w') as w:
        w.write('doc\ttopic\n')
    
        if word_vecs_included:

            for word in vocab:
                w.write(f'{word}\tword\n')

        if topic_df.shape[1] == 3:

            for i,j,k in zip(topic_df['doc'], topic_df['top'], topic_df['snippet']):
                w.write("%s_%s\t%s\n" % (i,k,label_list[j]))

        elif topic_df.shape[1] == 2:

            for i,j in zip(topic_df['doc'], topic_df['top']):
                w.write("%s_%s\t%s\n" % (i,label_list[j]))

        else:

            raise Exception('Error: no metadata column(s) to write!')
            
def vector_subset2tensor_without_words(topic_df, orig_vec_path, out_path):
    '''
    Takes as input an appropriately filtered pandas
    DataFrame of topic ids and document indeces
    (possibly also text snippets).
    
    This lookup table is used to extract document
    vectors from an existing TensorBoard-compatible
    .TSV file, writing this usually smaller subset
    of docvecs to a new, equivalent file.
    
    -----
    topic_df (pandas DataFrame): "lookup table"-style
        DataFrame containing a the document indeces
        and topic ids for a subset of topics, based
        on topics of interest from the politician2vec model
    
    orig_vec_path (str): original filepath of
        TensorBoard-compatible .TSV docvec file
    
    orig_vec_path (str): desired output path of
        new TensorBoard-compatible .TSV docvec file
    '''
    
    with open(orig_vec_path,'r') as r:
        lines = r.readlines()
    
    with open(out_path,'w') as w:
    
        for line_no in range(0, len(lines)): #+1?

            if line_no in topic_df['doc']:
                w.write(lines[line_no])
                
def restrict_w2v_to_topic(doc2vec_model, selected_topics, topic_words, terms_of_interest):
    '''
    Takes the doc2vec component of the politician2vec model,
    restricting the vocabulary to word vectors in the
    top 50 words within each specified topic. This allows
    for unobscured exploration of a smaller subset of
    word vectors, restricted by topic.
    
    May be useful when some topics are distributed in highly
    dense neighbourhoods placed far from others.
    
    A similar goal can be achieved by subsetting vectors
    before visualising them in TensorBoard. However, this
    function retains core gensim funcitonality such as
    analogy arithmetic (king-man+woman=queen) which
    is not natively available in TensorBoard. Highly useful
    when producing input for plot_utils.plot_venn_words().
    '''
    
    print('Please note that this function currently modifies the existing model.\nPlease use load_politician2vec_from_txt() to load original model again.')
    
    selected_words = []

    for topic in selected_topics:
        top_words = list(topic_words[topic])
        selected_words.append(top_words)
        
    restricted_words = [inner for outer in selected_words for inner in outer] + terms_of_interest
    restricted_words = list(set(restricted_words))
    
    restricted_words_indeces = []

    for word in restricted_words:
        word_idx = doc2vec_model.wv.key_to_index[word]
        restricted_words_indeces.append(word_idx)
        
    new_vectors = []

    for idx in restricted_words_indeces:
        vector = doc2vec_model.wv[idx]
        new_vectors.append(vector)
        
    restricted_words_indeces = [i for i in range(0,len(restricted_words))]
    
    new_key_to_index = doc2vec_model.wv.key_to_index = dict(zip(restricted_words, restricted_words_indeces))
    new_index_to_key = restricted_words
    
    doc2vec_model.wv.index_to_key = new_index_to_key
    doc2vec_model.wv.key_to_index = new_key_to_index
    doc2vec_model.wv.vectors = np.array(new_vectors)
    
    return doc2vec_model
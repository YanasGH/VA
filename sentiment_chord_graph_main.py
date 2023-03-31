from collections import Counter
import plotly.graph_objects as go
from nltk.sentiment import SentimentIntensityAnalyzer
from itertools import product
import holoviews as hv
from holoviews import opts, dim
from bokeh.models import HoverTool
from bokeh.themes import Theme
import json
from datetime import datetime as dt
import numpy as np
import copy
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print('----- Creating and saving 10 graphs -----')

########## IMPORT DATA AND PERFORM SENTIMENT ANALYSIS ##########
sia = SentimentIntensityAnalyzer()

email_df = pd.read_csv('email_headers.csv', encoding='cp1252')
sentiment = email_df['Subject'].apply(lambda x: sia.polarity_scores(x))
email_df[['neg', 'neu', 'pos', 'compound']] = pd.json_normalize(sentiment)
employee_recs = pd.read_excel("EmployeeRecords.xlsx")

email_df['Day'] = None
email_df['DepTo'] = None
email_df['DepFrom'] = None
email_df['sentiment'] = None
emp_dict = dict(zip(employee_recs.EmailAddress, employee_recs.CurrentEmploymentType))
for i in range(len(email_df)):
    email_df['Day'][i] = str(dt.strptime(email_df['Date'][i], "%m/%d/%Y %H:%M").day)
    email_df['DepFrom'][i]=emp_dict.get(email_df['From'][i])
    
    if email_df['compound'][i]==0:
        email_df['sentiment'][i]='neu'
    elif email_df['compound'][i]>0:
        email_df['sentiment'][i]='pos'
    else:
        email_df['sentiment'][i]='neg'
    
    to_list = []
    for to in email_df['To'][i].split(', '):
        to_list.append(emp_dict.get(to))
    email_df['DepTo'][i]=to_list

def edge_node(att, toatt, fromatt):
    """
    Computes the edges (from-to email correspondences) and nodes (employees) that are used to create the chord diagram.
    att: either "EmailAddress" if you wish to get the from-to email correspondences between individual employees or 
         "CurrentEmploymentType" if you wish to get the from-to email correspondences between departments 
    toatt: name of the column that contains the recipients, either "To" if att is "EmailAddress" or "DepTo" if att is
           "CurrentEmploymentType"
    fromatt: name of the column that contains the sender, either "From" if att is "EmailAddress" or "DepFrom" if att is
             "CurrentEmploymentType"
    Returns an adjacency matrix with senders as rows, recipients as column and number of emails sent as values, 
    a dataframe edge_temp containing the number of times email was exchanged, the source (From/DepFrom) and the target
    (To/DepTo), a dataframe node_temp containing the employee email and employee name
    """ 
    
    #compute the edges
    edge_temp = []
    for day in email_df['Day'].unique():
        email_df_day = email_df[email_df['Day']==day]
        if att=='EmailAddress':
            adj_matrix = pd.DataFrame(employee_recs[att]).set_index(att)
        else:
            adj_matrix = pd.DataFrame(employee_recs[att].unique(), columns = [att]).set_index(att)

        for i in adj_matrix.index: #set up the matrix: rows correspond to the "From" person, columns to the "To" person
            adj_matrix[f'{i}']= 0 


        for i in adj_matrix.index: # fill in the matrix
            from_df = email_df_day[email_df_day[fromatt]==i]
            to_total = []

            for j in from_df[toatt]: 
                try:
                    to_total += j.split(', ')
                except:
                    to_total += j
            to_total = Counter(to_total)

            for c in adj_matrix.columns:
                if (c != i) and (to_total.get(c) != None):
                    adj_matrix[c][i] = to_total.get(c) # the syntax here is reverse: first is the "To" person, then it is the "From" person
        for k in adj_matrix.index:
            for j in adj_matrix.columns:
                if k!=j and adj_matrix[j][k]!=0:
                    edge_temp.append([adj_matrix[j][k],k, j, day])
    
    edge_temp = pd.DataFrame(edge_temp, columns = ['TransactionAmt','Source','Target', 'Date'])
    
    #compute the nodes
    node_temp = []
    if att=='EmailAddress':
        for i in adj_matrix.columns:
            node_temp.append([i, i.split('@')[0].split('.')[0] + ' ' + i.split('@')[0].split('.')[1]])
        node_temp = pd.DataFrame(node_temp, columns = ['Account','CustomerName'])
    else:
        for i in adj_matrix.columns:
            node_temp.append([i, i])
        node_temp = pd.DataFrame(node_temp, columns = ['Account0', 'Account'])
    return adj_matrix, edge_temp, node_temp

_, edge2, node2 = edge_node('EmailAddress', 'To', 'From')

########## put the data in the correct "format" so that it can be used for the chord graph ###########
name_email_dict = dict(zip(node2['Account'].to_list(), node2['CustomerName'].to_list()))
emp_dep_dict = dict(zip(employee_recs['EmailAddress'].to_list(), employee_recs['CurrentEmploymentType'].to_list()))

df_chord_nodes = employee_recs[['EmailAddress']]
df_chord_nodes['Department'] = None
df_chord_nodes['Name'] = None
for i in range(len(df_chord_nodes)):
    df_chord_nodes['Department'][i] = emp_dep_dict.get(df_chord_nodes['EmailAddress'][i])
    df_chord_nodes['Name'][i] = name_email_dict.get(df_chord_nodes['EmailAddress'][i])   
dict_nodes_index = dict(zip(employee_recs['EmailAddress'], df_chord_nodes.index))

chord_sentiment_pre = pd.DataFrame(product(employee_recs['EmailAddress'], employee_recs['EmailAddress'], ['neg', 'pos', 'neu']), columns = ['source', 'target', 'sentiment'])#.head(4)
chord_sentiment_pre = chord_sentiment_pre[chord_sentiment_pre['source']!=chord_sentiment_pre['target']].reset_index(drop=True)
chord_sentiment_pre['value'] = np.zeros(len(chord_sentiment_pre))
chord_sentiment_pre['dep'] = None

def sentiment_chord_graph(DAY, chord_sentiment_pre):
    """
    Creates a chord diagram of the email correspondences, such that each employee node is colored wrt the sentiment of the email tehy sent.
    DAY: the day for which the chord diagram will be created.
    Returns the created chord diagram.
    """    
    chord_sentiment = copy.deepcopy(chord_sentiment_pre)
    for i in range(len(chord_sentiment)):
        email_df_days = email_df[email_df['Day']==str(DAY)]
        email_df_filtered = email_df_days[email_df_days['From']==chord_sentiment['source'][i]]

        for index in email_df_filtered.index:
            if chord_sentiment['target'][i] in email_df_filtered['To'][index]:
                if email_df_filtered['sentiment'][index]==chord_sentiment['sentiment'][i]:
                    chord_sentiment['value'][i]+=1

        chord_sentiment['dep'][i] = emp_dep_dict[chord_sentiment['source'][i]]
        chord_sentiment['source'][i] = dict_nodes_index[chord_sentiment['source'][i]]
        chord_sentiment['target'][i] = dict_nodes_index[chord_sentiment['target'][i]]

    chord_sentiment['value'] = chord_sentiment['value'].astype(int)
    chord_sentiment = chord_sentiment[chord_sentiment['value']!=0]

    df_count_sentiment = pd.DataFrame(chord_sentiment.groupby(by=['dep', 'sentiment'])['value'].sum()).reset_index().sort_values('value', ascending=False).drop_duplicates('dep')
    dict_count_sentiment = dict(zip(df_count_sentiment['dep'].tolist(), df_count_sentiment['sentiment'].tolist()))

    df_chord_nodes['sentimcolor'] = None
    for i in range(len(df_chord_nodes)):
        df_chord_nodes['sentimcolor'][i] = dict_count_sentiment[df_chord_nodes['Department'][i]]

    chord_sentiment = chord_sentiment[['source','target','sentiment','value']]
    dict_sentim_colors = {'pos': '#379475', 'neg': '#B51E17', 'neu': '#BDC4C5'}


    cmap_custom_e = []
    for s in chord_sentiment['sentiment'].unique():
        cmap_custom_e.append(dict_sentim_colors[s])

    cmap_custom_n = []
    for s in df_chord_nodes['sentimcolor'].unique():
        cmap_custom_n.append(dict_sentim_colors[s])

    tooltips = [('Department', '@Department'), ('Name', '@Name')]
    hover = HoverTool(tooltips=tooltips)
    hv.extension('bokeh')
    hv.output(size=200)

    df_chord_nodes_filtered = hv.Dataset(df_chord_nodes, 'index')

    chord = hv.Chord((chord_sentiment, df_chord_nodes_filtered))
    chord.opts(opts.Chord(inspection_policy='nodes', cmap=cmap_custom_n, edge_cmap= cmap_custom_e,
                                 edge_color=dim('sentiment').str(), labels='Name', node_color=dim('sentimcolor').str(),
                                 symmetric=True, bgcolor = '#26232C', label_text_color='#FEFEFE', tools = [hover]))
    return chord

c = 1
#create a chord diagram for each day in the dataset
for DAY in [6,7,8,9,10,13,14,15,16,17]:
    chord_s = sentiment_chord_graph(DAY, chord_sentiment_pre)
    renderer = hv.renderer('bokeh')
    renderer.theme = Theme('assets/theme_chord.json')
    renderer.save(chord_s, f'assets/graph_chord_sentiment_{DAY}')
    print(f'----- {10-c} graphs left -----')
    c+=1
from collections import Counter
import plotly.graph_objects as go
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

########## IMPORT DATA ##########
email_df = pd.read_csv('email_headers.csv', encoding='cp1252')
employee_recs = pd.read_excel("EmployeeRecords.xlsx")
cmap_custom = ['#5b4dd6', '#c933bc', '#ffc60a', '#ff5960', '#ff9232', '#ff2b90']

########## PREPROCESS DATA ##########
email_df['Day'] = None
email_df['DepTo'] = None
email_df['DepFrom'] = None
emp_dict = dict(zip(employee_recs.EmailAddress, employee_recs.CurrentEmploymentType))
for i in range(len(email_df)):
    email_df['Day'][i] = str(dt.strptime(email_df['Date'][i], "%m/%d/%Y %H:%M").day)
    email_df['DepFrom'][i]=emp_dict.get(email_df['From'][i])
    
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

_, df_chord_edges, _ = edge_node('EmailAddress', 'To', 'From')
df_chord_edges = df_chord_edges.rename(columns={"TransactionAmt": "value"})
df_chord_edges['source'] = None
df_chord_edges['target'] = None
df_chord_edges['Gsource'] = None
df_chord_edges['Gtarget'] = None

for i in range(len(df_chord_edges)):
    df_chord_edges['source'][i] = int(df_chord_nodes[df_chord_nodes['EmailAddress']==df_chord_edges['Source'][i]].index[0])
    df_chord_edges['target'][i] = int(df_chord_nodes[df_chord_nodes['EmailAddress']==df_chord_edges['Target'][i]].index[0])
    df_chord_edges['Gsource'][i] = emp_dep_dict.get(df_chord_edges['Source'][i])
    df_chord_edges['Gtarget'][i] = emp_dep_dict.get(df_chord_edges['Target'][i])

df_chord_nodes = df_chord_nodes.drop(columns=['EmailAddress'])
df_chord_edges = df_chord_edges.drop(columns = ['Source', 'Target'])
df_chord_edges = df_chord_edges[['source', 'target', 'value', 'Gsource','Gtarget', 'Date']]

def chord_graph(DAY):
    """
    Creates a chord diagram of the email correspondences, such that each employee node is colored wrt their department.
    DAY: the day for which the chord diagram will be created.
    Returns the created chord diagram.
    """
    df_chord_edges_filtered = df_chord_edges[df_chord_edges['Date']==str(DAY)].drop(columns = ['Date']) #using the df outside the function
    tooltips = [('Department', '@Department'), ('Name', '@Name')]
    hover = HoverTool(tooltips=tooltips)
    hv.extension('bokeh')
    hv.output(size=200)

    df_chord_nodes_filtered = hv.Dataset(df_chord_nodes, 'index')

    chord = hv.Chord((df_chord_edges_filtered, df_chord_nodes_filtered))
    chord.opts(opts.Chord(inspection_policy='nodes', cmap=cmap_custom, edge_cmap=cmap_custom,
                                 edge_color=dim('Gsource').str(), labels='Name', node_color=dim('Department').str(),
                                 symmetric=True, bgcolor = '#26232C', label_text_color='#FEFEFE', tools = [hover]))
    return chord

c = 1
#create a chord diagram for each day in the dataset
for DAY in [6,7,8,9,10,13,14,15,16,17]:
    chord_g = chord_graph(DAY)
    renderer = hv.renderer('bokeh')
    renderer.theme = Theme('assets/theme_chord.json')
    renderer.save(chord_g, f'assets/graph_chord_{DAY}')
    print(f'----- {10-c} graphs left -----')
    c+=1

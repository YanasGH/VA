import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
from apps import home
import numpy as np
import datetime
from datetime import datetime as dt
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import dedent as d
import copy
from collections import Counter
import networkx as nx 
import json
from colour import Color
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import holoviews as hv
from holoviews import opts, dim
from bokeh.models import HoverTool
from bokeh.themes import Theme
from nltk.sentiment import SentimentIntensityAnalyzer #joblib-1.2.0 nltk-3.8.1 regex-2023.3.23
from wordcloud import WordCloud, STOPWORDS #wordcloud-1.8.2.2
import warnings
warnings.filterwarnings("ignore")

############################## Colors ##############################
# Main background                   #26232C
# Secondary background / sidebar    #cbd3dd
# Color buttons sidebar
# Text mainbackground               white '#FEFEFE'
# Text sidebar                      white '#FEFEFE'
# cmap (rainbow ish)                cmap_custom = ['#5b4dd6', '#c933bc', '#ffc60a', '#ff5960', '#ff9232', '#ff2b90']
# color_discrete_sequence =        ['#B51E17', '#BDC4C5', '#379475']) (red, green, grey)
# discrete_colors = sample_colorscale('viridis', [0.6, 1.0, 0.4, 0.2, 0.8, 0])

########## CONFIGURE APP ##########
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, 'https://use.fontawesome.com/releases/v6.1.1/css/all.css'], meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])
server = app.server
app.config.suppress_callback_exceptions = True

########## DEFINE CONTENT STYLE ##########
CONTENT_STYLE = {
    "margin-left": "8rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

########## DEFINE SIDEABAR AND CONTENT; FINISH THE CONFIGURATION OF THE APP ##########
sidebar = html.Div(
    [
        html.Div(
            [
                html.H2("Explore", style={"color": "white"}),
            ],
            className="sidebar-header",
        ),
        html.Hr(),
        dbc.Nav(children=[
                dbc.NavLink(
                    [
                        html.I(className="fas fa-home me-2"), 
                        html.Span("Home", style={"color": "#FEFEFE"})
                    ],
                    href="/",
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fab fa-connectdevelop"),
                        html.Span("Email Exchange", style={"color": "#FEFEFE"}),
                    ],
                    href=  "/page-1", 
                    active="exact",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-envelope-open-text me-2"),
                        html.Span("Text analysis", style={"color": "#FEFEFE"}),
                    ],
                    href="/page-2", 
                    active="exact",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar", 
)

# define content
content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

# complete app layout
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

# define sentiment analuzer
sia = SentimentIntensityAnalyzer()

########## IMPORT DATA AND PREPROCESS ##########
employee_recs = pd.read_excel("EmployeeRecords.xlsx")
email_df = pd.read_csv('email_headers.csv', encoding='cp1252')
classification = pd.read_csv('email_classification.csv', encoding='cp1252')
sentiment = email_df['Subject'].apply(lambda x: sia.polarity_scores(x))
email_df[['neg', 'neu', 'pos', 'compound']] = pd.json_normalize(sentiment)

cmap_custom = ['#5b4dd6', '#c933bc', '#ffc60a', '#ff5960', '#ff9232', '#ff2b90']
# colors for plot_freq_words
discrete_colors = sample_colorscale('viridis', [0.6, 1.0, 0.4, 0.2, 0.8, 0])  

# get all unique email subject that have a response
all_re = []
for i in range(len(email_df)):
    if 'RE' in email_df['Subject'][i]:
        all_re.append(email_df['Subject'][i][4:])
all_re = np.unique(all_re)


def get_full_conv(subject: str):
    """
    subject: the subject of the email
    returns the full conversation
    """
    
    root = email_df[email_df['Subject'].isin([subject])]
    replies = email_df[email_df['Subject'].isin(['RE: ' + subject])]
    result = pd.concat([root, replies]).sort_values('Date')
    
    return result


# get the days from the date, and department for each employee
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

# data preprocessing to dotplot
email_df['time'] = pd.to_datetime(email_df['Date'])
email_df['from'] = email_df.From.str[:-19]#.replace(".", " ")
email_df['from'] = email_df['from'].str.replace('.', ' ')
email_df['number of recipients'] = [len(list) for list in email_df['To'].str.replace(',', '').str.split()]

# adding classification to email data frame
email_df['Subject without re'] = email_df['Subject'].apply(lambda x: x[4:] if x.startswith('Re: ') or x.startswith('RE: ') else x ) # join classification on cleaned string
# manually cleaning classification, error is caused by encoding/decoding error of "'"
classification.replace('Hey, Iâ€™m going home sick.', 'Hey, I’m going home sick.', inplace=True)
classification.replace('Iâ€™m in! -  post a list','I’m in! -  post a list', inplace=True )
classification.replace('Whoâ€™s tracking the office pool', 'Who’s tracking the office pool', inplace=True)
classification['Subject without re'] = classification['Subject']
email_df = email_df.merge(classification[['Subject without re','class']], on='Subject without re', how = 'outer')

# network functions
def edge_node(att, toatt, fromatt, day_search=None):
    """
    Computes the edges (from-to email correspondences) and nodes (employees) that are used to create the chord diagram.
    
    - att: either "EmailAddress" if you wish to get the from-to email correspondences between individual employees or 
         "CurrentEmploymentType" if you wish to get the from-to email correspondences between departments 
    - toatt: name of the column that contains the recipients, either "To" if att is "EmailAddress" or "DepTo" if att is
           "CurrentEmploymentType"
    - fromatt: name of the column that contains the sender, either "From" if att is "EmailAddress" or "DepFrom" if att is
             "CurrentEmploymentType"
    - day_search: the day for which the edges and nodes are to be computed, if None the edges and nodes are computed over all days
    
    Returns an adjacency matrix with senders as rows, recipients as column and number of emails sent as values, 
    a dataframe edge_temp containing the number of times email was exchanged, the source (From/DepFrom) and the target
    (To/DepTo), a dataframe node_temp containing the employee email and employee name
    """ 
    
    #compute the edges
    edge_temp = []
    if day_search==None:
        iter_days = email_df['Day'].unique()
    else:
        iter_days = [str(day_search)]
        
    for day in iter_days:
        email_df_day = email_df[email_df['Day']==day]
        if att=='EmailAddress':
            adj_matrix = pd.DataFrame(employee_recs[att]).set_index(att)
        else:
            adj_matrix = pd.DataFrame(['Administration', 'Information Technology', 'Executive', 'Facilities', 'Engineering', 'Security'], columns = [att]).set_index(att)

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

############################ Set variables ####################
DAY = 6 # starting value day page 1
analize_by = 'Department' # starting value color page 1 (2 graphs)
color_dot = 'Department' # starting value color page 1 (dotplot)
# color mapping of page 1 
c_map_dep =  {'Administration':'#5b4dd6', 'Engineering':'#c933bc', 'Executive': '#ffc60a','Facilities': '#ff5960','Information Technology': '#ff9232','Security': '#ff2b90'}
c_map_sent = {'pos': '#379475','neu': '#BDC4C5', 'neg': '#B51E17'}
c_map_class = dict(zip(['Work', 'Change/schedule', 'Social', 'Weird', 'Other', 'Undefined','Spam'], ['#fb6161','#ff8e02','#ffda00' , '#33d592', '#5bccff', '#5475ff', '#c435ff' ]))
# setting order of legend for multiple columns
category_order = {'class': ['Work', 'Change/schedule', 'Social', 'Weird', 'Other', 'Undefined', 'Spam'],
                       'DepFrom' : ['Administration', 'Engineering', 'Executive', 'Facilities','Information Technology', 'Security'],
                       'sentiment' : ['pos', 'neu', 'neg'] }
################################# Graph functions page 1 ###########################
def chord_graph(DAY, analize_by):
    """
    Loads the already created chord diagrams.
    
    - DAY: the selected day the user is interested in
    - analyze_by: either "Department" or "Sentiment", depending on what type of analysis the user wants to perform
    """
    if analize_by == 'Department':
        return f'assets/graph_chord_{DAY}.html'
    else:
        return f'assets/graph_chord_sentiment_{DAY}.html'

def bar_deps(DAY, analize_by):
    """
    Creates and returned a stacked bar chart displaying the number of sent email per department.
    
    - DAY: the selected day the user is interested in
    - analyze_by: either "Department" or "Sentiment", depending on what type of analysis the user wants to perform
    """
    
    if analize_by == 'Department':
        adj_matrix, _, _ = edge_node('CurrentEmploymentType', 'DepTo', 'DepFrom', DAY)
        adj_matrix = adj_matrix.rename(columns={"Information Technology": "IT"})
        adj_matrix = adj_matrix.reset_index()
        adj_matrix['CurrentEmploymentType'][1] = "IT"

        fig = px.bar(adj_matrix, x="CurrentEmploymentType", y=['Administration', 'Engineering', 'Executive', 'Facilities', 'IT', 'Security'], color_discrete_sequence =cmap_custom,
                         labels={
                                 "value": "Number of times",
                                 "CurrentEmploymentType": "From department",
                                 "variable": "To department"
                             })
    else:
        dfbs = pd.DataFrame(np.zeros([18, 3]), columns = ['Department', 'Sentiment', 'value'])
        dfbs['Department'] = 3*['Administration']+3*['IT']+3*['Executive']+3*['Facilities']+3*['Engineering']+3*['Security']
        dfbs['Sentiment'] = 6*['neg', 'neu', 'pos']
        
        dfday = email_df[email_df['Day']==str(DAY)]
        sents = []
        
        for d in dfbs['Department'].unique():
            c = Counter(dfday[dfday['DepFrom']==d].explode('DepTo')['sentiment'])
            for s in ['neg', 'neu', 'pos']:
                if s in c.keys():
                    sents.append(c[s])
                else:
                    sents.append(0)
        dfbs['value'] = sents
        
        fig = px.bar(dfbs, x="Department", y="value", color="Sentiment",
             color_discrete_sequence = ['#B51E17', '#BDC4C5', '#379475'])
        

    fig.update_layout(paper_bgcolor='#26232C',plot_bgcolor='#26232C', font_color = "#FEFEFE",font_size=13, yaxis=dict(gridcolor='#5c5f63'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),font=dict(size=10))
    return fig

def dotplotgraph(color_dot = 'Department', DAY='6', dropdown = [], category = 'All'):
    """
    The function takes an array of input values from callbacks. Which influence the graph and ploting data of the graph. 
    Commenting uses (variable)names as shown in the dashboard and from code/data.

    Input: callback values
        - color_dot         from 'Analyze by' button
        - DAY              from chosen day time range
        - dropdown             from 'Multi-Select Dropdown' containing strings of the selected values
                            options: (show workhours, filter RE, corresponding day, show number of recipients, show categories)
        - category          from 'Select category' dropdown
    Some variables used in function:
        color_dot           input from 'Analyze by' dropdown or 'Class', used in title of legend
        color_dot_name      name of column in dataframe
        c_dict              c_map corresponding to color_dot
        category_order      global variable with order of column variables for legend

    Returns: figure object
    """
    ### data ###
    data = email_df 

    ### multi select dropdown callbacks ###
    # show number of recipients: input of size variable in px.scatter
    b_size = None
    if 'show number of recipients' in dropdown:
        b_size = "number of recipients"
    # show workhours: show translucent bars during workhours 9-17
    work_hour = 'No'
    if 'show workhours' in dropdown:
        work_hour = 'Yes'
    # corresponding day: filter data so only email of one day are shown (same day selected in time range visualizer)
    from_chord = 'No'
    if 'corresponding day' in dropdown:
        from_chord = 'yes' # set variable here, changes are made later
        data = data[data['Day']==str(DAY)]
    # filter RE: only show first email sent in conversation. Replies of this email have the same sentiment and the content of the email are unknown.
    if 'filter RE' in dropdown:
        # data = data.drop_duplicates(subset = ['Subject'], keep='first' )
        data = data[data['Subject']==data['Subject without re']]

    # dropdown select category: select data of one category
    if category != 'All':
        data = data[data['class'] == category]
    # 'analyze by' dropdown: color dot
    color_dot_name = {'Department': 'DepFrom', 'Sentiment': 'sentiment'}[color_dot] # get column name used in data frame
    # select color corresponding to color
    c_dict = c_map_dep 
    if color_dot_name == 'sentiment':
        c_dict = c_map_sent
    # show color of categories only if 'show categories' is chosen in 'multi-select dropdown' and all categories are selected in the 'select category'
    # if one category is selected all would be the same color so then keep color by 'analyze by' 
    if 'show categories' in dropdown:
        if category == 'All':
            color_dot_name = 'class'
            color_dot = 'Class'
            c_dict = c_map_class
        
    ### start figure ###
    if from_chord == 'No': # show all days
        fig=px.scatter(data, x='time', y='from',hover_name='Subject', size = b_size,
                                                        hover_data=["number of recipients", "DepFrom", "sentiment", "class"], color = color_dot_name, color_discrete_map=c_dict, category_orders=category_order)
        if work_hour == 'Yes': # show workdays for two weeks
            fig.add_vrect(pd.to_datetime('2014-01-06 9:00'), pd.to_datetime('2014-01-06 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-07 9:00'), pd.to_datetime('2014-01-07 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-08 9:00'), pd.to_datetime('2014-01-08 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-09 9:00'), pd.to_datetime('2014-01-09 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-10 9:00'), pd.to_datetime('2014-01-10 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)

            fig.add_vrect(pd.to_datetime('2014-01-13 9:00'), pd.to_datetime('2014-01-13 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-14 9:00'), pd.to_datetime('2014-01-14 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-15 9:00'), pd.to_datetime('2014-01-15 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-16 9:00'), pd.to_datetime('2014-01-16 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-17 9:00'), pd.to_datetime('2014-01-17 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)

    else: # from_chord == true, show only one day
        fig=px.scatter(data, x='time', y='from',hover_name='Subject', size = b_size,
                                                        hover_data=["number of recipients","DepFrom", "sentiment", "class"], color = color_dot_name, color_discrete_map=c_dict,category_orders=category_order )
        if work_hour == 'Yes': # show workhours for one day
            fig.add_vrect(pd.to_datetime('2014-01-'+str(DAY)+' 9:00'), pd.to_datetime('2014-01-'+str(DAY)+' 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)

    # layout of figure, axes names, background color, legend and sorting of names by department first and than names.
    fig.update_xaxes(title_text = "Date", showgrid=False, color='white') # shows only horizontal lines
    fig.update_yaxes(title_text = 'Person', color = 'white', categoryorder = 'array', categoryarray= data.sort_values(['DepFrom','from'], ascending=False)['from'])
    fig.update_layout({'paper_bgcolor' : '#26232C', 'plot_bgcolor': '#26232C'}, legend_font_color='white', legend_title =color_dot)
    return fig

#**************************************
# Preprocessing of the data and functions for page 2:
#**************************************
def get_news_papers():
    """ 
    This function makes a dictionary that contains:
    - keys : newspaper names
    - values : a list of all files that are written by this news paper
    """
    news_papers = {}

    for k in range(845):
        path = 'data/articles/' + str(k) + '.txt'
        
        # collect the name of the news paper and remove the final space where needed
        title = next(line for line in open(path, 'r', encoding='latin-1').read().split('\n') if line != '')
        if title[-1] == ' ':
            title = title[:-1]

        # add the news paper and the index of the file to the dictionary
        if title not in news_papers.keys():
            news_papers[title] = [k]
        else:
            news_papers[title].append(k)

    return news_papers

# get all the newspapers, their corresponding files and names and store them
news_papers = get_news_papers()
news_papers_names = sorted(list(news_papers.keys()))

# set some punctuation and stopwords to remove from the words within the articles
removable_punctuation = '",:;?!()-.'
removable_words = set(STOPWORDS)

def get_word_frequency(file_list: list, rem_punc: str, rem_words: list) -> Counter:
    """
    This function goes over news articles and creates a counter to keep track of how many times each words appears.

    - file_list: list of the news articles
    - rem_punc: punctuation that can be removed from words
    - rem_words: stopwords (often used, meaningless words (for sentiment)) that we filter out
    Returns: counter of the interesting words
    """
    counter = Counter()
    
    for file in file_list:
        path = 'data/articles/' + str(file) + '.txt'
        
        with open(path, encoding='latin-1') as f:
            words = f.read().split()
            # convert all words to lower case
            words_lower = [word.lower() for word in words]
            # remove punctuation
            no_punc = [''.join(char for char in word if char not in rem_punc) for word in words_lower]
            # remove stopwords
            interesting_words = [word for word in no_punc if word not in rem_words and word != '']
            counter = counter + Counter(interesting_words)

    return counter


def plot_most_common_words(news_paper : str, n : int):
    """
    This function creates a graph that shows the n most frequently occuring words for a given newspaper.

    - news_paper: name of the newspaper of interest
    - n: number of (most frequent) words to show in the graph
    Returns: a barchart of the n most frequent words
    """
    # retrieve required files form news_papers dictionary for the given newspaper
    files = news_papers[news_paper] 
    counter = get_word_frequency(files, removable_punctuation, removable_words)
    df = pd.DataFrame(dict(counter.most_common(n)).items(), columns=['Word', 'Frequency'])
    
    fig = px.bar(df, x='Word', y='Frequency', color='Frequency', color_continuous_scale='viridis', title='Top ' + str(n) + ' most frequent words for "' + news_paper + '"')
    fig.update_layout(paper_bgcolor='#26232C', plot_bgcolor='#26232C', font_color = "#FEFEFE", font_size=13, yaxis=dict(gridcolor='#5c5f63'), font=dict(size=10))
    return fig


def plot_freq_words(words: list, news_papers_list : list = []):
    """
    This function creates the graph that shows:
    1) the total number of occurrences of a given list of words for all news articles
    2) If a (list of) newspaper(s) is selected, it shows the frequency of the given words for each newspaper

    - words: the words for which the frequencies are retrieved
    - news_papers_list: the newspapers for which the frequencies of the words are shown.
    Returns: A graph with word frequencies (for articles of newspapers)
    """

    # if the number of given newspapers is more than 10, only count the first word
    # this is done to keep the graph as readable as possible
    if len(news_papers_list) > 10:
        words = [words[0]]
    
    words_lower = [word.lower() for word in words]

    # create a dictionary to count the frequencies of the words
    freq_words = {w:0 for w in words_lower}

    # if no newspaper is given, go over all articles
    if news_papers_list == []:
        for k in range(845):
            path = 'data/articles/' + str(k) + '.txt'
            for word in words_lower:
                with open(path,  encoding='latin-1') as f:
                    read_words = f.read().split()
                    # convert to lower case and count the frequencies
                    read_words_lower = [word.lower() for word in read_words]
                    freq_words[word] = freq_words[word] + read_words_lower.count(word)
        
        # remove words with a frequency of 0
        freq_words = {k: v for k, v in freq_words.items() if v > 0}

        paper_counts_sorted = dict(sorted(freq_words.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame(paper_counts_sorted.items(), columns=['Word', 'Frequency'])
        
        title = 'Frequency of given words combined for all newspapers'
        fig = px.bar(df, x='Word', y='Frequency', color='Frequency', color_continuous_scale='viridis', title=title)
        fig.update_layout(paper_bgcolor='#26232C', plot_bgcolor='#26232C', font_color = "#FEFEFE", font_size=13, yaxis=dict(gridcolor='#5c5f63'), font=dict(size=10))
        return fig
    
    else:
        # make containing a dictionary for every newspaper for the word frequencies
        paper_freqs = {paper: freq_words.copy() for paper in news_papers_list}
        for paper in news_papers_list:
            files = news_papers[paper]
            for file in files:
                path = 'data/articles/' + str(file) + '.txt'
                for word in words_lower:
                    with open(path, encoding='latin-1') as f:
                        read_words = f.read().split()
                        read_words_lower = [word.lower() for word in read_words]
                        no_punc_words = [''.join(char for char in word if char not in removable_punctuation) for word in read_words_lower]
                        # remove the words that are empty due to punctuation removal
                        no_punc_words = list(filter(None, no_punc_words))
                        paper_freqs[paper][word] = paper_freqs[paper][word] + no_punc_words.count(word)

        
        for key, val in paper_freqs.items():
            paper_freqs[key] = {k: v for k, v in val.items() if v > 0}

        df = pd.DataFrame.from_dict(paper_freqs, orient="index").stack().to_frame().reset_index()
        df.rename(columns={'level_0': 'Newspaper', 'level_1': 'Word', 0: 'Frequency'}, inplace=True)
        
        title = 'Frequency of given words per newspaper'
        fig = px.bar(df, x='Newspaper', y='Frequency', color='Word', category_orders={'Newspaper': sorted(news_papers_list), 'Word':sorted(words)}, barmode='group', text='Word', color_discrete_sequence= discrete_colors, title=title)
        fig.update_layout(paper_bgcolor='#26232C', plot_bgcolor='#26232C', font_color = "#FEFEFE", font_size=13, yaxis=dict(gridcolor='#5c5f63'), font=dict(size=10))
        return fig
    
def centrum_sentinel(path : str):
    """
    This functions retrieves the date and time of the news article at the given path.
    
    - path: the location of the article
    Returns: time that the article was written.

    Note: This is for the Centrum Sentinel newspaper as it only had newsarticles written on the same day.
    So the graph would have been a vetical line without this. 
    """
    lines = [line for line in open(path, 'r').read().split('\n')]
    # up on inspecting the files, the dates and times were always on these specific lines and places
    date = lines[3] +' '+ lines[5][0:4]
    
    return dt.strptime(date, '%d %B %Y %H%M')


def modern_rubicon(path : str):
    """
    This functions retrieves the date and time of the news article at the given path.
    
    - path: the location of the article
    Returns: time that the article was written.

    Note: This is for the Modern Rubicon newspaper as it only had newsarticles written on the same day.
    So the graph would have been a vetical line without this. 
    """

    lines = [line for line in open(path, 'r').read().split('\n')]
    # these were the lines containing the time
    split_line = lines[5].split('-')[0].split(' ')
    
    # get a list of all words starting with a digit
    time = [word for word in split_line if word!='' and word[0].isdigit()]
    
    # if multiple words were contained, take the first one
    # and add a 0 to make sure the time is of from 00:00 instead of i.e. 9:50
    if len(time) > 1:
        time = ['0'+n for n in time if len(n)==3]
    if len(time[0]) == 3:
        time[0] = '0'+time[0]

    # put the date and time together
    date = lines[3] +' '+ time[0]
        
    return dt.strptime(date, '%d %B %Y %H%M')

def tethys_news(path : str, file : int):
    """
    This functions retrieves the date and time of the news article at the given path.
    
    - path: the location of the article
    Returns: time that the article was written.

    Note: This is for the Tethys News newspaper as it only had newsarticles written on the same day.
    So the graph would have been a vetical line without this. 
    """
    
    # store the files that contain an AM time instead of PM
    am_files = [92, 453, 539, 726, 829]

    lines = [line for line in open(path, 'r').read().split('\n')]
    split_line = lines[5].split(' ')

    # retrieve all words starting with a digit
    time = [word for word in split_line if word!='' and word[0].isdigit()]
    
    # one file did not contain a time, so based on context we assumed it
    if time == []:
        time = '0705'
    else:
        # remove the semicolon in the times
        time = time[0].replace(':', '')
        if len(time) == 3:
            time = '0' + time

    if file in am_files:
        time = time + 'AM'
    else:
        time = time + 'PM'

    # add the date and time together
    date = lines[3] +' '+ time
    
    return dt.strptime(date, '%d %B %Y %I%M%p')


def plot_sentiment_newspaper(newspaper : str):
    """
    This function creates a graph that shows the sentiment score of the news articles for a given newspaper.
    
    - newspaper: the name of the newspaper
    Returns: a line graph that shows the sentiment over time of the news articles
    """
    files = news_papers[newspaper]
    sia = SentimentIntensityAnalyzer()

    dates = []
    sent_scores = []

    for file in files:
        path = 'data/articles/' + str(file) + '.txt'
        
        # if the newspaper is of one of the three below,it only made news articles during one day, 
        # so the times were retrieved in a hardcoded way
        if newspaper == 'Centrum Sentinel':
            date = centrum_sentinel(path)
        
        elif newspaper =='Modern Rubicon':
            date = modern_rubicon(path)

        elif newspaper == 'Tethys News':
            date = tethys_news(path, file)
        
        else:
            date = next(line for line in open(path, 'r',  encoding='latin-1').read().split('\n') if line != '' and line[0].isdigit() and line.count('of')==0)

            # filter out the faulty dates and correct them
            if date[-1] == ' ':
                date = date[:-1]
            if date == '21 January 2014  1405':
                date = '21 January 2014'
            if date == '13June 2010':
                date = '13 June 2010'

            # based on the format of the date in the article, create a date object
            if date.count('/') > 0:
                date = dt.strptime(date, '%Y/%m/%d').date()
            else:
                date = dt.strptime(date, '%d %B %Y').date()
        
        dates.append(date)
        
        # compute the sentiment scores for each article
        with open(path, 'r',  encoding='latin-1') as f:
            sent = sia.polarity_scores(f.read())
            sent_scores.append(sent['compound'])

    d = {'Date': dates, 'Sentiment Score': sent_scores}
    df = pd.DataFrame(d).sort_values('Date').reset_index(drop='index')
    
    fig = px.scatter(df, x='Date', y='Sentiment Score', color='Sentiment Score', color_continuous_scale='rdylgn', range_color=[-1, 1], title='Sentiment score over time for "' + newspaper + '"')
    fig.update_traces(mode="markers+lines", line_color='#CCCCCC', marker=dict(size=12,line=dict(width=1, color='#FEFEFE')))
    fig.update_layout(paper_bgcolor='#26232C', plot_bgcolor='#26232C', font_color = "#FEFEFE", xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#5c5f63', zerolinecolor='#5c5f63'), font=dict(size=10))
    return fig


########## RENDER PAGE CONTENT --> I.E. CHANGE THE PAGE WHEN THE USER NAVIGATES TO DIFFERENT PAGE VIA THE SIDEBAR ##########
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    """
    Renders the page content given a path.
    - pathname: the path name of the page (selected from the sidebar); either "/" (home), "/page-1" (page 1) or "/page-2" (page 2)
    """
    
    # show home page as a starting page
    if pathname == "/": 
        return home.layout

################ page 1 ########################
    elif pathname == "/page-1":
        return html.Div([
    html.Div([html.H1("Email Correspondence Network Graph \n")],
             className="row",
             style={'textAlign': "center"}),
    html.Div(
        className="row",
        children=[
            # left side two input components of the network graph
            html.Div(
                className="two columns",
                children=[
                    dcc.Markdown(d("""
                            ** **
                            **Time Range To Visualize** 
                            
                            Choose a day (all days are in January 2014).
                            """),style = {'font-size': 16, "color": '#FEFEFE'}),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.RadioItems(id='my-range-slider', options=[6,7,8,9,10,13,14,15,16,17], value = 6, style={'color': '#FEFEFE', 'font-size': 13,  "margin": "auto", "max-width": "800px", 'display': 'flex'}),
                            html.Br(),
                            html.Div(id='output-container-range-slider')
                        ],
                    ), 
                ],style={'height': '400px', 'text-align': 'left','position':'relative', 'left':6}
            ),
            
            # display dropdown component
            html.Div(
                className="two columns",
                children=[dcc.Markdown(d("""
                            ** **
                            **Analyze by**
                            
                            """), style = {'font-size': 16, "color": '#FEFEFE'}
                                      ),
                            html.Div(className="six columns",
                                     children=[dcc.Dropdown(id="input1", options=['Department', 'Sentiment'],placeholder="Analyze by", value='Department', clearable=False), html.Div(id='output-container-dropdown')
                                              ],
                                    ), 
                         ],style={'font-size': 13, 'position':'relative', "margin": "auto", "top": "-270px", 'left':0, 'width': "150px"}),
            
            # display the graph component
            html.Div(
                children=[
                    html.Iframe(id="my-graph",
                        src=chord_graph(DAY, analize_by),
                        style={'text-align': 'left','position':'relative', 'left':20, "height": "640px", "width": "640px", 'border':"0"},
                    )
                ]
            ),
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph2",
                                    figure=bar_deps(DAY, analize_by))], style ={'text-align': 'left','position':'relative', "height": "400px", "width": "650px", "top": "-480px"}
                ),
            # div to house multi-select dropdown
            html.Div(
                className="Rianne Row",
                children=[
                    dcc.Markdown(d("""
                            Multi-Select Dropdown
                            """),style = {'font-size': 16, "color": '#FEFEFE'}),
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Dropdown(id='dropdown', options=['show workhours','filter RE', 'show number of recipients', 'corresponding day', 'show categories'],multi = True, value = 'show workhours', style={'color': '#26232C', 'font-size': 13}),
                            html.Br(),
                            html.Div(id='output-container-chord-value')
                        ],
                    ), ],style={'text-align': 'left','position':'relative', "left": -640, "top": "-400 px"}
                ),
            # div to house select category drowdown
            html.Div(
                className="Rianne Row",
                children=[
                    dcc.Markdown(d("""
                        Select category
                        """), style = {'font-size': 16, "color": '#FEFEFE'}
                                  ),
                        html.Div(className="six columns",
                                 children=[dcc.Dropdown(id="category", options=['All', 'Work', 'Change/schedule', 'Social', 'Weird', 'Other', 'Undefined','Spam'],value='All', clearable=False,), 
                                           html.Div(id='output-container-cat')
                                          ],
                                ), 
                ],style={'text-align': 'left','position':'relative', "left": -550, "top": "-400 px"}
            ),
            # div to house dotplot graph
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph-dotplot",
                                    # NOTE: 
                                    # commented configuration in the github. Gives a bit of overlap on my laptop (Rianne) #330
                                    # figure = dotplotgraph(color_dot, DAY = 6))], style ={'text-align': 'center','position': 'relative', 'width':1500, "top": "-350px"} 
                                    figure = dotplotgraph(color_dot, DAY = 6))], style ={'text-align': 'center','position': 'relative', 'width':1500, "top": "-300px"}
                ),

            ]
        )
    ])
######################## page 2 ############################
    elif pathname == "/page-2": 
        return html.Div(className='row0-ramon', children=[        
            html.Div(children=[html.H1('NewsPaper Analysis')]),
            
            html.Div(className='row1-1-ramon',
                     children=[
                        # div to house the word cloud and its title
                        html.Div(className='row2-3-ramon',
                                 children=[
                                    dcc.Markdown(d("""
                                        **Word cloud of the most frequent words** 
                                        """), style = {'width':'100%', 'font-size': 16, "color": '#FEFEFE', 'text-align':'center'}),
                                    html.Img(className='img-ramon', src="assets/wordcloud.png"),            
                        ]),
                        
                        
                        html.Div(className='row2-0-ramon',
                                 children=[
                                    # div to house the search bar and the frequency graph
                                    html.Div(className='row2-1-ramon', 
                                             children=[
                                                html.Div(className='row3-ramon',
                                                            children=[
                                                                dcc.Markdown("Type some words, seperated by spaces, to show their frequencies:", style = {'font-size': 16, "color": '#FEFEFE'}),
                                                                dcc.Input(id='multi-words', value='kronos pok government', type='text', placeholder='Type your words here', style={'width':'95%', "border-radius": 3})
                                                                
                                                ]),
                                                html.Div(className='graph-ramon',
                                                            children=[
                                                                dcc.Graph(id="freq-words", figure=plot_freq_words(['kronos', 'pok'], news_papers_names))
                                                            ])
                                             ]),
                                    # div to house the newspaper options for the frequency graph
                                    html.Div(className='row2-2-ramon',
                                             children=[
                                                dcc.Markdown("Select the newspaper(s)", style = {'font-size': 16, "color": '#FEFEFE', 'text-align':'center'}),
                                                dcc.Checklist(id="all-or-none", options=[{"label": "(De)Select All", "value": "All"}], value=[], style={'font-size': 12, 'text-align':'center', "color": '#FEFEFE'}, inputStyle={"margin-right": "3px", 'margin-left': '3px'}),
                                                dcc.Checklist(options=sorted(news_papers_names), value=news_papers_names, id='np-dropdown1', style={'font-size': 12, 'text-align':'center', "color": '#FEFEFE'}, inputStyle={"margin-right": "3px", 'margin-left': '3px'})
                                             ])
                                    
                        ])
                        
                     ]),
            
            # div to house the inputs for the most common words graph and sentiment graph
            html.Div(className='row1-2-ramon', 
                     children=[
                        html.Div(style={'width':'40%'}, children=[
                            dcc.Markdown("Choose the number of words for the left graph (5-50):", style = {'font-size': 16, "color": '#FEFEFE'}),
                            dcc.Input(id='input-number', value=20, type='number', placeholder='Type your number here', min=5, max=50, step=1, style = {'width': '25%', "border-radius": 3})
                        ]),
                        html.Div(style={'width':'32%'}, children=[
                            dcc.Markdown("Choose the newspaper for both graphs:", style = {'font-size': 16, "color": '#FEFEFE'}),
                            dcc.Dropdown(options=sorted(news_papers_names), value='Worldwise', id='np-dropdown2', placeholder='Select a newspaper')
                        ])  
                     ]),

            # div to house the most common words graph and the sentiment graph
            html.Div(className='row1-ramon',
                     children=[
                        html.Div(className='row2-ramon',
                                 children=[        
                                    html.Div(className='graph-ramon',
                                             children=[
                                                dcc.Graph(id="mc-words", figure=plot_most_common_words('Worldwise', 20))
                                             ]
                                    )
                        ]),
                        html.Div(className='row2-ramon',
                                 children=[
                                    html.Div(className='graph-ramon',
                                             children=[
                                                dcc.Graph(id="sentiment", figure=plot_sentiment_newspaper('Worldwise'))
                                             ]
                                    )
                        ])
                     ])
        ])
    # if the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

###################################### CALLBACKS #######################################33
### callbacks page 1 ###
# callback for chord diagram to update the global variable of DAY and analize_by
@app.callback(
    dash.dependencies.Output('my-graph', 'src'),
    [dash.dependencies.Input('my-range-slider', 'value'), dash.dependencies.Input('input1', 'value')])
def update_output(value, value2):
    """
    Updates the output based on a callback. Returns the chord diagram based on the callbacks.
    - value: the callback from the time range (it is a day)
    - value2: the callback from the top most dropdown menu (either "Department" or "Sentiment")
    """
    DAY = value
    analize_by = value2
    return chord_graph(value, value2)
# callback for bar chart to update the global variable of DAY and analize_by
@app.callback(
    dash.dependencies.Output('my-graph2', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value'), dash.dependencies.Input('input1', 'value')])
def update_output(value, value2):
    """
    Updates the output based on a callback. Returns the stacked bar chart based on the callbacks.
    - value: the callback from the time range (it is a day)
    - value2: the callback from the top most dropdown menu (either "Department" or "Sentiment")
    """
    # to update the global variable of DAY
    # DAY = value
    # analize_by = value2
    return bar_deps(value, value2)
# callbacks for dotplot to update figure by DAY, analyze_by, multi-select dropdown and select category
@app.callback(
    dash.dependencies.Output('my-graph-dotplot', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value'), dash.dependencies.Input('input1', 'value') ,
     dash.dependencies.Input('dropdown', 'value'),dash.dependencies.Input('category', 'value')])
def update_output(time, c_dot, dropdown, category):
    """
    Updates the output based on a callback. Returns the dot plot graph based on the callbacks.
    - time: the callback from the time range (it is a day)
    - c_dot: the callback from the top most dropdown menu (either "Department" or "Sentiment")
    - dropdown: the callback from the multi-select dropdown (one string with the selected options (combination of: "show workhours", "filter RE", "show number of recipients", "corresponding day", "show categories"))
    - category: the callback from left bottom dropdown select category (either "All", "Work", "Change/schedule", "Social", "Weird", "Other", "Undefined","Spam"])
    """
    return dotplotgraph( color_dot = c_dot, DAY = time, dropdown = dropdown, category=category)

### callbacks page 2 ###
# callback for the (de)select all checklist
@app.callback(
    Output("np-dropdown1", "value"),
    [Input("all-or-none", "value")]
)
def select_all_none(all_selected):
    """
    Updates the output based on a callback. Returns the bar chart based on the callbacks.
    - all_selected: callback from the (de)select all checklist (top right)
    """
    all_or_none = []
    all_or_none = [paper for paper in news_papers_names if all_selected]
    return all_or_none

#callback for the frequency graph, using the given words and the selected newspaper
@app.callback(
    dash.dependencies.Output('freq-words', 'figure'),
    dash.dependencies.Input('multi-words', 'value'),
    dash.dependencies.Input('np-dropdown1', 'value'))
def update_output(value1, value2):
    """
    Updates the output based on a callback. Returns the bar chart based on the callbacks.
    - value1: callback of the words typed in the searchbar (top middle)
    - value2: callback for the selected newspapers in the checklist (top right)
    """
    words = value1.split(' ')
    return plot_freq_words(words, value2)

# callback for the most common words graph, using n most frequent words and the selected newspaper
@app.callback(
    dash.dependencies.Output('mc-words', 'figure'),
    dash.dependencies.Input('np-dropdown2', 'value'),
    dash.dependencies.Input('input-number', 'value'))
def update_output(value1, value2):
    """
    Updates the output based on a callback. Returns the bar chart based on the callbacks.
    - value1: callback of the dropdown menu of all the different newspapers (bottom right)
    - value2: callback of the number shown in the bar (bottom left)
    """
    return plot_most_common_words(value1, value2)

# callback for the sentiment graph, using the selected newspaper
@app.callback(
    dash.dependencies.Output('sentiment', 'figure'),
    dash.dependencies.Input('np-dropdown2', 'value'))
def update_output(value1):
    """
    Updates the output based on a callback. Returns the line graph based on the callbacks.
    - value1: callback of the dropdown menu of all the different newspapers (bottom right)
    """
    return plot_sentiment_newspaper(value1)


if __name__=='__main__':
    app.run_server(debug=False, port=3000)

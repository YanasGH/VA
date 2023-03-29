
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
from collections import Counter
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

########## BEGIN: LEFT HERE AS AN EXAMPLE --> DELETE LATER ##########
df = pd.read_csv('https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Bootstrap/Side-Bar/iranian_students.csv')
############################## END ##############################

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
sentiment = email_df['Subject'].apply(lambda x: sia.polarity_scores(x))
email_df[['neg', 'neu', 'pos', 'compound']] = pd.json_normalize(sentiment)

cmap_custom = ['#5b4dd6', '#c933bc', '#ffc60a', '#ff5960', '#ff9232', '#ff2b90']
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

def edge_node(att, toatt, fromatt, day_search=None):
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

YEAR = 6
analize_by = 'Department'
def chord_graph(YEAR, analize_by):
    if analize_by == 'Department':
        return f'assets/graph_chord_{YEAR}.html'
    else:
        return f'assets/graph_chord_sentiment_{YEAR}.html'

def bar_deps(YEAR, analize_by):
    if analize_by == 'Department':
        adj_matrix, _, _ = edge_node('CurrentEmploymentType', 'DepTo', 'DepFrom', YEAR)
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
        
        dfday = email_df[email_df['Day']==str(YEAR)]
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

#**************************************
# Ramon's preprocessing of the data and functions:
#**************************************
def get_news_papers():
    """ 
    This function makes a dictionary that contains:
    - keys : news paper names
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

news_papers = get_news_papers()
news_papers_names = list(news_papers.keys())

removable_punctuation = '",:;?!()-.'
removable_words = set(STOPWORDS)

def get_word_frequency(file_list: list, rem_punc: str, rem_words: list) -> Counter:
    counter = Counter()
    
    for file in file_list:
        path = 'data/articles/' + str(file) + '.txt'
        
        with open(path, encoding='latin-1') as f:
            words = f.read().split()
            words_lower = [word.lower() for word in words]
            no_punc = [''.join(char for char in word if char not in rem_punc) for word in words_lower]
            interesting_words = [word for word in no_punc if word not in rem_words and word != '']
            counter = counter + Counter(interesting_words)

    return counter


def plot_most_common_words(news_paper : str, n : int):
    files = news_papers[news_paper]
    counter = get_word_frequency(files, removable_punctuation, removable_words)
    df = pd.DataFrame(dict(counter.most_common(n)).items(), columns=['Word', 'Frequency'])
    
    fig = px.bar(df, x='Word', y='Frequency', color='Frequency', color_continuous_scale='viridis', title='Top ' + str(n) + ' most frequent words for "' + news_paper + '"')
    fig.update_layout(paper_bgcolor='#26232C', plot_bgcolor='#26232C', font_color = "#FEFEFE", font_size=13, yaxis=dict(gridcolor='#5c5f63'), font=dict(size=10))
    return fig


def plot_freq_words(words: list, news_papers_list : list = []):
    if len(news_papers_list) > 10:
        words = [words[0]]
    
    words_lower = [word.lower() for word in words]

    freq_words = {w:0 for w in words_lower}

    if news_papers_list == []:
        for k in range(845):
            path = 'data/articles/' + str(k) + '.txt'
            for word in words_lower:
                with open(path,  encoding='latin-1') as f:
                    read_words = f.read().split()
                    read_words_lower = [word.lower() for word in read_words]
                    freq_words[word] = freq_words[word] + read_words_lower.count(word)
        
        freq_words = {k: v for k, v in freq_words.items() if v > 0}

        paper_counts_sorted = dict(sorted(freq_words.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame(paper_counts_sorted.items(), columns=['Word', 'Frequency'])
        
        title = 'Frequency of given words combined for all newspapers'
        fig = px.bar(df, x='Word', y='Frequency', color='Frequency', color_continuous_scale='viridis', title=title)
        fig.update_layout(paper_bgcolor='#26232C', plot_bgcolor='#26232C', font_color = "#FEFEFE", font_size=13, yaxis=dict(gridcolor='#5c5f63'), font=dict(size=10))
        return fig
    
    else:
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
                        no_punc_words = list(filter(None, no_punc_words))
                        paper_freqs[paper][word] = paper_freqs[paper][word] + no_punc_words.count(word)

        
        for key, val in paper_freqs.items():
            paper_freqs[key] = {k: v for k, v in val.items() if v > 0}

        df = pd.DataFrame.from_dict(paper_freqs, orient="index").stack().to_frame().reset_index()
        df.rename(columns={'level_0': 'Newspaper', 'level_1': 'Word', 0: 'Frequency'}, inplace=True)
        
        title = 'Frequency of given words per newspaper'
        fig = px.bar(df, x='Newspaper', y='Frequency', color='Word', barmode='group', text='Word', color_discrete_sequence= discrete_colors, title=title)
        fig.update_layout(paper_bgcolor='#26232C', plot_bgcolor='#26232C', font_color = "#FEFEFE", font_size=13, yaxis=dict(gridcolor='#5c5f63'), font=dict(size=10))
        return fig
    
def centrum_sentinel(path : str):
    lines = [line for line in open(path, 'r').read().split('\n')]
    date = lines[3] +' '+ lines[5][0:4]
    
    return dt.strptime(date, '%d %B %Y %H%M')


def modern_rubicon(path : str):
    lines = [line for line in open(path, 'r').read().split('\n')]
    split_line = lines[5].split('-')[0].split(' ')
    
    time = [word for word in split_line if word!='' and word[0].isdigit()]
    
    if len(time) > 1:
        time = ['0'+n for n in time if len(n)==3]
    if len(time[0]) == 3:
        time[0] = '0'+time[0]

    date = lines[3] +' '+ time[0]
        
    return dt.strptime(date, '%d %B %Y %H%M')

def tethys_news(path : str, file : int):
    
    am_files = [92, 453, 539, 726, 829]

    lines = [line for line in open(path, 'r').read().split('\n')]
    split_line = lines[5].split(' ')

    time = [word for word in split_line if word!='' and word[0].isdigit()]
    if time == []:
        time = '0705'
    else:
        time = time[0].replace(':', '')
        if len(time) == 3:
            time = '0' + time

    if file in am_files:
        time = time + 'AM'
    else:
        time = time + 'PM'

    date = lines[3] +' '+ time
    
    return dt.strptime(date, '%d %B %Y %I%M%p')


def plot_sentiment_newspaper(newspaper : str):
    files = news_papers[newspaper]
    sia = SentimentIntensityAnalyzer()

    dates = []
    sent_scores = []

    for file in files:
        path = 'data/articles/' + str(file) + '.txt'
        
        if newspaper == 'Centrum Sentinel':
            date = centrum_sentinel(path)
        
        elif newspaper =='Modern Rubicon':
            date = modern_rubicon(path)

        elif newspaper == 'Tethys News':
            date = tethys_news(path, file)
        
        else:
            date = next(line for line in open(path, 'r',  encoding='latin-1').read().split('\n') if line != '' and line[0].isdigit() and line.count('of')==0)

            if date[-1] == ' ':
                date = date[:-1]
            if date == '21 January 2014  1405':
                date = '21 January 2014'
            if date == '13June 2010':
                date = '13 June 2010'

            if date.count('/') > 0:
                date = dt.strptime(date, '%Y/%m/%d').date()
            else:
                date = dt.strptime(date, '%d %B %Y').date()
        
        dates.append(date)
        
        with open(path, 'r',  encoding='latin-1') as f:
            sent = sia.polarity_scores(f.read())
            sent_scores.append(sent['compound'])

    d = {'Date': dates, 'Sentiment Score': sent_scores}
    df = pd.DataFrame(d).sort_values('Date').reset_index(drop='index')
    
    fig = px.line(df, x='Date', y='Sentiment Score', title='Sentiment score over time for "' + newspaper + '"')
    fig.update_traces(mode="markers+lines", line_color='#ffff33')
    fig.update_layout(paper_bgcolor='#26232C', plot_bgcolor='#26232C', font_color = "#FEFEFE", xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#5c5f63', zerolinecolor='#5c5f63'), font=dict(size=10))
    return fig


########## RENDER PAGE CONTENT --> I.E. CHANGE THE PAGE WHEN THE USER NAVIGATES TO DIFFERENT PAGE VIA THE SIDEBAR ##########
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    # show home page as a starting page
    if pathname == "/": 
        return home.layout

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
                                     children=[dcc.Dropdown(id="input1", options=['Department', 'Sentiment'],placeholder="Analyze by", value='Department'), html.Div(id='output-container-dropdown')
                                              ],
                                    ), 
                         ],style={'font-size': 13, 'position':'relative', "margin": "auto", "top": "-270px", 'left':0, 'width': "150px"}),
            
            # display the graph component
            html.Div(
                children=[
                    html.Iframe(id="my-graph",
                        src=chord_graph(YEAR, analize_by),
                        style={'text-align': 'left','position':'relative', 'left':20, "height": "640px", "width": "640px", 'border':"0"},
                    )
                ]
            ),
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph2",
                                    figure=bar_deps(YEAR, analize_by))], style ={'text-align': 'left','position':'relative', "height": "400px", "width": "650px", "top": "-480px"}
                ),

            ]
        )
    ])

    elif pathname == "/page-2": #THIS IS AN EXAMPLE FOR NOW
        return html.Div(className='row0-ramon', children=[        
            html.Div(children=[html.H1('NewsPaper Analysis')]),
            
            html.Div(className='row1-1-ramon',
                     children=[
                        html.Div(className='row2-3-ramon',
                                 children=[
                                    dcc.Markdown(d("""
                                        **Word cloud of the most frequent words** 
                                        """), style = {'width':'100%', 'font-size': 16, "color": '#FEFEFE', 'text-align':'center'}),
                                    html.Img(className='img-ramon', src="assets/wordcloud.png"),            
                        ]),
                        
                        
                        html.Div(className='row2-0-ramon',
                                 children=[
                                    html.Div(className='row2-1-ramon', 
                                             children=[
                                                html.Div(className='row3-ramon',
                                                            children=[
                                                                dcc.Markdown("Type some words, seperated by spaces, to show their frequencies:", style = {'font-size': 16, "color": '#FEFEFE'}),
                                                                dcc.Input(id='multi-words', value='kronos pok', type='text', placeholder='Type your words here', style={'width':'100%'})
                                                                
                                                ]),
                                                html.Div(className='graph-ramon',
                                                            children=[
                                                                dcc.Graph(id="freq-words", figure=plot_freq_words(['kronos', 'pok'], news_papers_names))
                                                            ])
                                             ]),
                                    html.Div(className='row2-2-ramon',
                                             children=[
                                                dcc.Markdown("Select the newspaper(s)", style = {'font-size': 16, "color": '#FEFEFE', 'text-align':'center'}),
                                                dcc.Checklist(id="all-or-none", options=[{"label": "(De)Select All", "value": "All"}], value=[], style={'font-size': 16, 'text-align':'center', "color": '#FEFEFE'}, inputStyle={"margin-right": "3px", 'margin-left': '3px'}),
                                                dcc.Checklist(options=news_papers_names, value=news_papers_names, id='np-dropdown1', style={'font-size': 16, 'text-align':'center', "color": '#FEFEFE'}, inputStyle={"margin-right": "3px", 'margin-left': '3px'})
                                             ])
                                    
                        ])
                        
                     ]),
            
            html.Div(className='row1-2-ramon', 
                     children=[
                        html.Div(style={'width':'30%'}, children=[
                            dcc.Markdown("Choose the number of words for the left graph (1-50):", style = {'font-size': 16, "color": '#FEFEFE'}),
                            dcc.Input(id='input-number', value=20, type='number', placeholder='Type your number here', min=1, max=50, step=1, style = {'width': '25%'})
                        ]),
                        html.Div(style={'width':'40%'}, children=[
                            dcc.Markdown("Choose the newspaper for both graphs:", style = {'font-size': 16, "color": '#FEFEFE'}),
                            dcc.Dropdown(options=news_papers_names, value='The Orb', id='np-dropdown2', placeholder='Select a newspaper')
                        ])  
                     ]),

            html.Div(className='row1-ramon',
                     children=[
                        html.Div(className='row2-ramon',
                                 children=[        
                                    html.Div(className='graph-ramon',
                                             children=[
                                                dcc.Graph(id="mc-words", figure=plot_most_common_words('The Orb', 20))
                                             ]
                                    )
                        ]),
                        html.Div(className='row2-ramon',
                                 children=[
                                    html.Div(className='graph-ramon',
                                             children=[
                                                dcc.Graph(id="sentiment", figure=plot_sentiment_newspaper('The Orb'))
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

########## CALLBACKS ##########
@app.callback(
    dash.dependencies.Output('my-graph', 'src'),
    [dash.dependencies.Input('my-range-slider', 'value'), dash.dependencies.Input('input1', 'value')])
def update_output(value, value2):
    # to update the global variable of YEAR and analize_by
    YEAR = value
    analize_by = value2
    return chord_graph(value, value2)
@app.callback(
    dash.dependencies.Output('my-graph2', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value'), dash.dependencies.Input('input1', 'value')])
def update_output(value, value2):
    # to update the global variable of YEAR
    YEAR = value
    analize_by = value2
    return bar_deps(value, value2)


# Ramon's callbacks
@app.callback(
    Output("np-dropdown1", "value"),
    [Input("all-or-none", "value")]
)
def select_all_none(all_selected):
    all_or_none = []
    all_or_none = [paper for paper in news_papers_names if all_selected]
    return all_or_none

@app.callback(
    dash.dependencies.Output('freq-words', 'figure'),
    dash.dependencies.Input('multi-words', 'value'),
    dash.dependencies.Input('np-dropdown1', 'value'))
def update_output(value1, value2):
    words = value1.split(' ')
    return plot_freq_words(words, value2)

@app.callback(
    dash.dependencies.Output('mc-words', 'figure'),
    dash.dependencies.Input('np-dropdown2', 'value'),
    dash.dependencies.Input('input-number', 'value'))
def update_output(value1, value2):
    return plot_most_common_words(value1, value2)

@app.callback(
    dash.dependencies.Output('sentiment', 'figure'),
    dash.dependencies.Input('np-dropdown2', 'value'))
def update_output(value1):
    return plot_sentiment_newspaper(value1)


if __name__=='__main__':
    app.run_server(debug=True, port=3000)

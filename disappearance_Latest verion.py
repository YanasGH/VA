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
import holoviews as hv
from holoviews import opts, dim
from bokeh.models import HoverTool
from bokeh.themes import Theme
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")

# from gensim.utils import simple_preprocess
# from gensim.models.doc2vec import TaggedDocument

########## BEGIN: LEFT HERE AS AN EXAMPLE --> DELETE LATER ##########
df = pd.read_csv('https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Bootstrap/Side-Bar/iranian_students.csv')
############################## END ##############################

############################## Colors ##############################
# Main background                   #26232C
# Secondary background / sidebar    #cbd3dd
# Color buttons sidebar
# Text mainbackground               white '#FEFEFE'
# Text sidebar                      white '#FEFEFE'
# cmap (rainbow ish)                cmap_custom = ['#5b4dd6', '#c933bc', '#ffc60a', '#ff5960', '#ff9232', '#ff2b90']
# color_discrete_sequence =        ['#B51E17', '#BDC4C5', '#379475']) (red, green, grey)




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
                        html.Span("Sth else", style={"color": "#FEFEFE"}),
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
email_df['DateTime'] = pd.to_datetime(email_df['Date'])
email_df['clean_name_from'] = email_df.From.str[:-19].replace(".", " ")
email_df['nr_recipients'] = [len(list) for list in email_df['To'].str.replace(',', '').str.split()]
email_df = email_df.sort_values('DepFrom')

email_df['Subject without re'] = email_df['Subject'].apply(lambda x: x[4:] if x.startswith('Re: ') or x.startswith('RE: ') else x )
email_first = email_df.drop_duplicates(subset = ['Subject'], keep='first' )

# df_class = pd.DataFrame(columns = ['Class', 'Description'], data= [['Spam', 'Spam'],['Social', 'Anniversary Birthday Event Social Night Evening'],['Work', 'Report Team Data Meeting Inspection Visit IPO Protocol Supplies GasTech Files'], [ 'Change/schedule', 'Change Schedule Appointments Meeting Deadline New Shift Announcement'],['Weird', 'Weird Plants text and drive'],['Other', 'Other No Hurry'],['Undefined', 'Undefined']])
# df_class['token'] = df_class['Description'].apply(lambda x: nlp(x))
# df_class

# classes = df_class['Class']
# for index, row in df_class.iterrows():
#     row
#     x = df_text['token'].apply(lambda x: x.similarity(row['token']))
#     print(row['Class'])


# def tokenize(doc):
#     return simple_preprocess(strip_tags(doc), deacc=True, min_len=4, max_len=15) #originally len 2 and 15


# df_text['tagged_docs'] = df_text.apply(lambda row: TaggedDocument(tokenize(row['Subject']), [str(row.name)]), axis=1)


# df_text['c_spam'] = df_text.apply(lambda x: 0 if x['class_spam']> 0.5 else -1, axis =1)
# df_text['c_social'] = df_text.apply(lambda x: 1 if x['class_social']> 0.5 else -1, axis =1)
# df_text['c_work'] = df_text.apply(lambda x: 2 if x['class_work']> 0.5 else -1, axis =1)
# df_text['c_schedule'] = df_text.apply(lambda x: 3 if x['class_schedule']> 0.5 else -1, axis =1)
# df_text['c_weird'] = df_text.apply(lambda x: 4 if x['class_weird']> 0.5 else -1, axis =1)
# df_text['c_undefined'] = df_text.apply(lambda x: 5 if x['class_undefined']> 0.5 else -1, axis =1)

# df_text['max'] = df_text[['class_spam', 'class_social', 'class_work', 'class_schedule', 'class_weird', 'class_other', 'class_undefined']].idxmax(axis=1)



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

############################# Set variables ####################
YEAR = 6
work_hour = 'Yes'
analize_by = 'Department'
color_dot = 'Department'
c_map_dep =  {'Administration':'#5b4dd6', 'Engineering':'#c933bc', 'Executive': '#ffc60a','Facilities': '#ff5960','Information Technology': '#ff9232','Security': '#ff2b90'}
c_map_sent = {'pos': '#379475','neu': '#BDC4C5', 'neg': '#B51E17'}

################################# Graph functions page 1 ###########################

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

def dotplotgraph( color_dot = 'Department', YEAR='6', dropdown = []): ##cbd3dd
    b_size = None
    work_hour = 'No'
    from_chord = 'No'
    data = email_df
    RE = False

    color_dot = {'Department': 'DepFrom', 'Sentiment': 'sentiment'}[color_dot]

    if 'show number of recipients' in dropdown:
        b_size = "nr_recipients"
    if 'show workhours' in dropdown:
        work_hour = 'Yes'
    if 'corresponding day' in dropdown:
        from_chord = 'yes'
    if 'filter RE' in dropdown:
        data = email_df.drop_duplicates(subset = ['Subject'], keep='first' )
        # RE = True

    c_dict = c_map_dep
    # if color_dot == 'DepFrom':
    #     c_dict = c_map_dep
    if color_dot == 'sentiment':
        c_dict = c_map_sent


    if from_chord == 'No':
        fig=px.scatter(data, x='DateTime', y='clean_name_from',hover_name='Subject', size = b_size,
                                                        hover_data=["nr_recipients"], color = color_dot, color_discrete_map=c_dict) #change hover column names color_discrete_map= cmap_custom
        if work_hour == 'Yes':
            fig.add_vrect(x0 = pd.to_datetime('2014-01-06 9:00'), x1 = pd.to_datetime('2014-01-06 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-07 9:00'), pd.to_datetime('2014-01-07 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-08 9:00'), pd.to_datetime('2014-01-08 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-09 9:00'), pd.to_datetime('2014-01-09 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-10 9:00'), pd.to_datetime('2014-01-10 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)

            fig.add_vrect(pd.to_datetime('2014-01-13 9:00'), pd.to_datetime('2014-01-13 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-14 9:00'), pd.to_datetime('2014-01-14 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-15 9:00'), pd.to_datetime('2014-01-15 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-16 9:00'), pd.to_datetime('2014-01-16 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)
            fig.add_vrect(pd.to_datetime('2014-01-17 9:00'), pd.to_datetime('2014-01-17 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)

        # if RE:
        #     fig.add_scatter(data,x='DateTime', y='clean_name_from',hover_name='Subject', size = b_size, hover_data=["nr_recipients"])
    else:
        subset = data[data['Day']==str(YEAR)]
        fig=px.scatter(subset, x='DateTime', y='clean_name_from',hover_name='Subject', size = b_size,
                                                        hover_data=["nr_recipients"], color = color_dot, color_discrete_map=c_dict ) #change hover column names
        if work_hour == 'Yes': # why does this not work??
            fig.add_vrect(pd.to_datetime('2014-01-'+str(YEAR)+' 9:00'), pd.to_datetime('2014-01-'+str(YEAR)+' 17:00'), line_width=0, fillcolor='#cbd3dd', opacity=0.2)


        # if RE:
        #     fig.add_scatter(subset,x='DateTime', y='clean_name_from',hover_name='Subject', size = b_size, hover_data=["nr_recipients"])


    fig.update_xaxes(title_text = "Date", showgrid=False, color='white')
    fig.update_yaxes(title_text = 'Person', color = 'white')
    fig.update_layout({'paper_bgcolor' : '#26232C', 'plot_bgcolor': '#26232C'}, legend_font_color='white')
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
                                     children=[dcc.Dropdown(id="input1", options=['Department', 'Sentiment'],placeholder="Analyze by", value='Department'), 
                                               html.Div(id='output-container-dropdown')
                                              ],
                                    ), 
                         ],style={'font-size': 13, 'position':'relative', "margin": "auto", "top": "-270px", 'left':0, 'width': "150px"}),

            #call backs Rianne

            html.Div(
                className="Rianne Row",
                children=[
                    dcc.Markdown(d("""
                            Multi-Select Dropdown
                            """),style = {'font-size': 16, "color": '#FEFEFE'}),
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Dropdown(id='dropdown', options=['show workhours','filter RE', 'show number of recipients', 'corresponding day'],multi = True, value = 'show workhours', style={'color': '#26232C', 'font-size': 13}),
                            html.Br(),
                            html.Div(id='output-container-chord-value')
                        ],
                    ), 
                ],style={'text-align': 'left','position':'relative', "left": 0, "top": "500 px"}
            ),
            
            # # display dropdown component
            # html.Div(
            #     className="two columns",
            #     children=[dcc.Markdown(d("""
            #                 ** **
            #                 **Color by**
                            
            #                 """), style = {'font-size': 16, "color": '#FEFEFE'}
            #                           ),
            #                 html.Div(className="six columns",
            #                          children=[dcc.Dropdown(id="color_dot", options=['DepFrom', 'sentiment'],placeholder="Color by", value='DepFrom'), 
            #                                    html.Div(id='output-container-dropdown-dotplot')
            #                                   ],
            #                         ), 
            #              ],style={'text-align': 'left','position':'relative'}
            #              ),
            
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

            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph-dotplot",
                                    figure = dotplotgraph(color_dot, YEAR = 6))], style ={'text-align': 'center','position': 'relative', 'width':1500}
                ),   

            ]
        )
    ])

    elif pathname == "/page-2": #THIS IS AN EXAMPLE FOR NOW
        return [
                html.H1('High School in Iran',
                        style={'textAlign':'center'}),
                dcc.Graph(id='bargraph',
                         figure=px.bar(df, barmode='group', x='Years',
                         y=['Girls High School', 'Boys High School']))
                ]
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

# value is sentiment or department in chord diagram

# dash.dependencies.Input('work_hour_id', 'value') dash.dependencies.Input('chord_value_id', 'value'),
@app.callback(
    dash.dependencies.Output('my-graph-dotplot', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value'), dash.dependencies.Input('input1', 'value') , #dash.dependencies.Input('color_dot', 'value')
     dash.dependencies.Input('dropdown', 'value')])
def update_output(chordplot, c_dot, dropdown):
    # to update the global variable of YEAR

    # analize_by = value2
    # return bar_deps(value, value2)
    return dotplotgraph( c_dot, YEAR = chordplot, dropdown = dropdown)

if __name__=='__main__':
    app.run_server(debug=True, port=3000)
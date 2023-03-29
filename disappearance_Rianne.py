
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
import networkx as nx #ADDED
import json #ADDED
from collections import Counter #ADDED 
from colour import Color #ADDED
import plotly.graph_objects as go

############################## Colors ##############################
# Main background                   #26232C
# Secondary background / sidebar    #cbd3dd
# Color buttons sidebar
# Text mainbackground               white '#FEFEFE'
# Text sidebar                      white '#FEFEFE'
# cmap 


########## BEGIN: LEFT HERE AS AN EXAMPLE --> DELETE LATER ##########
df = pd.read_csv('https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Bootstrap/Side-Bar/iranian_students.csv')
############################## END ##############################

########## CONFIGURE APP ##########
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, 'https://use.fontawesome.com/releases/v6.1.1/css/all.css'])
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


########## IMPORT DATA AND PREPROCESS ##########
email_df = pd.read_csv('email_headers.csv', encoding='cp1252')
employee_recs = pd.read_excel("EmployeeRecords.xlsx")

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
emp_dict = dict(zip(employee_recs.EmailAddress, employee_recs.CurrentEmploymentType))
for i in range(len(email_df)):
    email_df['Day'][i] = str(dt.strptime(email_df['Date'][i], "%m/%d/%Y %H:%M").day)
    email_df['DepFrom'][i]=emp_dict.get(email_df['From'][i])
    to_list = []
    for to in email_df['To'][i].split(', '):
        to_list.append(emp_dict.get(to))
    email_df['DepTo'][i]=to_list

# data preprocessing to dotplot
email_df['DateTime'] = pd.to_datetime(email_df['Date'])
email_df['clean_name_from'] = email_df.From.str[:-19].replace(".", " ")
email_df['nr_recipients'] = [len(list) for list in email_df['To'].str.replace(',', '').str.split()]

# create adjacency matrix to represent the from-to relationships of the emails
# edge2 = []
# for day in email_df['Day'].unique():
#     email_df_day = email_df[email_df['Day']==day]
#     adj_matrix = pd.DataFrame(employee_recs['EmailAddress']).set_index('EmailAddress')
    
#     for i in adj_matrix.index: #set up the matrix: rows correspond to the "From" person, columns to the "To" person
#         adj_matrix[f'{i}']= 0 


#     for i in adj_matrix.index: # fill in the matrix
#         from_df = email_df_day[email_df_day['From']==i]
#         to_total = []

#         for j in from_df['To']: 
#             to_total += j.split(', ')
#         to_total = Counter(to_total)

#         for c in adj_matrix.columns:
#             if (c != i) and (to_total.get(c) != None):
#                 adj_matrix[c][i] = to_total.get(c) # the syntax here is reverse: first is the "To" person, then it is the "From" person
    
#     for k in adj_matrix.index:
#         for j in adj_matrix.columns:
#             if k!=j and adj_matrix[j][k]!=0:
#                 edge2.append([adj_matrix[j][k],k, j, day])

# edge2 = pd.DataFrame(edge2, columns = ['TransactionAmt','Source','Target', 'Date'])

# nodes2 = []
# for i in adj_matrix.columns:
#     nodes2.append([i, i.split('@')[0].split('.')[0] + ' ' + i.split('@')[0].split('.')[1]])
# node2 = pd.DataFrame(nodes2, columns = ['Account','CustomerName'])
def edge_node(att, toatt, fromatt):
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

# adj_matrix, edge2, node2 = edge_node('EmailAddress', 'To', 'From')

YEAR=6 
ACCOUNT="Isia.Vann@gastech.com.kronos"

# define network function that plots the email correspondences
def network_graph(adj_matrix, edge2, node2, yearRange, AccountToSearch):

    edge1 = copy.deepcopy(edge2) 
    node1 = copy.deepcopy(node2) 

    # filter the record by datetime, to enable interactive control through the input box
    edge1['Datetime'] = "" # add empty Datetime column to edge1 dataframe
    accountSet=set() # contain unique account
    for index in range(0,len(edge1)):
        edge1['Datetime'][index] = dt.strptime(edge1['Date'][index], "%d") 
        if edge1['Datetime'][index].day!=yearRange: 
            edge1.drop(axis=0, index=index, inplace=True)
            continue
        accountSet.add(edge1['Source'][index])
        accountSet.add(edge1['Target'][index])

    # to define the centric point of the networkx layout
    shells=[]
    shell1=[]
    shell1.append(AccountToSearch)
    shells.append(shell1)
    shell2=[]
    for ele in accountSet:
        if ele!=AccountToSearch:
            shell2.append(ele)
    shells.append(shell2)

    G = nx.from_pandas_edgelist(edge1, 'Source', 'Target', ['Source', 'Target', 'TransactionAmt'], create_using=nx.MultiDiGraph())
#     pos = nx.drawing.layout.kamada_kawai_layout(G)
    if 'CustomerName' not in node1.columns:
        pos = nx.drawing.layout.random_layout(G)
        nx.set_node_attributes(G, node1.set_index('Account0')['Account'].to_dict(), 'Account')
    else:
        pos = nx.drawing.layout.kamada_kawai_layout(G)
        nx.set_node_attributes(G, node1.set_index('Account')['CustomerName'].to_dict(), 'CustomerName')
    

    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])

    if len(shell2)==0:
        traceRecode = []  # contains edge_trace, node_trace, middle_node_trace

        node_trace = go.Scatter(x=tuple([1]), y=tuple([1]),
                                mode='markers+text',
                                marker={'size': 50, 'color': '#6778b5'})
        traceRecode.append(node_trace)

        node_trace1 = go.Scatter(x=tuple([1]), y=tuple([1]),
                                mode='markers',
                                marker={'size': 50, 'color': '#6778b5'},
                                opacity=0)
        traceRecode.append(node_trace1)

        figure = {
            "data": traceRecode,
            "layout": go.Layout(showlegend=False, #title='Interactive Email Correspondence Visualization', 
                                margin={'b': 40, 'l': 2, 'r': 2, 't': 40},
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height=600, plot_bgcolor='#26232C', paper_bgcolor='#26232C'
                                )}
        return figure


    traceRecode = []  # contains edge_trace, node_trace, middle_node_trace
   
    colors = list(Color('lightcoral').range_to(Color('darkred'), len(G.edges())))
    colors = ['rgb' + str(x.rgb) for x in colors]

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        
        weight = 0 
        trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                           mode='lines',
                           line={'width': weight},
                           marker=dict(color='black'), 
                           line_shape='spline',
                           opacity=1)
        traceRecode.append(trace)
        index = index + 1
   
    node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="bottom center",
                            hoverinfo="text", marker={'size': 40, 'color': '#7e8cc4', 'line':dict(width=2, #6778b5
                                        color='#acb6e1')})

    index = 0
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        if 'CustomerName' not in node1.columns:
            hovertext = "Employee Name: " + str(G.nodes[node]['Account']) + "<br>"
        else:
            hovertext = "Employee Name: " + str(G.nodes[node]['CustomerName']) + "<br>"
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['hovertext'] += tuple([hovertext])
        index = index + 1

    traceRecode.append(node_trace)
   
    middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",
                                    marker={'size': 20, 'color': '#6778b5'},
                                    opacity=0)

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        hovertext = "From: " + str(G.edges[edge]['Source']) + "<br>" + "To: " + str(
            G.edges[edge]['Target']) + "<br>" + "Number of times: " + str(
            G.edges[edge]['TransactionAmt']) 
        middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
        middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
        middle_hover_trace['hovertext'] += tuple([hovertext])
        index = index + 1

    traceRecode.append(middle_hover_trace)
    
    figure = {
        "data": traceRecode,
        "layout": go.Layout(showlegend=False, hovermode='closest', #title='Interactive Email Correspondence Visualization',
                            margin={'b': 40, 'l': 2, 'r': 2, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=600, plot_bgcolor='#26232C', paper_bgcolor='#26232C', 
                            clickmode='event+select',
                            annotations=[
                                dict(
                                    ax=G.nodes[edge[0]]['pos'][0],# + G.nodes[edge[1]]['pos'][0]) / 2,
                                    ay=G.nodes[edge[0]]['pos'][1],# + G.nodes[edge[1]]['pos'][1]) / 2, 
                                    axref='x', ayref='y',
                                    x=G.nodes[edge[1]]['pos'][0]/1.003,# * 3 + G.nodes[edge[0]]['pos'][0]) / 4,
                                    y=G.nodes[edge[1]]['pos'][1]/1.003,# * 3 + G.nodes[edge[0]]['pos'][1]) / 4,
                                    xref='x', yref='y',
                                    showarrow=True,
                                    arrowhead=3,
                                    arrowsize=3,
                                    arrowwidth=1,
                                    opacity=1, arrowcolor = '#65696e', standoff=20, startstandoff=16
                                ) for edge in G.edges]
                            )}
    return figure

def network_emp(yearRange, AccountToSearch):
    adj_matrix, edge2, node2 = edge_node('EmailAddress', 'To', 'From')
    return network_graph(adj_matrix, edge2, node2, yearRange, AccountToSearch)

def network_dep(yearRange, AccountToSearch):
    adj_matrix, edge2, node2 = edge_node('CurrentEmploymentType', 'DepTo', 'DepFrom')
    return network_graph(adj_matrix, edge2, node2, yearRange, AccountToSearch)

def dotplotgraph(email_df): ##cbd3dd
    fig=px.scatter(email_df, x='DateTime', y='clean_name_from',hover_name='Subject', 
                                                        hover_data=["nr_recipients"], color = 'DepFrom' ) #change hover column names
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
                            
                            Choose the day you are intererested in (all days are in Jan '14).
                            """),style = {'font-size': 16, "color": '#FEFEFE'}),
                    html.Div( 
                        className="twelve columns",
                        children=[
                            dcc.RadioItems(id='my-range-slider', options = [6,7,8,9,10,13,14,15,16,17], value = 6),

                            html.Br(),
                            html.Div(id='output-container-range-slider')
                        ],
#                         style={'height': '300px', 'text-align': 'left','position':'relative','left':50}
                    ),
                    html.Div(
                        className="twelve columns",
                        children=[ 
                            dcc.Markdown(d("""
                            ** **
                            **Employee Email To Search**
                            
                            Choose the email to highlight the employee node.
                            """),style = {'font-size': 16, "color": '#FEFEFE'}),
                            dcc.Dropdown(id="input1",
                            options = employee_recs['EmailAddress'], #adj_matrix.columns,
                            value = employee_recs['EmailAddress'][0], 
                            clearable=False, style = {'background-color': '#FEFEFE', "color": '#56595C'}),
                            html.Div(id="output")
                        ],
                    )
                ],style={'height': '300px', 'text-align': 'left','position':'relative', 'left':6}
            ),

            # display the graph component
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph",
                                    figure=network_emp(YEAR, ACCOUNT))], style ={'text-align': 'left','position':'relative', 'left':100}
                ),
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph2",
                                    figure=network_dep(YEAR, ACCOUNT))], style ={'text-align': 'left','position': 'relative', 'left':100}
                ),
            html.Div(),
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph-dotplot",
                                    # figure=px.scatter(email_df, x='DateTime', y='clean_name_from',hover_name='Subject', 
                                    #                     hover_data=["nr_recipients"], color = 'DepFrom')
                                    figure = dotplotgraph(email_df))], style ={'text-align': 'center','position': 'relative', 'width':1500}
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
# callback for left side components of the network graph
@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value'), dash.dependencies.Input('input1', 'value')])
def update_output(value,input1):
    # to update the global variable of YEAR and ACCOUNT
    YEAR = value
    ACCOUNT = input1
    return network_emp(value, input1)

@app.callback(
    dash.dependencies.Output('my-graph2', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value'), dash.dependencies.Input('input1', 'value')])
def update_output(value,input1):
    # to update the global variable of YEAR and ACCOUNT
    YEAR = value
    ACCOUNT = input1
    return network_dep(value, input1)
    

if __name__=='__main__':
    app.run_server(debug=True, port=3000)
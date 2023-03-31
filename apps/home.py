import dash_html_components as html
import dash_bootstrap_components as dbc

# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to the GAStech Visual Analytics tool of group 3", className="text-center")
                    , className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='You can find the complete data of the GAStech Disappearance Challenge by clicking at the link below.'
                                     )
                    , className="mb-4")
            ]),

        dbc.Row([
            dbc.Col(html.H5(children='The tool consists of three main pages: Home tab, which is an introduction page to the visual analytics tool. Exploration tab about the email correspondence between employees, which gives the oppurtunity to find interesting patterns as well as a third page which allows the user to get a deeper insight into the general public opinion of GASTech by a detailed analysis on many news articles.')
                    , className="mb-5")
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H3(children='Get the complete data used in this project',
                                               className="text-center"),
                                       dbc.Button("GAStech Disappearance",
                                                  href="https://openscholarship.wustl.edu/cgi/viewcontent.cgi?filename=1&article=2189&context=cse_research&type=additional",
                                                  color="primary",
                                                  target="_blank",
                                                  className="mt-3")
                                       ],
                             body=True, color="dark", outline=True)
                    , width=6, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='You can find the code for this project in',
                                               className="text-center"),
                                       dbc.Button("GitHub",
                                                  href="https://github.com/YanasGH/VA/",
                                                  color="primary",
                                                  target="_blank",
                                                  className="mt-3"),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=6, className="mb-4")
        ], className="mb-5")

    ])

])

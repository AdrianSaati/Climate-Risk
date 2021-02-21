import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.express as px
import dash_table
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import Contours, Histogram2dContour, Marker, Scatter
import plotly.graph_objects as go
import dash_pivottable as dp
import dash_table as dt
from dash.dependencies import Input, Output

trdf = pd.read_excel('DF.xlsx', sheet_name=1)

template = 'simple_white'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#DIV.MAIN
app.layout = html.Div([
        
        html.Div([
          dcc.Graph(
                    id = 'distributionOfCurrentEPC',
                    # figure = distributionOfCurrentEPC,
            ),
        ]),

        dt.DataTable(
                      id='distributionOfCurrentEPC1',
                      columns=[{"name": i, "id": i} for i in trdf[['EPC','Risk']].groupby('EPC').count().reset_index().rename(columns={'Risk': 'No. of Properties'}).columns],
                      data=trdf[['EPC','Risk']].groupby('EPC').count().reset_index().rename(columns={'Risk': 'No. of Properties'}).to_dict('records'),
                    ),

        html.Br(),
        html.Br(),
        html.Br(),

        html.Div([
          dcc.Dropdown(
            id = 'tr_epcDist',
            options = [
              {'label': 'Low risk', 'value':'Low'},
              {'label': 'Medium risk', 'value':'Medium'},
              {'label': 'High risk', 'value':'High'},
            ],
            # value = ['Low'],
            # multi = True,
          ),
        ]),
])
#---------------------------------------------------------------------------------------------------------------
@app.callback(
    Output('distributionOfCurrentEPC','figure'),
    Input('tr_epcDist', 'value'),
    
)
def tf_distEPC(risk):
    filtered_epcDist = trdf[trdf.Risk == risk]

    graph_distributionOfCurrentEPC = px.bar(
      filtered_epcDist,
      x='EPC',
      color='EPC',
      title="EPC distribution",
      category_orders={"EPC": ["A", "B", "C", "D","E","F","G"]},
      labels={"Risk": "Risk"},
      # orientation='h'
      )

    # graph_distributionOfCurrentEPC.update_layout(transition_duration=500)
    return (graph_distributionOfCurrentEPC)

if __name__ == '__main__':
    app.run_server()
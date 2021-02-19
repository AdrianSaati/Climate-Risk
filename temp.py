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

############################################

# Importing the datasets
trdf = pd.read_excel('DF.xlsx',sheet_name=1)
prdf = pd.read_excel('DF.xlsx',sheet_name=0)
# print(trdf.head())
############################################

# Creating graphs

template = 'simple_white'
# template = 'plotly_dark'
# template = 'presentation'
# template = 'xgridoff'
# template = 'xgridoff'
# template = 'plotly_white'





## Transition Risk
#df_tr - prepared trdf dataset to use in dash pivot table
df_tr = pd.read_excel('DF.xlsx',sheet_name=1,header=None)
#Zmniejszenie długości etykiet zeby pivot table zmiescila sie na ekranie
df_tr.rename(columns={'Property value_AfterChangedEPCUnderParisScenario':'Changed property value'}, inplace = True)
df_tr.rename(columns={'LGD_AfterChangedEPCUnderParisScenario':'Changed LGD'}, inplace = True)
df_tr = df_tr.values.tolist()


## Physical Risk
#df_pr - prepared trdf dataset to use in dash pivot table
df_pr = pd.read_csv('JBA.csv',header=None)
df_pr = df_pr.values.tolist()



### Primary Portfolio Metrics

#### Grouped by Low/Med/High risk categories

##### Number of properties
# creating temp. df showing amount of properties for each risk
df_tr_noOfProperties = trdf['Risk'].value_counts()
df_tr_noOfProperties = df_tr_noOfProperties.to_frame()
df_tr_noOfProperties.reset_index(level=0,inplace=True)
df_tr_noOfProperties.rename(columns={'Risk':'Properties'},inplace=True)
df_tr_noOfProperties.rename(columns={'index':'Risk'},inplace=True)
tr_numberOfProperties1 = px.pie(df_tr_noOfProperties,values='Properties',names='Risk',template=template, title="Number of Properties")
tr_numberOfProperties2 = px.histogram(trdf,x='Risk',template=template, title="Number of Properties")
#metric
df_tr_noOfProperties = df_tr_noOfProperties










##### Total outstanding loans
# creating temp. df with cumulation of outstanding loan for each risk in transition risk
df_tr_TotalOutstandingLoans = pd.DataFrame(trdf.groupby('Risk')['Outstanding loan'].sum()).reset_index()
# final graph of total outstanding loans for each risk in transition risk
tr_TotalOutstandingLoans = px.pie(df_tr_TotalOutstandingLoans,names='Risk',values='Outstanding loan',template=template, title="Total Outstanding Loans")
#metric
df_tr_TotalOutstandingLoans.rename(columns = {'Outstanding loan':'GBP'},inplace = True)
df_tr_TotalOutstandingLoans = df_tr_TotalOutstandingLoans









##### Maximum/range of financial impact (e.g. RWA / ECL)
tr_financialImpact = px.box(trdf, x="Risk", y="ECL", points="all",template=template, title="Financial Impact")
# ECL min values
df_tr_financialImpact_min = pd.DataFrame(trdf.groupby('Risk')['ECL'].min()).reset_index().rename(columns={'ECL':'min ECL'})
# ECL max values
df_tr_financialImpact_max = pd.DataFrame(trdf.groupby('Risk')['ECL'].max()).reset_index().rename(columns={'ECL':'max ECL'})
# metric ECL min values
df_tr_financialImpact = pd.merge(left = df_tr_financialImpact_min, right = df_tr_financialImpact_max, how='inner')
df_tr_financialImpact['Range of ECL'] = round(df_tr_financialImpact['max ECL'] - df_tr_financialImpact['min ECL'],2)
df_tr_financialImpact = df_tr_financialImpact







##### Distribution of current EPC
# creating additional column with '1' in each row to count occurances of each row
cc = []
for i in range(trdf.count()[0]):
  cc.append(1)
trdf['Control Column'] = cc
# creating df with summed EPC occurances
df_distributionOfCurrentEPC = pd.DataFrame(trdf.groupby('EPC')['Control Column'].count()).reset_index()
# dodać nazwy axis i tytuł
distributionOfCurrentEPC = px.bar(df_distributionOfCurrentEPC, x="EPC", y='Control Column', category_orders={"EPC": ["A", "B", "C", "D","E","F","G"]}, 
                          labels={"Risk": "Risk"},template=template, title="Distribution of Current EPC")
distributionOfCurrentEPC.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'})
# metric
df_distributionOfCurrentEPC.rename(columns={'Control Column':'No. of Properties'}, inplace=True)
df_distributionOfCurrentEPC.rename(columns={'EPC':'EPC Class'}, inplace=True)
df_distributionOfCurrentEPC = df_distributionOfCurrentEPC
# trdf metric
# trdf[['EPC','Control Column']].groupby('EPC').sum().reset_index()




### Exposure Impacts

#### Data generated at customer level per scenario

##### GBP cost of property refurbishment to align with net zero scenario
# assumed that "Cost" column is the cost of net zero refurbishment. Creating a graph presenting cost of refurbishment distribution split into scenarios
costOfPropertyRef = px.histogram(trdf, x='Cost', marginal='rug',nbins=20, color='Scenario',template=template, title="Cost of Property Refurbishment")
# metric
# The only possible way to create metrics of above is to sum up all properties Cost of refurbishment per scenario
df_costOfPropertyRef = trdf[['Cost','Scenario']]
df_costOfPropertyRef = df_costOfPropertyRef.groupby('Scenario').sum().reset_index()
df_costOfPropertyRef = df_costOfPropertyRef









##### Property refurbishment cost as % of property value
# temp df. with 3 values
df_refCostAsPercentOfPropertyValue = trdf[['Cost','Property value','Scenario']]
# temp df. and additional column with refurbishment and property value ratio rounded to 2 decimal places
df_refCostAsPercentOfPropertyValue['Cost as percent of property value'] = df_refCostAsPercentOfPropertyValue.apply(lambda x: round(x[0]/x[1]*100,10), axis=1)
# final scatter plot rug shows density, size shows property value, y axis shows cost of refurbishment, x axis shows Cost to Property Value ratio, Color shows the scenario
refCostAsPercentOfPropertyValue = px.scatter(df_refCostAsPercentOfPropertyValue, x="Cost as percent of property value", y='Cost',
	         size="Property value", color="Scenario",
          log_x=True, size_max=60, marginal_x = 'rug',
          template=template, title="Refurbishment Cost as Percent of Property Value")
# metric
# the only way to create metric is to sum all costs and property values and count percent of summed variables
metric_df_refCostAsPercentOfPropertyValue = df_refCostAsPercentOfPropertyValue[['Cost', 'Property value', 'Scenario']].groupby('Scenario').sum()
metric_df_refCostAsPercentOfPropertyValue[r'Property refurbishment cost as % of property value'] = metric_df_refCostAsPercentOfPropertyValue['Cost']/metric_df_refCostAsPercentOfPropertyValue['Property value']*100
metric_df_refCostAsPercentOfPropertyValue.rename(columns={'Cost':'Summed Properties Refurbishment Cost'}, inplace = True)
metric_df_refCostAsPercentOfPropertyValue.rename(columns={'Property value':'Summed Properties Value'}, inplace = True)
metric_df_refCostAsPercentOfPropertyValue.rename(columns={r'Property refurbishment cost as % of property value':r'Summed Properties Refurbishment Cost as % of Summed Properties Value'}, inplace = True)
metric_df_refCostAsPercentOfPropertyValue.reset_index(inplace=True)
metric_df_refCostAsPercentOfPropertyValue = metric_df_refCostAsPercentOfPropertyValue







##### Annualised refurbishment cost as % of mortgage payment
#formula: Total Refurbishment Cost / (Monthly mortgage payment * 12)
# temp. df with cost, mmp and scenario columns
df_annualisedRefCostAsPercentOfMortgagePayment = trdf[['Cost','Monthly mortgage payment','Scenario']]
# temp list cost / (mmp*12)
list_cost = trdf['Cost'].tolist()
list_mmp = trdf['Monthly mortgage payment'].tolist()
list_annualisedRefCostAsPercOfMortgagePayment = []
for i in range(len(list_cost)):
  list_annualisedRefCostAsPercOfMortgagePayment.append(list_cost[i]/(list_mmp[i]*12))
# add list to temp. df
df_annualisedRefCostAsPercentOfMortgagePayment['Annual. Ref. Cost as Percent of Mortgage Payment'] = list_annualisedRefCostAsPercOfMortgagePayment
# final plot
annualisedRefCostAsPercentOfMortgagePayment = px.histogram(df_annualisedRefCostAsPercentOfMortgagePayment, x="Annual. Ref. Cost as Percent of Mortgage Payment", color='Scenario', marginal='rug', template=template, title="Annualised Refurbishment Cost as Percent of Mortgage Payment" ) 
# metric

metric_df_annualisedRefCostAsPercentOfMortgagePayment = trdf.groupby('Scenario').sum()[['Cost','Monthly mortgage payment']].reset_index()
metric_df_annualisedRefCostAsPercentOfMortgagePayment['Summed Annualised cost percent of mortgage'] = (metric_df_annualisedRefCostAsPercentOfMortgagePayment['Cost']/(metric_df_annualisedRefCostAsPercentOfMortgagePayment['Monthly mortgage payment']*12))*100
metric_df_annualisedRefCostAsPercentOfMortgagePayment.rename(columns={'Cost':'Sum. Properties Ref. Costs'}, inplace = True)
metric_df_annualisedRefCostAsPercentOfMortgagePayment.rename(columns={'Monthly mortgage payment':'Summed yr. payments'}, inplace = True)
metric_df_annualisedRefCostAsPercentOfMortgagePayment['Summed Annualised cost percent of mortgage'] = metric_df_annualisedRefCostAsPercentOfMortgagePayment['Summed Annualised cost percent of mortgage'].apply(lambda x: round(x,2))
metric_df_annualisedRefCostAsPercentOfMortgagePayment










##### Property valuation haircut
propertyValuationHaircut1 = px.histogram(trdf, x="Property valuation haircut", color='Scenario', barmode='group',template=template, title="Property Valuation Haircut") 
propertyValuationHaircut2 = px.scatter(trdf, x="Property valuation haircut", hover_data=['Property value', 'Property valuation haircut','ECL'], color='Scenario',  marginal_x='rug', size='ECL', template=template, title="Property Valuation Haircut") 
#metric
#average property valuation haircut
metric_propertyValuationHaircut = trdf[['Scenario','Property valuation haircut']]
metric_propertyValuationHaircut = metric_propertyValuationHaircut.groupby('Scenario').mean().reset_index()
metric_propertyValuationHaircut.rename(columns={'Property valuation haircut':'Avg. percent of property valuation haircut'}, inplace=True)
metric_propertyValuationHaircut['Avg. percent of property valuation haircut'] = metric_propertyValuationHaircut['Avg. percent of property valuation haircut'].apply(lambda x: round(x,2))
metric_propertyValuationHaircut








#### Cross-scenario customer metrics used in underwriting

###### Risk rating = High/Med/Low
x = trdf.Risk.values.tolist()
y = trdf.Scenario.values.tolist()
# RiskRating1 = iplot([Histogram2dContour(x=x, y=y, contours=Contours(coloring='heatmap')),
#        Scatter(x=x, y=y, mode='markers', marker=Marker(color='white', size=1, opacity=0.3))], show_link=False)
RiskRating1 = px.density_contour(trdf,x='Risk',y='Scenario',template=template, title="Risk Rating")
RiskRating1.update_traces(contours_coloring="fill", contours_showlabels = True)
#metric transformative risk
metric_df_tr_riskRating_TransChange = trdf[['Scenario','Risk']]
metric_df_tr_riskRating_TransChange = metric_df_tr_riskRating_TransChange[metric_df_tr_riskRating_TransChange['Scenario'] == 'Transformative change']
metric_df_tr_riskRating_TransChange = metric_df_tr_riskRating_TransChange.groupby('Risk').count().reset_index()
metric_df_tr_riskRating_TransChange.rename(columns = {'Scenario':'Counts'},inplace = True)
metric_df_tr_riskRating_TransChange['Scenario'] = 'Transformative change'
metric_df_tr_riskRating_TransChange = metric_df_tr_riskRating_TransChange[['Scenario','Risk','Counts']]
metric_df_tr_riskRating_TransChange
#metric high ambition risk
metric_df_tr_riskRating_HighAmbition = trdf[['Scenario','Risk']]
metric_df_tr_riskRating_HighAmbition = metric_df_tr_riskRating_HighAmbition[metric_df_tr_riskRating_HighAmbition['Scenario'] == 'High ambition']
metric_df_tr_riskRating_HighAmbition = metric_df_tr_riskRating_HighAmbition.groupby('Risk').count().reset_index()
metric_df_tr_riskRating_HighAmbition.rename(columns = {'Scenario':'Counts'},inplace = True)
metric_df_tr_riskRating_HighAmbition['Scenario'] = 'High ambition'
metric_df_tr_riskRating_HighAmbition = metric_df_tr_riskRating_HighAmbition[['Scenario','Risk','Counts']]
metric_df_tr_riskRating_HighAmbition
#metric final
metric_df_tr_riskRating = metric_df_tr_riskRating_TransChange.append(metric_df_tr_riskRating_HighAmbition, ignore_index=True)
metric_df_tr_riskRating = metric_df_tr_riskRating.groupby(['Scenario','Risk']).sum()
metric_df_tr_riskRating = metric_df_tr_riskRating.reset_index()









###### Maximum Annualised Refurbishment Cost Across Scenarios
tr_maximumAnnualisedRefCostAcrossScenarios = px.box(trdf,x='Scenario',y='Cost',template=template, title="Maximum Annualised Refurbishment Cost Across Scenarios")
#metric
metric_df_maxAnnualRefCostCrossScenarios = trdf[['Scenario','Cost']]
metric_df_maxAnnualRefCostCrossScenarios = metric_df_maxAnnualRefCostCrossScenarios.groupby('Scenario').max().reset_index()
metric_df_maxAnnualRefCostCrossScenarios.rename(columns={'Cost':'Max. Refurbishment Cost GBP'}, inplace = True)
metric_df_maxAnnualRefCostCrossScenarios = metric_df_maxAnnualRefCostCrossScenarios







####################################
####################################
####################################

## Dataset
trdf = pd.read_excel('DF.xlsx' ,sheet_name=1)



app = dash.Dash(__name__)

#DIV.MAIN
app.layout = html.Div([
        dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Transition Risk', value='tab-1'),
        dcc.Tab(label='Physical Risk', value='tab-2'),
    ]),
    html.Div(id='tabs-example-content')
])

@app.callback(Output('tabs-example-content', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(children=[

    #DIV.TRANSITION
    html.Div(children=[

        #DIV.1
        html.Div(children=[
            html.H1('Transition Risk'),
            
              dp.PivotTable(
                            data=df_tr,
                            cols=["Risk"],
                            aggregatorName = "Count",
                            rendererName = "Grouped Column Chart",
                          # rows=["Risk"],
                          # vals=["Count"]
                    ),
            ],className = 'Pivot'),

        #DIV.2
        html.Div(children=[

            #DIV.2.1
            html.Div(children=[
                html.H2('Primary portfolio metrics'),
            ]),

            #DIV.2.2
            html.Div(children=[

                #DIV.2.2.1
                html.Div(children=[
                    html.H3('Distribution of current EPC'),

                    # ##### Distribution of current EPC
                    dcc.Graph(
                        id = 'distributionOfCurrentEPC',
                        figure = distributionOfCurrentEPC,
                    ),
                    dt.DataTable(
                                    id='table_df_distributionOfCurrentEPC',
                                    columns=[{"name": i, "id": i} for i in df_distributionOfCurrentEPC.columns],
                                    data=df_distributionOfCurrentEPC.to_dict('records'),
                    ),
                    # DCC

                    html.Br(),
                    html.Br(),
                    html.Br(),

                    dcc.RangeSlider(
                    id='range-slider',
                    # min=0,
                    # max=20,
                    # step=0.5,
                    # value=[5, 15]
                    ),
                    html.Div(id='output-container-range-slider'),

  
                ], className = 'six columns'),

                #DIV.2.2.2
                html.Div(children=[
                    html.H3('Financial impact (ECL)'),
                    # ##### Maximum/range of financial impact (e.g. RWA / ECL)
                    dcc.Graph(
                        id = 'tr_financialImpact',
                        figure = tr_financialImpact,
                    ),

                    dt.DataTable(
                                    id='table_df_tr_financialImpact',
                                    columns=[{"name": i, "id": i} for i in df_tr_financialImpact.columns],
                                    data=df_tr_financialImpact.to_dict('records'),
                    ),

                    #jeszcze do dodania są tabele z min ECL i z max ECL
                ], className = 'six columns'),
            ], className = 'row'),

            #DIV.2.3
            html.Div(children=[

                # DIV.2.3.1
                html.Div(children=[
                    html.H3('Total outstanding loans'),
                    # ##### Total outstanding loans
                    dcc.Graph(
                        id = 'tr_TotalOutstandingLoans',
                        figure = tr_TotalOutstandingLoans,
                    ),

                    dt.DataTable(
                                    id='table_df_tr_TotalOutstandingLoans',
                                    columns=[{"name": i, "id": i} for i in df_tr_TotalOutstandingLoans.columns],
                                    data=df_tr_TotalOutstandingLoans.to_dict('records'),
                    )

                ], className= 'six columns'),

                # DIV.2.3.2
                html.Div(children=[
                    html.H3('Number of properties'),
                    # dp.PivotTable(
                    #                 data=df_tr,
                    #                 cols=["Risk"],
                    #                 aggregatorName = "Count",
                    #                 rendererName = "Grouped Column Chart",
                    #                 # rows=["Risk"],
                    #                 # vals=["Count"]
                    # ),
                     ##### Number of properties
                    dcc.Graph(
                                    id = 'tr_numberOfProperties1',
                                    figure = tr_numberOfProperties1,
                    ),

                    dt.DataTable(
                                    id='table_numberOfProperties1',
                                    columns=[{"name": i, "id": i} for i in df_tr_noOfProperties.columns],
                                    data=df_tr_noOfProperties.to_dict('records'),
                    )

                ], className = 'six columns'),

            ], className = 'row'),
        ]),

        #DIV.3
        html.Div(children=[
            #DIV.3.1
            html.Div(children=[
                html.H2('Exposure Impacts')
            ]),

            #DIV.3.2
            html.Div(children=[

                #DIV.3.2.1
                html.Div(children=[
                    html.H3('GBP cost of property refurbishment to align with net zero scenario'),
                     # ##### GBP cost of property refurbishment to align with net zero scenario
                     dcc.Graph(
                         id = 'costOfPropertyRef',
                         figure = costOfPropertyRef,
                     ),

                     dt.DataTable(
                                    id='table_df_costOfPropertyRef',
                                    columns=[{"name": i, "id": i} for i in df_costOfPropertyRef.columns],
                                    data=df_costOfPropertyRef.to_dict('records'),
                    )
                ], className='six columns'),

                # DIV.3.2.2
                html.Div(children=[
                    html.H3('Property refurbishment cost as percent of property value'),
                    # ##### Property refurbishment cost as % of property value
                     dcc.Graph(
                        id = 'refCostAsPercentOfPropertyValue',
                        figure = refCostAsPercentOfPropertyValue,
                    ),

                    dt.DataTable(
                                    style_cell={
                                                'whiteSpace': 'normal',
                                                'height': 'auto',
                                                'overflow': 'hidden',
                                                'textOverflow': 'ellipsis',
                                                'maxWidth': 0
                                              },
                                    id='table_metric_df_refCostAsPercentOfPropertyValue',
                                    columns=[{"name": i, "id": i} for i in metric_df_refCostAsPercentOfPropertyValue.columns],
                                    data=metric_df_refCostAsPercentOfPropertyValue.to_dict('records'),
                    )
                ], className='six columns'),
            ],className = 'row'),

            #DIV.3.3
            html.Div(children=[

                #DIV.3.3.1
                html.Div(children=[
                    html.H3('Annualised refurbishment cost as percent of mortgage payment'),
                    # ##### Annualised refurbishment cost as % of mortgage payment
                    dcc.Graph(
                        id = 'annualisedRefCostAsPercentOfMortgagePayment',
                        figure = annualisedRefCostAsPercentOfMortgagePayment,
                    ),

                    dt.DataTable(
                                    id='table_metric_df_annualisedRefCostAsPercentOfMortgagePayment',
                                    columns=[{"name": i, "id": i} for i in metric_df_annualisedRefCostAsPercentOfMortgagePayment.columns],
                                    data=metric_df_annualisedRefCostAsPercentOfMortgagePayment.to_dict('records'),
                    ),

                ], className='six columns'),

                # DIV.3.3.2
                html.Div(children=[
                    html.H3('Property valuation haircut'),
                        ##### Property valuation haircut
                        dcc.Graph(
                            id = 'propertyValuationHaircut1',
                            figure = propertyValuationHaircut1,
                        ),
                        # dcc.Graph(
                        #     id = 'propertyValuationHaircut2',
                        #     figure = propertyValuationHaircut2,
                        # ),

                        dt.DataTable(
                                    id='table_metric_propertyValuationHaircut',
                                    columns=[{"name": i, "id": i} for i in metric_propertyValuationHaircut.columns],
                                    data=metric_propertyValuationHaircut.to_dict('records'),
                    ),

                ], className='six columns'),
            ], className= 'row'),

            #DIV.3.4
            html.Div(children=[

                #DIV.3.4.1
                html.Div(children=[
                    html.H3('Risk rating High/Mid/Low'),
                    #### Risk rating = High/Med/Low
                    dcc.Graph(
                        id = 'RiskRating1',
                        figure = RiskRating1,
                    ),

                    dt.DataTable(
                                    id='table_metric_df_tr_riskRating',
                                    columns=[{"name": i, "id": i} for i in metric_df_tr_riskRating.columns],
                                    data=metric_df_tr_riskRating.to_dict('records'),
                    ),
                ], className='six columns'),

                # DIV.3.4.2
                html.Div(children=[
                    html.H3('Maximum GBP annualised refurbishment cost across scenarios'),
                    #### Maximum Annualised Refurbishment Cost Across Scenarios
                    dcc.Graph(
                        id = 'tr_maximumAnnualisedRefCostAcrossScenarios',
                        figure = tr_maximumAnnualisedRefCostAcrossScenarios,
                    ),

                    dt.DataTable(
                                    id='table_metric_df_maxAnnualRefCostCrossScenarios',
                                    columns=[{"name": i, "id": i} for i in metric_df_maxAnnualRefCostCrossScenarios.columns],
                                    data=metric_df_maxAnnualRefCostCrossScenarios.to_dict('records'),
                    ),

                ], className='six columns'),
            ], className = 'row'),
        ]),

        # #DIV.4
        # html.Div(children=[

        #     #DIV.4.1
        #     html.Div(children=[
        #         html.H2('Map')
        #     ]),

        #     #DIV.4.2
        #     html.Div(children=[

        #     ]),
        # ]),

        #DIV.5
        # html.Div(children=[

        #     #DIV.5.1
        #     html.Div(children=[
        #         html.H2('AIB Financial Impact')
        #     ]),

        #     #DIV.5.2
        #     html.Div(children=[

        #         #DIV.5.2.1
        #         html.Div(children=[
        #             html.H3('LGD')
        #         ]),

        #         #DIV.5.2.2
        #         html.Div(children=[
        #             html.H3('RWA')
        #         ]),

        #         #DIV.5.2.3
        #         html.Div(children=[
        #             html.H3('ECL')
        #         ]),
        #     ]),

        # ]),

    ]),

    #DIV.PHYSICAL
    html.Div(children=[

    ]),
]
            
        )
    elif tab == 'tab-2':
        return html.Div(children=[
            html.Div(children=[
            html.H1('Physical Risk'),
            
              dp.PivotTable(
                            data=df_pr,
                            cols=["Coastal_Floodscore_Def"],
                            rows=['LGD'],
                            aggregatorName = "Count",
                            rendererName = "Table Row Heatmap",
                          # rows=["Risk"],
                          # vals=["Count"]
                    ),
            ],className = 'Pivot'),
        ])
#Callback - konieczne do wprowadzenia logiki do layoutu, backend
#----------------

#Callback decorator
# @app.callback(
#     Output(component_id='my_graph', component_property='figure'),
#     Input(component_id='choice', component_property='value')
# )
# def interactive_graphing(value_choice):
#     print(value_choice)
#     dff = trdf[trdf['CostofFloodRisk_2day'] == value_choice]
#     fig = px.histogram(data_frame=dff, x = 'Scenario', y ='Cost')
#     return fig

app.css.append_css({
    "external_url":'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

#uruchamianie serwera pod linkiem localhost
if __name__=='__main__':
    app.run_server()
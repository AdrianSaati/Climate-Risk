# Libs
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

############################################

# Importing the datasets
trdf = pd.read_excel('DF.xlsx',sheet_name=1)
# print(trdf.head())
############################################

# Creating graphs

template = 'plotly_dark'

## Transition Risk
#df_tr - prepared trdf dataset to use in dash pivot table
df_tr = pd.read_excel('DF.xlsx',sheet_name=1,header=None)
df_tr = df_tr.values.tolist()

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







##### Total outstanding loans
# creating temp. df with cumulation of outstanding loan for each risk in transition risk
df_tr_TotalOutstandingLoans = pd.DataFrame(trdf.groupby('Risk')['Outstanding loan'].sum()).reset_index()
# final graph of total outstanding loans for each risk in transition risk
tr_TotalOutstandingLoans = px.pie(df_tr_TotalOutstandingLoans,names='Risk',values='Outstanding loan',template=template, title="Total Outstanding Loans")



##### Maximum/range of financial impact (e.g. RWA / ECL)
tr_financialImpact = px.box(trdf, x="Risk", y="ECL", points="all",template=template, title="Financial Impact")
# ECL min values
df_tr_financialImpact_min = pd.DataFrame(trdf.groupby('Risk')['ECL'].min()).reset_index().rename(columns={'ECL':'min ECL'})
# ECL max values
df_tr_financialImpact_max = pd.DataFrame(trdf.groupby('Risk')['ECL'].max()).reset_index().rename(columns={'ECL':'max ECL'})



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



### Exposure Impacts

#### Data generated at customer level per scenario

##### GBP cost of property refurbishment to align with net zero scenario
# assumed that "Cost" column is the cost of net zero refurbishment. Creating a graph presenting cost of refurbishment distribution split into scenarios
costOfPropertyRef = px.histogram(trdf, x='Cost', marginal='rug',nbins=20, color='Scenario',template=template, title="Cost of Property Refurbishment")



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



##### Property valuation haircut
propertyValuationHaircut1 = px.histogram(trdf, x="Property valuation haircut", color='Scenario', barmode='group',template=template, title="Property Valuation Haircut") 
propertyValuationHaircut2 = px.scatter(trdf, x="Property valuation haircut", hover_data=['Property value', 'Property valuation haircut','ECL'], color='Scenario',  marginal_x='rug', size='ECL', template=template, title="Property Valuation Haircut") 



#### Cross-scenario customer metrics used in underwriting

###### Risk rating = High/Med/Low
x = trdf.Risk.values.tolist()
y = trdf.Scenario.values.tolist()
# RiskRating1 = iplot([Histogram2dContour(x=x, y=y, contours=Contours(coloring='heatmap')),
#        Scatter(x=x, y=y, mode='markers', marker=Marker(color='white', size=1, opacity=0.3))], show_link=False)
RiskRating1 = px.density_contour(trdf,x='Risk',y='Scenario',template=template, title="Risk Rating")
RiskRating1.update_traces(contours_coloring="fill", contours_showlabels = True)



###### Maximum Annualised Refurbishment Cost Across Scenarios
tr_maximumAnnualisedRefCostAcrossScenarios = px.box(trdf,x='Scenario',y='Cost',template=template, title="Maximum Annualised Refurbishment Cost Across Scenarios")




####################################
####################################
####################################




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# @import url(https://fonts.googleapis.com/css?family=Roboto);

app = dash.Dash(
      __name__,
    external_stylesheets=external_stylesheets
)

app.layout = html.Div([
    html.Div(children= 'Climate Risk',
    style={
      'textAlign':'left',
      'color':'#8359A3',
      # 'backgroundColor':'#0e0e0e',
      'font-size':'40px',
      'font_family': 'Roboto',
      'opacity':0.6,
      'padding':'15px',
    }),





    # html.Div(children = 'Transition Risk',
    # style={
    #   'textAlign':'left',
    #   'color':'#8800FF',
    #   # 'backgroundColor':'#0e0e0e',
    #   'font-size':'30px',
    #   'font_family': 'Roboto',
    #   'opacity':0.9,
    #   'padding':'8px',
    #   'padding-left':'15px',
    # }),
    # html.Div(),

    ##############################################
    # Displaying Graphs

    ## Transition Risk
    
    # ##### Number of properties
    # dcc.Graph(
    #     id = 'tr_numberOfProperties1',
    #     figure = tr_numberOfProperties1,
      
    # ),
    # dcc.Graph(
    #     id = 'tr_numberOfProperties2',
    #     figure = tr_numberOfProperties2,
    # ),

    # ##### Total outstanding loans
    # dcc.Graph(
    #     id = 'tr_TotalOutstandingLoans',
    #     figure = tr_TotalOutstandingLoans,
    # ),

    # ##### Maximum/range of financial impact (e.g. RWA / ECL)
    # dcc.Graph(
    #     id = 'tr_financialImpact',
    #     figure = tr_financialImpact,
    # ),
    # #jeszcze do dodania są tabele z min ECL i z max ECL

    # ##### Distribution of current EPC
    # dcc.Graph(
    #     id = 'distributionOfCurrentEPC',
    #     figure = distributionOfCurrentEPC,
    # ),

    # ##### GBP cost of property refurbishment to align with net zero scenario
    # dcc.Graph(
    #     id = 'costOfPropertyRef',
    #     figure = costOfPropertyRef,
    # ),

    # ##### Property refurbishment cost as % of property value
    #  dcc.Graph(
    #     id = 'refCostAsPercentOfPropertyValue',
    #     figure = refCostAsPercentOfPropertyValue,
    # ),

    # ##### Annualised refurbishment cost as % of mortgage payment
    # dcc.Graph(
    #     id = 'annualisedRefCostAsPercentOfMortgagePayment',
    #     figure = annualisedRefCostAsPercentOfMortgagePayment,
    # ),

    # ##### Property valuation haircut
    # dcc.Graph(
    #     id = 'propertyValuationHaircut1',
    #     figure = propertyValuationHaircut1,
    # ),
    # dcc.Graph(
    #     id = 'propertyValuationHaircut2',
    #     figure = propertyValuationHaircut2,
    # ),

    # #### Risk rating = High/Med/Low
    # dcc.Graph(
    #     id = 'RiskRating1',
    #     figure = RiskRating1,
    # ),

    # #### Maximum Annualised Refurbishment Cost Across Scenarios
    # dcc.Graph(
    #     id = 'tr_maximumAnnualisedRefCostAcrossScenarios',
    #     figure = tr_maximumAnnualisedRefCostAcrossScenarios,
    # ),
    
    # # dcc.Graph(
    # #     id = 'test',
    # #     figure = fig,
    # # ),

    html.Div(
    dp.PivotTable(
        'Number of Properties',
        data=df_tr,
        cols=["Risk"],
        aggregatorName = "Count",
        rendererName = "Grouped Column Chart",
        # rows=["Risk"],
        # vals=["Count"]
    )
)
])

if __name__ == '__main__':
    app.run_server(port = 4050)
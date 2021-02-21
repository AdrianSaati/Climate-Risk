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

# Importing the datasets Transition Risk
trdf = pd.read_excel('DF.xlsx',sheet_name=1)
## Transition Risk
#df_tr - prepared trdf dataset to use in dash pivot table
df_tr = pd.read_excel('DF.xlsx',sheet_name=1,header=None)
#Zmniejszenie długości etykiet zeby pivot table zmiescila sie na ekranie
df_tr.rename(columns={'Property value_AfterChangedEPCUnderParisScenario':'Changed property value'}, inplace = True)
df_tr.rename(columns={'LGD_AfterChangedEPCUnderParisScenario':'Changed LGD'}, inplace = True)
df_tr = df_tr.values.tolist()


# Importing the datasets Physical Risk
prdf = pd.read_excel('DF.xlsx',sheet_name=0)
physical_Risk_data = pd.read_excel('DF.xlsx',sheet_name=0,header=None)
physical_Risk_data = physical_Risk_data.values.tolist()



































# Creating graphs

template = 'simple_white'
# template = 'plotly_dark'
# template = 'presentation'
# template = 'xgridoff'
# template = 'xgridoff'
# template = 'plotly_white'













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
# # # creating additional column with '1' in each row to count occurances of each row
# # cc = []
# # for i in range(trdf.count()[0]):
# #   cc.append(1)
# # trdf['Control Column'] = cc
# # # creating df with summed EPC occurances
# # df_distributionOfCurrentEPC = pd.DataFrame(trdf.groupby('EPC')['Control Column'].count()).reset_index()
# # # dodać nazwy axis i tytuł
# # distributionOfCurrentEPC = px.bar(df_distributionOfCurrentEPC, x="EPC", y='Control Column', category_orders={"EPC": ["A", "B", "C", "D","E","F","G"]}, 
# #                           labels={"Risk": "Risk"},template=template, title="Distribution of Current EPC")
# # distributionOfCurrentEPC.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'})
# # metric
# df_distributionOfCurrentEPC.rename(columns={'Control Column':'No. of Properties'}, inplace=True)
# df_distributionOfCurrentEPC.rename(columns={'EPC':'EPC Class'}, inplace=True)
# # metric on
# #trdf[['EPC','Risk']].groupby('EPC').count().reset_index().rename(columns={'Risk': 'No. of Properties'})
# distributionOfCurrentEPC = px.bar(df_distributionOfCurrentEPC, x="No. of Properties", y='EPC Class', color='EPC Class', title="EPC distribution", category_orders={"EPC": ["A", "B", "C", "D","E","F","G"]}, labels={
#                      "Risk": "Risk"},orientation='h')
graph_distributionOfCurrentEPC = px.bar(trdf[['EPC','Risk']], y='EPC', color='EPC', title="EPC distribution", category_orders={"EPC": ["A", "B", "C", "D","E","F","G"]}, labels={
                     "Risk": "Risk"},orientation='h')





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



##### LTV
trdf['Loan to Value'] = trdf['Loan'] / trdf['Property value']
graph_tr_ltv = px.bar(trdf, x ='Risk', y='Loan to Value', color = 'Scenario')









##Pyhysical Risk

#####EXPOSURE CATEGORISATION
#Flood Risk Ascribing Function
def floodRisk(cols):
  # In20River2050= cols['1-in-20_river_2050'] 
  # In75Coastal2050 = cols['1-in-20_river_2050'] 
  # In200River2050 = cols['1-in-200_river_2050'] 
  # In200Coastal2050 = cols['1-in-200_coastal_2050'] 

  # temporary variable
  fs = cols['River_Floodscore_Def']

  # if In20River2050 == 1 or In75Coastal2050 == 1:
  #   return 'High'
  # elif In200Coastal2050 == 0 and In200River2050 == 0:
  #   return 'Low'
  # else:
  #   return 'Medium'

  if fs>15:
    return 'High'
  elif fs>10:
    return 'Medium'
  else:
    return 'Low'
#Creating Flood Risk column based on floodRisk function above
prdf['Flood Risk'] = prdf.apply(floodRisk, axis=1)



####BACKBOOK QUANTIFICATION
#PD
def propertyReduction(cols):

  insurancePremiumLoading = 0.5
  marketSentimentAdjustment = 0.2

  projectedInsurancePremium = cols['projected insurance premium 2050']
  currentInsurancePremium = cols['insurance premium current climate']
  risk = cols['Flood Risk']


  if risk=='High' or risk=='Medium':
     insuranceDiff = projectedInsurancePremium - currentInsurancePremium
     propertyReduc = insuranceDiff + insuranceDiff * insurancePremiumLoading + insuranceDiff * marketSentimentAdjustment
     return propertyReduc
  else:
     return 0

prdf['Property reduction'] = prdf.apply(propertyReduction, axis = 1)


graph_pr_backBook = px.scatter(prdf[['Flood Risk','Property reduction']],y = 'Property reduction', x = 'Flood Risk',color = 'Flood Risk')




















#Number of Properties
df_pr_noOfProperties = prdf['Risk'].value_counts()
df_pr_noOfProperties = df_pr_noOfProperties.to_frame()
df_pr_noOfProperties.reset_index(level=0,inplace=True)
df_pr_noOfProperties.rename(columns={'Risk':'Properties'},inplace=True)
df_pr_noOfProperties.rename(columns={'index':'Risk'},inplace=True)
pr_numberOfProperties1 = px.pie(df_pr_noOfProperties,values='Properties',names='Risk',template=template, title="Number of Properties")
pr_numberOfProperties2 = px.histogram(trdf,x='Risk',template=template, title="Number of Properties")
#metric
df_pr_noOfProperties = df_pr_noOfProperties





## Distribution of current flood score
graph_distributionOfCurrentFloodScore = px.bar(prdf[['Flood_Score','Risk']], y='Flood_Score', color='Flood_Score', title="Flood score distribution", labels={
                     "Risk": "Risk"},orientation='h')


##############################################
##############################################
##############################################
##############################################



#####GRAPHS AND DFS

###Number of Properties
# #df_pr - prepared trdf dataset to use in dash pivot table
# df_pr = pd.read_excel('DF.xlsx',sheet_name=0,header=None)




##### Maximum/range of financial impact (e.g. RWA / ECL)
pr_financialImpact = px.box(prdf, x="Risk", y="ECL", points="all",template=template, title="Financial Impact")
# ECL min values
df_pr_financialImpact_min = pd.DataFrame(prdf.groupby('Risk')['ECL'].min()).reset_index().rename(columns={'ECL':'min ECL'})
# ECL max values
df_pr_financialImpact_max = pd.DataFrame(prdf.groupby('Risk')['ECL'].max()).reset_index().rename(columns={'ECL':'max ECL'})
# metric ECL min values
df_pr_financialImpact = pd.merge(left = df_pr_financialImpact_min, right = df_pr_financialImpact_max, how='inner')
df_pr_financialImpact['Range of ECL'] = round(df_pr_financialImpact['max ECL'] - df_pr_financialImpact['min ECL'],2)
df_pr_financialImpact = df_pr_financialImpact




##### Total outstanding loans
# creating temp. df with cumulation of outstanding loan for each risk in transition risk
df_pr_TotalOutstandingLoans = pd.DataFrame(prdf.groupby('Risk')['Outstanding loan'].sum()).reset_index()
# final graph of total outstanding loans for each risk in transition risk
pr_TotalOutstandingLoans = px.pie(df_pr_TotalOutstandingLoans,names='Risk',values='Outstanding loan',template=template, title="Total Outstanding Loans")
#metric
df_pr_TotalOutstandingLoans.rename(columns = {'Outstanding loan':'GBP'},inplace = True)
df_pr_TotalOutstandingLoans = df_pr_TotalOutstandingLoans



##### GBP increase in insurance cost
prdf['increase in insurance'] = prdf['projected insurance premium 2050'] - prdf['insurance premium current climate']
df_pr_insInc = prdf[['Scenario','increase in insurance']].groupby('Scenario').mean().reset_index().rename(columns={'increase in insurance': 'Avg. insurance increase'})
df_pr_insInc
graph_pr_insInc = px.scatter(prdf, y='increase in insurance', color = 'Scenario', marginal_y='rug')


#####Present value of increased insurance cost as percent property value
prdf['future insurance to property val.'] = prdf['projected insurance premium 2050'] / prdf['Property value']
df_pr_FutInsuranceToPropertyVal = prdf[['Scenario','future insurance to property val.']].groupby('Scenario').mean().reset_index().rename(columns = {'future insurance to property val.':'Avg future insurance cost to property val.'})
graph_pr_FutInsuranceToPropertyVal = px.scatter(prdf,y='future insurance to property val.',color='Scenario', marginal_y='rug')



######## Increase in insurance cost as % of mortgage payment
prdf['inc.in insu. as % of mort. paym.'] = (prdf['projected insurance premium 2050'] - prdf['insurance premium current climate']) / prdf['Monthly mortgage payment']
df_pr_increaseInsurAsPercMortg = prdf[['Scenario','inc.in insu. as % of mort. paym.']].groupby('Scenario').mean().reset_index().rename(columns={'inc.in insu. as % of mort. paym.':"Avg. inc. in insu. as % of mort. paym."})
graph_pr_increaseInsurAsPercMortg = px.scatter(prdf[['Scenario','inc.in insu. as % of mort. paym.']], y='inc.in insu. as % of mort. paym.', color='Scenario', marginal_y='rug')




##### #### Property valuation haircut
#### BASED ON BACK BOOK QUANTIFICATION
#average property valuation haircut
metric_pf_propertyValuationHaircut = prdf[['Scenario','Property reduction']]
metric_pf_propertyValuationHaircut = metric_pf_propertyValuationHaircut.groupby('Scenario').mean().reset_index()
metric_pf_propertyValuationHaircut.rename(columns={'Property reduction':'Avg. percent of property valuation haircut'}, inplace=True)
metric_pf_propertyValuationHaircut
graph_pr_propVal = px.scatter(prdf, y='Property reduction', color='Scenario', marginal_y='rug')



#### Risk Rating = low/mid/hig
pr_RiskRating1 = px.density_contour(prdf,x='Risk',y='Scenario')
pr_RiskRating1.update_traces(contours_coloring="fill", contours_showlabels = True) 

#metric transformative risk
metric_df_pr_riskRating_TransChange = prdf[['Scenario','Risk']]
metric_df_pr_riskRating_TransChange = metric_df_pr_riskRating_TransChange[metric_df_pr_riskRating_TransChange['Scenario'] == 'Transformative change']
metric_df_pr_riskRating_TransChange = metric_df_pr_riskRating_TransChange.groupby('Risk').count().reset_index()
metric_df_pr_riskRating_TransChange.rename(columns = {'Scenario':'Counts'},inplace = True)
metric_df_pr_riskRating_TransChange['Scenario'] = 'Transformative change'
metric_df_pr_riskRating_TransChange = metric_df_pr_riskRating_TransChange[['Scenario','Risk','Counts']]
metric_df_pr_riskRating_TransChange

#metric high ambition risk
metric_df_pr_riskRating_HighAmbition = prdf[['Scenario','Risk']]
metric_df_pr_riskRating_HighAmbition = metric_df_pr_riskRating_HighAmbition[metric_df_pr_riskRating_HighAmbition['Scenario'] == 'High ambition']
metric_df_pr_riskRating_HighAmbition = metric_df_pr_riskRating_HighAmbition.groupby('Risk').count().reset_index()
metric_df_pr_riskRating_HighAmbition.rename(columns = {'Scenario':'Counts'},inplace = True)
metric_df_pr_riskRating_HighAmbition['Scenario'] = 'High ambition'
metric_df_pr_riskRating_HighAmbition = metric_df_pr_riskRating_HighAmbition[['Scenario','Risk','Counts']]
metric_df_pr_riskRating_HighAmbition

metric_df_pr_riskRating = metric_df_pr_riskRating_TransChange.append(metric_df_pr_riskRating_HighAmbition, ignore_index=True)
metric_df_pr_riskRating = metric_df_pr_riskRating.groupby(['Scenario','Risk']).sum()
metric_df_pr_riskRating = metric_df_pr_riskRating.reset_index()
metric_df_pr_riskRating


#### Maximum GBP increase in insurance costs, across scenarios
df_pr_maxIncreaseInsurance = prdf[['Scenario','increase in insurance']].groupby('Scenario').max().reset_index().rename(columns = {'increase in insurance':'Max. increase in insurance'})
df_pr_maxIncreaseInsurance
graph_pr_maxIncIns = px.box(prdf,x='Scenario',y='increase in insurance')

####################################
####################################
####################################
####################################
####################################
####################################
####################################
####################################




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
                                                ]),

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
                                                            figure = graph_distributionOfCurrentEPC,
                                                        ),
                                                        dt.DataTable(
                                                                        id='distributionOfCurrentEPC1',
                                                                        columns=[{"name": i, "id": i} for i in trdf[['EPC','Risk']].groupby('EPC').count().reset_index().rename(columns={'Risk': 'No. of Properties'}).columns],
                                                                        data=trdf[['EPC','Risk']].groupby('EPC').count().reset_index().rename(columns={'Risk': 'No. of Properties'}).to_dict('records'),
                                                        ),
                                                        
                                                        # dp.PivotTable(
                                                        #         data=df_pr[['Risk','EPC']].values.tolist()
                                                        #         # cols=["EPC"],
                                                        #         # rows=['LGD'],
                                                        #         # aggregatorNa   me = "Count",
                                                        #         # rendererName = "Table Row Heatmap",
                                                        #       # rows=["Risk"],
                                                        #       # vals=["Count"]
                                                        #         # menuLimit= 3
                                                        # ),
                                                        # DCC

                                                        html.Br(),
                                                        html.Br(),
                                                        html.Br(),
                                                        
                                                        dcc.Dropdown(
                                                                    id = 'tr_epcDist',
                                                                    options = [
                                                                                {'label': 'Low risk', 'value':'Low'},
                                                                                {'label': 'Medium risk', 'value':'Medium'},
                                                                                {'label': 'High risk', 'value':'High'},
                                                                    ],
                                                                    value = ['Low'],
                                                                    # multi = True,
                                                            ),
                                    
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

                                            # DIV.5
                                            html.Div(children=[

                                                #DIV.5.1
                                                html.Div(children=[
                                                    html.H2('AIB Financial Impact')
                                                ]),

                                                #DIV.5.2
                                                html.Div(children=[

                                                    #DIV.5.2.1
                                                    html.Div(children=[
                                                        html.H3('LTV'),

                                                        dcc.Graph(
                                                            id = 'tr_ltv',
                                                            # figure = graph_tr_ltv
                                                        ),

                                                        dcc.RangeSlider(
                                                            id='slider_ltv',
                                                            min=0,
                                                            max=trdf.Loan.max(),
                                                            # step=0.5,
                                                            value=[502312, 1523330],
                                                            ),
                                                            html.Div(id='output_ltv')
                                                            # step = 1000,

                                                            # max=trdf['Property value'],
                                                            # step=1,
                                                            # value=[30000, 100000]

                                                        
                                                        
                                                    ]),

                                                    # #DIV.5.2.2
                                                    # html.Div(children=[
                                                    #     html.H3('RWA')
                                                    # ]),

                                                    # #DIV.5.2.3
                                                    # html.Div(children=[
                                                    #     html.H3('ECL')
                                                    # ]),
                                                ]),

                                            ]),

                                        ]),
                                 ])

    elif tab == 'tab-2':
        return html.Div(children=[
                                     #DIV.Physical
                                        html.Div(children=[

                                            #DIV.1
                                            html.Div(children=[
                                                html.H1('Physical Risk'),
                                                
                                                # dp.PivotTable(
                                                #             data=physical_Risk_data,
                                                #             cols=["Property value"],
                                                #             aggregatorName = "Count",
                                                #             rendererName = "Scatter Chart",
                                                #             # rows=["Risk"],
                                                #             # vals=["Count"]
                                                #         ),
                                                ]),

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
                                                        html.H3('Distribution of current Flood Score'),

                                                        # ##### Distribution of current Flood Score
                                                        dcc.Graph(
                                                            id = 'distributionOfCurrentFloodScore',
                                                            figure = graph_distributionOfCurrentFloodScore,
                                                        ),
                                                        dt.DataTable(
                                                                        id='distributionOfCurrentFloodScore1',
                                                                        columns=[{"name": i, "id": i} for i in prdf[['Flood_Score','Risk']].groupby('Flood_Score').count().reset_index().rename(columns={'Risk': 'No. of Properties'}).columns],
                                                                        data=prdf[['Flood_Score','Risk']].groupby('Flood_Score').count().reset_index().rename(columns={'Risk': 'No. of Properties'}).to_dict('records'),
                                                        ),
                                                        
                                                        # dp.PivotTable(
                                                        #         data=df_pr[['Risk','EPC']].values.tolist()
                                                        #         # cols=["EPC"],
                                                        #         # rows=['LGD'],
                                                        #         # aggregatorNa   me = "Count",
                                                        #         # rendererName = "Table Row Heatmap",
                                                        #       # rows=["Risk"],
                                                        #       # vals=["Count"]
                                                        #         # menuLimit= 3
                                                        # ),
                                                        # DCC

                                                        html.Br(),
                                                        html.Br(),
                                                        html.Br(),
                                                        
                                                        dcc.Dropdown(
                                                                    id = 'pr_FloodScoreDist',
                                                                    options = [
                                                                                {'label': 'Low risk', 'value':'Low'},
                                                                                {'label': 'Medium risk', 'value':'Medium'},
                                                                                {'label': 'High risk', 'value':'High'},
                                                                    ],
                                                                    value = ['Low'],
                                                                    # multi = True,
                                                            ),
                                    
                                                    ], className = 'six columns'),

                                                    #DIV.2.2.2
                                                    html.Div(children=[
                                                        html.H3('Financial impact (ECL)'),
                                                        # ##### Maximum/range of financial impact (e.g. RWA / ECL)
                                                        dcc.Graph(
                                                            id = 'pr_financialImpact',
                                                            figure = pr_financialImpact,
                                                        ),

                                                        dt.DataTable(
                                                                        id='table_df_pr_financialImpact',
                                                                    columns=[{"name": i, "id": i} for i in df_pr_financialImpact.columns],
                                                                        data=df_pr_financialImpact.to_dict('records'),
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
                                                            id = 'pr_TotalOutstandingLoans',
                                                            figure = pr_TotalOutstandingLoans,
                                                        ),

                                                        dt.DataTable(
                                                                        id='table_df_pr_TotalOutstandingLoans',
                                                                        columns=[{"name": i, "id": i} for i in df_pr_TotalOutstandingLoans.columns],
                                                                        data=df_pr_TotalOutstandingLoans.to_dict('records'),
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
                                                                        id = 'pr_numberOfProperties1',
                                                                        figure = pr_numberOfProperties1,
                                                        ),

                                                        dt.DataTable(
                                                                        id='df_pr_noOfProperties',
                                                                        columns=[{"name": i, "id": i} for i in df_pr_noOfProperties.columns],
                                                                        data=df_pr_noOfProperties.to_dict('records'),
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
                                                        html.H3('GBP increase in insurance'),
                                                        # ##### GBP increase in insurance
                                                        dcc.Graph(
                                                            id = 'graph_InsIns',
                                                            figure = graph_pr_insInc,
                                                        ),

                                                        dt.DataTable(
                                                                        id='table_insInc',
                                                                        columns=[{"name": i, "id": i} for i in df_pr_insInc.columns],
                                                                        data=df_pr_insInc.to_dict('records'),
                                                        )
                                                    ], className='six columns'),

                                                    # DIV.3.2.2
                                                    html.Div(children=[
                                                        html.H3('Present value of increased insurance cost as percent property value'),
                                                        # ##### Property refurbishment cost as % of property value
                                                        dcc.Graph(
                                                            id = 'graph_pr_FutInsuranceToPropertyVal',
                                                            figure = graph_pr_FutInsuranceToPropertyVal,
                                                        ),

                                                        dt.DataTable(
                                                                        style_cell={
                                                                                    'whiteSpace': 'normal',
                                                                                    'height': 'auto',
                                                                                    'overflow': 'hidden',
                                                                                    'textOverflow': 'ellipsis',
                                                                                    'maxWidth': 0
                                                                                },
                                                                        id='df_pr_FutInsuranceToPropertyVal',
                                                                        columns=[{"name": i, "id": i} for i in df_pr_FutInsuranceToPropertyVal.columns],
                                                                        data=df_pr_FutInsuranceToPropertyVal.to_dict('records'),
                                                        )
                                                    ], className='six columns'),
                                                ],className = 'row'),

                                                #DIV.3.3
                                                html.Div(children=[

                                                    #DIV.3.3.1
                                                    html.Div(children=[
                                                        html.H3('Increase in insurance cost as perc of mortgage payment'),
                                                        # ##### Increase in insurance cost as % of mortgage payment
                                                        dcc.Graph(
                                                            id = 'Increase in insurance cost as perc of mortgage payment',
                                                            figure = graph_pr_increaseInsurAsPercMortg,
                                                        ),

                                                        dt.DataTable(
                                                                        id='table_metric_df_annualisedRefCostAsPercentOfMortgagePayment',
                                                                        columns=[{"name": i, "id": i} for i in df_pr_increaseInsurAsPercMortg.columns],
                                                                        data=df_pr_increaseInsurAsPercMortg.to_dict('records'),
                                                        ),

                                                    ], className='six columns'),

                                                    # DIV.3.3.2
                                                    html.Div(children=[
                                                        html.H3('Property valuation haircut'),
                                                            ##### Property valuation haircut
                                                            dcc.Graph(
                                                                id = 'graph_pr_propVal',
                                                                figure = graph_pr_propVal,
                                                            ),
                                                            # dcc.Graph(
                                                            #     id = 'propertyValuationHaircut2',
                                                            #     figure = propertyValuationHaircut2,
                                                            # ),

                                                            dt.DataTable(
                                                                        id='df_prophaircut',
                                                                        columns=[{"name": i, "id": i} for i in metric_pf_propertyValuationHaircut.columns],
                                                                        data=metric_pf_propertyValuationHaircut.to_dict('records'),
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
                                                            id = 'pr_RiskRating1',
                                                            figure = pr_RiskRating1,
                                                        ),

                                                        dt.DataTable(
                                                                        id='metric_df_pr_riskRating',
                                                                        columns=[{"name": i, "id": i} for i in metric_df_pr_riskRating.columns],
                                                                        data=metric_df_pr_riskRating.to_dict('records'),
                                                        ),
                                                    ], className='six columns'),

                                                    # DIV.3.4.2
                                                    html.Div(children=[
                                                        html.H3('Maximum GBP increase in insurance costs, across scenarios'),
                                                        #### Maximum GBP increase in insurance costs, across scenarios
                                                        dcc.Graph(
                                                            id = 'graph_pr_maxIncIns',
                                                            figure = graph_pr_maxIncIns,
                                                        ),

                                                        dt.DataTable(
                                                                        id='table_metric_df_maxAnnualRefCostCrossScenarios',
                                                                        columns=[{"name": i, "id": i} for i in df_pr_maxIncreaseInsurance.columns],
                                                                        data=df_pr_maxIncreaseInsurance.to_dict('records'),
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

                                            # DIV.5
                                            html.Div(children=[

                                                #DIV.5.1
                                                html.Div(children=[
                                                    html.H2('AIB Financial Impact')
                                                ]),

                                                #DIV.5.2
                                                html.Div(children=[

                                                    #DIV.5.2.1
                                                    html.Div(children=[
                                                        html.H3('Property reduction'),

                                                        dcc.Graph(
                                                            id = 'pr_propertyReduction',
                                                            # figure = 
                                                        ),
                                                        
                                                        dcc.RangeSlider(
                                                            id='slider_propertyReduction',
                                                            min=prdf['Property value'].min(),
                                                            max=prdf['Property value'].max(),
                                                            # step=0.5,
                                                            value=[5312, 333330],
                                                            ),
                                                            html.Div(id='output_propertyReduction')
                                                            # step = 1000,

                                                            # max=trdf['Property value'],
                                                            # step=1,
                                                            # value=[30000, 100000]
                                                        
                                                    ]),

                                                    # #DIV.5.2.2
                                                    # html.Div(children=[
                                                    #     html.H3('RWA')
                                                    # ]),

                                                    # #DIV.5.2.3
                                                    # html.Div(children=[
                                                    #     html.H3('ECL')
                                                    # ]),
                                                ]),

                                            ]),

                                        ]),
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

@app.callback(
    Output('distributionOfCurrentEPC','figure'),
    [Input('tr_epcDist', 'value')]
)
def tf_distEPC(risk):
    filtered_epcDist = trdf[trdf.Risk == risk]

    graph_distributionOfCurrentEPC = px.bar(
      filtered_epcDist,
      y='EPC',
      color='EPC',
      title="EPC distribution",
      category_orders={"EPC": ["A", "B", "C", "D","E","F","G"]},
      labels={"Risk": "Risk"},
      orientation='h'
      )

    # graph_distributionOfCurrentEPC.update_layout(transition_duration=500)
    return (graph_distributionOfCurrentEPC)

@app.callback(
    Output('distributionOfCurrentFloodScore','figure'),
    [Input('pr_FloodScoreDist', 'value')]
)
def tf_distFloodScore(risk1):
    filtered_floodScoreDist = prdf[prdf.Risk == risk1]

    graph_distributionOfCurrentFloodScore = px.bar(
      filtered_floodScoreDist,
      y='Flood_Score',
      color='Flood_Score',
      title="Flood score distribution",
    #   category_orders={"EPC": ["A", "B", "C", "D","E","F","G"]},
      labels={"Risk": "Risk"},
      orientation='h'
      )

    # graph_distributionOfCurrentEPC.update_layout(transition_duration=500)
    return (graph_distributionOfCurrentFloodScore)

@app.callback(
    Output('tr_financialImpact','figure'),
    [Input('tr_finImpactt', 'value')]
)
def tf_finImpact(risk1):
    filtered_finImp = trdf[trdf.Risk == risk1]

    graph_finImp = px.box(
        filtered_finImp, 
        # x="Risk", 
        y="ECL", 
        points="all",
        template=template, 
        title="Financial Impact"
        )

    # graph_distributionOfCurrentEPC.update_layout(transition_duration=500)
    return (graph_finImp)

@app.callback(
    dash.dependencies.Output('output_ltv', 'children'),
    [dash.dependencies.Input('slider_ltv', 'value')]
)
def output(loan):
    return 'Selected value of the mortgage: from £{0} to £{1}'.format(loan[0],loan[1])
@app.callback(
    Output('tr_ltv', 'figure'),
    Input('slider_ltv', 'value'))
def tr_ltv(loan):
    trdf['Loan to Value'] = trdf['Loan'] / trdf['Property value'] 
    filtered_ltv = trdf[(trdf['Loan']>=loan[0])&(trdf['Loan']<=loan[1])]
    graph_ltv= px.box(filtered_ltv, x ='Risk', y='Loan to Value', color = 'Scenario')
    # graph_distributionOfCurrentEPC.update_layout(transition_duration=500)
    graph_ltv.update_traces
    return(graph_ltv)


@app.callback(
    dash.dependencies.Output('output_propertyReduction', 'children'),
    [dash.dependencies.Input('slider_propertyReduction', 'value')]
)
def pr_output(propValue):
    return 'Selected property value: from £{0} to £{1}'.format(propValue[0],propValue[1])
@app.callback(
    Output('pr_propertyReduction', 'figure'),
    Input('slider_propertyReduction', 'value'))
def pr_propertyReduction(propValue):

    filtered_propReduction = prdf[(prdf['Property value']>=propValue[0])&(prdf['Property value']<=propValue[1])]
    graph_propValue= px.line(filtered_propReduction ,y = 'Property reduction', x = 'projected insurance premium 2050', color = 'Flood Risk')
    # graph_distributionOfCurrentEPC.update_layout(transition_duration=500)
    # graph_ltv.update_traces
    return(graph_propValue)



if __name__=='__main__':
    app.run_server()
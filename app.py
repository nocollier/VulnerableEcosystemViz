import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import os,glob
import pickle

colors_file = "data/random_colors.200"

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

class ClusterViz():

    def __init__(self):
        self.cases = {}
        self.selected_case = None
        self.selected_k = None
        self.df = self.dfu = self.dfc = self.dfi = self.dfr = None
        self.names = None # ["gpp [g m-2 d-1]","seasonal pr [mm d-1]","tas [degK]"]
        self.t = None # np.asarray(range(1850,2101,10),dtype=int)
        self.selected_t0 = None # self.t[0]
        self.selected_tf = None # self.t[-2]
        self.res = None
        
    def FindDatafiles(self):
        clusters = glob.glob("data/clusters*")
        seeds    = glob.glob("data/seeds*")
        seeds_un = [s for s in seeds if s.endswith(".unstd")]
        seeds    = [s for s in seeds if s not in seeds_un]
        cases    = []  # what cases do we have?
        for f in clusters:
            case = f.split(".")[-2]
            if case not in cases: cases.append(case)
        ks = {}  # for each case, how many k's have we run?
        for case in cases:
            ks[case] = []
            for cluster in clusters:
                if ".%s." % case in cluster:
                    ks[case].append(int(cluster.split(".")[-1]))
            ks[case] = sorted(ks[case])
        self.cases = ks
        self.selected_case = list(self.cases.keys())[0]
        self.selected_k = ks[self.selected_case][0]

    def LoadDatasets(self):
        
        case = self.selected_case
        k = self.selected_k
        seed_file = "data/seeds.out.%s.%d.final" % (case,k)
        cluster_file = "data/clusters.out.%s.%d" % (case,k)
        coords_file = "data/coords.%s" % (case)
        years_file = "data/years.%s" % (case)
        names_file = "data/names.%s" % (case)
        
        # these are things that we would need to read out of files
        t = np.fromfile(years_file,sep=' ')
        t = np.hstack([t,2*t[-1]-t[-2]])
        self.t = t
        if self.selected_t0 is None: self.selected_t0 = self.t[0]
        if self.selected_tf is None: self.selected_tf = self.t[-2]
        with open(names_file,'rb') as f:
            names = self.names = pickle.load(f)

        # read in the clustering datafiles
        df  = pd.read_csv(seed_file           ,sep="\t",header=None,names=['junk',]+names)
        dfu = pd.read_csv(seed_file + ".unstd",sep=" " ,header=None,names=['junk',]+names)
        dfc = pd.read_csv(coords_file         ,sep=" " ,header=None,names=['lon','lat'])
        dfi = pd.read_csv(cluster_file        ,sep=" " ,header=None,names=['id'])
        
        # intermediate quantities
        ntimes = t.size-1
        ncells = int(dfi.size / ntimes)
        dfc = dfc.iloc[:ncells] # we only need the first time slice of coordinates
        dfc.lon -= 180
        self.res = np.sqrt(((dfc.lat-dfc.lat[0])**2 + (dfc.lon-dfc.lon[0])**2).sort_values())[1]
        
        # reduce the cluster dataframes
        dfi = pd.DataFrame(dict(id=dfi.id,cell=np.tile(range(ncells),ntimes),time=np.repeat(t[:-1],ncells)))
        dfr = dfi.groupby(['id','time']).count().reset_index()

        # store in the struct, maybe not all of these are needed
        self.df = df
        self.dfu = dfu
        self.dfc = dfc
        self.dfi = dfi
        self.dfr = dfr
        self.names = names

    def UpdateParameterPlots(self, x_col, y_col, selectedpoints, selectedpoints_local):
     
        df = self.dfu
        colors = [c.strip() for c in open(colors_file).readlines()[:(df.index.max()-df.index.min()+1)]]
        if selectedpoints_local and selectedpoints_local['range']:
            ranges = selectedpoints_local['range']
            selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                                'y0': ranges['y'][0], 'y1': ranges['y'][1]}
        else:
            selection_bounds = {'x0': np.min(df[x_col]), 'x1': np.max(df[x_col]),
                                'y0': np.min(df[y_col]), 'y1': np.max(df[y_col])}
        fig = px.scatter(df,x=df[x_col],y=df[y_col],text=df.index+1,color=df.index,color_continuous_scale=colors)
        fig.update_traces(selectedpoints=selectedpoints, 
                          customdata=df.index,
                          mode='markers+text',
                          marker={ 'size' : 20 },
                          textfont_size=14,
                          textfont_color='#E5ECF6',
                          unselected={'marker'  : { 'opacity': 0.05 },
                                      'textfont': { 'color'  : 'rgba(0, 0, 0, 0)' }})
        fig.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5}, dragmode='select', hovermode=False)
        fig.update(layout_coloraxis_showscale=False)
        fig.add_shape(dict({'type': 'rect', 
                            'line': { 'width': 1, 'dash': 'dot', 'color': 'darkgrey' }}, 
                           **selection_bounds))
        return fig

    def GenerateLayout(self):
        
        # Enumerate cases and their numbers of clusters
        case_options = [dict(label=n,value=n) for n in self.cases]
        case_names   = [n for n in self.cases]
        self.selected_case = case_names[0]
        row1 = html.Div([
            html.Label("Select Case Name",
                       style = {'display':'inline-block','verticalAlign':'middle'}),
            html.Div(
                dcc.Dropdown(id="case",
                             value = case_names[0],
                             options = case_options),
                style = {'margin-left':'15px','width':'24%','display':'inline-block','verticalAlign':'middle'}),
            html.Label("Select Number of Clusters",
                       style = {'margin-left':'15px','display':'inline-block','verticalAlign':'middle'}),
            html.Div(
                dcc.Dropdown(id="k"),
                style = {'margin-left':'15px','width':'24%','display':'inline-block','verticalAlign':'middle'}),
        ],style = {'width':'100%','height': '5%'})

        # Parameter plots
        options = [dict(label=n,value=n) for n in self.names]
        panel1 = html.Div([
            html.H6("Cluster Centroid Isoplane 1",
                       style = {'width':'100%','display':'inline-block','verticalAlign':'middle'}),
            html.Label("x-axis",
                       style = {'display':'inline-block','verticalAlign':'middle'}),
            html.Div(
                dcc.Dropdown(id="p1x",value=self.names[1],options=options),
                style = {'margin-left':'15px','width':'30%','display':'inline-block','verticalAlign':'middle'}),
            html.Label("y-axis",
                       style = {'margin-left':'15px','display':'inline-block','verticalAlign':'middle'}),
            html.Div(
                dcc.Dropdown(id="p1y",value=self.names[0],options=options),
                style = {'margin-left':'15px','width':'30%','display':'inline-block','verticalAlign':'middle'}),
            dcc.Graph(id='g1',style={'height':'80%'}),
        ],style = {'width':'33%','height': '45%','display':'inline-block'})
        
        panel2 = html.Div([
            html.H6("Cluster Centroid Isoplane 2",
                       style = {'width':'100%','display':'inline-block','verticalAlign':'middle'}),
            html.Label("x-axis",
                       style = {'display':'inline-block','verticalAlign':'middle'}),
            html.Div(
                dcc.Dropdown(id="p2x",value=self.names[2],options=options),
                style = {'margin-left':'15px','width':'30%','display':'inline-block','verticalAlign':'middle'}),
            html.Label("y-axis",
                       style = {'margin-left':'15px','display':'inline-block','verticalAlign':'middle'}),
            html.Div(
                dcc.Dropdown(id="p2y",value=self.names[0],options=options),
                style = {'margin-left':'15px','width':'30%','display':'inline-block','verticalAlign':'middle'}),
            dcc.Graph(id='g2',style={'height':'80%'}),
        ],style = {'width':'33%','height': '45%','display':'inline-block'})
        
        panel3 = html.Div([
            html.H6("Cluster Centroid Isoplane 3",
                       style = {'width':'100%','display':'inline-block','verticalAlign':'middle'}),
            html.Label("x-axis",
                       style = {'display':'inline-block','verticalAlign':'middle'}),
            html.Div(
                dcc.Dropdown(id="p3x",value=self.names[2],options=options),
                style = {'margin-left':'15px','width':'30%','display':'inline-block','verticalAlign':'middle'}),
            html.Label("y-axis",
                       style = {'margin-left':'15px','display':'inline-block','verticalAlign':'middle'}),
            html.Div(
                dcc.Dropdown(id="p3y",value=self.names[1],options=options),
                style = {'margin-left':'15px','width':'30%','display':'inline-block','verticalAlign':'middle'}),
            dcc.Graph(id='g3',style={'height':'80%'}),
        ],style = {'width':'33%','height': '45%','display':'inline-block'})

        times = [dict(label=n,value=n) for n in self.t[:-1]]
        panel4 = html.Div([
            html.H6("Cluster Area Contribution",
                       style = {'width':'100%','display':'inline-block'}),
            dcc.Graph(id='line',style={'height':'80%'}),
            html.Label("Initial Time",
                       style = {'display':'inline-block','verticalAlign':'middle'}),
            html.Div(
                dcc.Dropdown(id="t0",value=self.t[0],options=times),
                style = {'margin-left':'15px','width':'30%','display':'inline-block','verticalAlign':'middle'}),
            html.Label("Final Time",
                       style = {'margin-left':'15px','display':'inline-block','verticalAlign':'middle'}),
            html.Div(
                dcc.Dropdown(id="tf",value=self.t[-2],options=times),
                style = {'margin-left':'15px','width':'30%','display':'inline-block','verticalAlign':'middle'}),
        ],style = {'width':'26%','height': '100%','display':'inline-block','vertical-align':'top'})
        
        panel5 = html.Div([
            html.H6(id="T0",style={'width':'100%','display':'inline-block'}),
            dcc.Graph(id='g4',style={'height':'95%'}),
        ],style = {'width':'36%','height': '100%','display':'inline-block','vertical-align':'top'})
        
        panel6 = html.Div([
            html.H6(id="Tf",style={'width':'100%','display':'inline-block'}),
            dcc.Graph(id='g5',style={'height':'95%'}),
        ],style = {'width':'36%','height': '100%','display':'inline-block','vertical-align':'top'})
        
        app.layout = html.Div([
            row1,
            panel1,panel2,panel3,
            html.Div([panel4,panel5,panel6],style={'height':'49%','vertical-align':'top'}),
        ],style = {'width':'100%','height': '95vh'})

cv = ClusterViz()
cv.FindDatafiles()
cv.LoadDatasets()
cv.GenerateLayout()
    
@app.callback(
    [Output('k','options'),Output('k','value'),
     Output('t0','options'), Output('t0','value'),
     Output('tf','options'), Output('tf','value'),
     Output('p1x','options'), Output('p1x','value'),
     Output('p1y','options'), Output('p1y','value'),
     Output('p2x','options'), Output('p2x','value'),
     Output('p2y','options'), Output('p2y','value'),
     Output('p3x','options'), Output('p3x','value'),
     Output('p3y','options'), Output('p3y','value')],
    Input('case','value')
)
def SelectCase(case):
    """
    Once the case is selected, we need to populate menus, reload the
    datafiles, and populate graphs.
    """
    cv.selected_case = case
    
    # Choose the k value and reload datasets. Keep the current k if it
    # is available in the new case.
    k_options = [dict(label=n,value=n) for n in cv.cases[case]]
    if cv.selected_k not in cv.cases[case]: cv.selected_k = cv.cases[case][0]
    cv.LoadDatasets()
    
    # Times depend on the files being updated. Keep the current times
    # if they are available in the new case/k.
    time_options = [dict(label=n,value=n) for n in cv.t[:-1]]
    if cv.selected_t0 not in cv.t[:-1]: cv.selected_t0 = cv.t[ 0]
    if cv.selected_tf not in cv.t[:-1]: cv.selected_tf = cv.t[-2]
        
    # Update parameter plot options
    p_options = [dict(label=n,value=n) for n in cv.names]
    
    return (k_options,cv.selected_k,
            time_options,cv.selected_t0,
            time_options,cv.selected_tf,
            p_options,cv.names[1],
            p_options,cv.names[0],
            p_options,cv.names[2],
            p_options,cv.names[0],
            p_options,cv.names[2],
            p_options,cv.names[1])

@app.callback(
    [Output('g1', 'figure'),
     Output('g2', 'figure'),
     Output('g3', 'figure')],
    [Input('k', 'value'),
     Input('g1', 'selectedData'),
     Input('g2', 'selectedData'),
     Input('g3', 'selectedData'),
     Input('p1x', 'value'),
     Input('p1y', 'value'),
     Input('p2x', 'value'),
     Input('p2y', 'value'),
     Input('p3x', 'value'),
     Input('p3y', 'value')]
)
def UpdateParameterPlots(value, selection1, selection2, selection3, v1x, v1y, v2x, v2y, v3x, v3y):
    if value != cv.selected_k:
        cv.selected_k = value
        cv.LoadDatasets()
    selectedpoints = cv.dfu.index
    for selected_data in [selection1, selection2, selection3]:
        if selected_data and selected_data['points']:
            selectedpoints = np.intersect1d(selectedpoints,
                [p['customdata'] for p in selected_data['points']])    
    return [cv.UpdateParameterPlots(v1x, v1y, selectedpoints, selection1),
            cv.UpdateParameterPlots(v2x, v2y, selectedpoints, selection2),
            cv.UpdateParameterPlots(v3x, v3y, selectedpoints, selection3)]

@app.callback(
    [Output('line', 'figure'),
     Output('g4', 'figure'),
     Output('g5', 'figure'),
     Output('T0', 'children'),
     Output('Tf', 'children')],
    [Input('t0', 'value'),
     Input('tf', 'value'),
     Input('k', 'value')]
)
def UpdateLineAndMaps(t0,tf,k):
    if cv.df is None: raise PreventUpdate
    colors = [c.strip() for c in open(colors_file).readlines()[:(cv.df.index.max()-cv.df.index.min()+1)]]
    t = cv.t
    cv.selected_t0 = t0
    cv.selected_tf = tf
    
    # for some reason datasets aren't always up to date at this point
    if len(cv.dfi.query("time == %d" % t0).id.unique()) != k: cv.LoadDatasets()
    size = int(round((9-4)/(2.8125-1.25)*(cv.res-1.25)+4))
    
    # left map
    q = cv.dfi.query("time == %d" % t0)
    q = pd.DataFrame({'id':q.id.to_numpy(),'lat':cv.dfc.lat.to_numpy(),'lon':cv.dfc.lon.to_numpy()})
    f1 = px.scatter(q,x='lon',y='lat',color='id',color_continuous_scale=colors)
    f1.update_traces(marker=dict(size=size))
    f1.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5})
    f1.update(layout_coloraxis_showscale=False)

    # right map
    q = cv.dfi.query("time == %d" % tf)
    q = pd.DataFrame({'id':q.id.to_numpy(),'lat':cv.dfc.lat.to_numpy(),'lon':cv.dfc.lon.to_numpy()})
    f2 = px.scatter(q,x='lon',y='lat',color='id',color_continuous_scale=colors)
    f2.update_traces(marker=dict(size=size))
    f2.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5})
    f2.update(layout_coloraxis_showscale=False)

    # line plot
    dt = 0.025*(t[-1]-t[0])
    lo = cv.dfr.cell.min()
    hi = cv.dfr.cell.max()
    ln = px.line(cv.dfr,x='time',y='cell',color='id',color_discrete_sequence=colors)
    ln.add_trace(go.Scatter(x=[t0,t0,tf,tf,tf],
                            y=[lo,hi,None,lo,hi],
                            mode='lines',line=dict(color='black',dash='dash')))
    ln.update_layout(xaxis=dict(range=[t[0]-dt,t[-1]+dt]),
                     showlegend=False,margin={'l': 20, 'r': 0, 'b': 15, 't': 5})
    return ln,f1,f2,"Initial Cluster Map (%g)" % t0,"Final Cluster Map (%g)" % tf

if __name__ == "__main__":
    app.run_server(debug=True)

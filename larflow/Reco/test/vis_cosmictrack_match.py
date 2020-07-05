from __future__ import print_function
import os,sys,argparse,json

parser = argparse.ArgumentParser("Visualize CRT-match output")
parser.add_argument("-kp","--input-kpreco",type=str,required=True,help="input DL merged file")
args = parser.parse_args()

import numpy as np
import ROOT as rt
from larlite import larlite
from larcv import larcv
from larflow import larflow
larcv.SetPyUtil()

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import lardly

# get graphs that draw the detector outlines
detdata = lardly.DetectorOutline().getlines()
crtdata = lardly.CRTOutline().getlines()

io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( args.input_kpreco )
io.open()

NENTRIES = io.get_entries()
print("NENTRIES: ",NENTRIES)

def load_event_data( ioll, ientry ):
    """
    get the graph objects for each crt track
    """
    io.go_to(ientry)

    print("entry[",ientry,"]")    
    entrydata = []
    
    for treename in ["matchedcosmictrack"]:
        ev_track    = io.get_data(larlite.data.kTrack, treename )
        ev_opflash  = io.get_data(larlite.data.kOpFlash,  treename )

        ntracks = ev_track.size()

        print(" tree set: ",treename)
        print("   ntrack=",ev_track.size(),")")
        print("   nopflash=",ev_opflash.size(),")")

        for n in xrange(ntracks):
            trktrace = lardly.data.visualize_larlite_track( ev_track[n] )
            trktrace["name"] = "%s[%d]"%("cosmic",n)
            trktrace["line"]["color"] = "rgb(50,0,100)"
            trktrace["line"]["width"] = 5
            trktrace["line"]["opacity"] = 1.0
            vis_opflash  =  lardly.data.visualize_larlite_opflash_3d( ev_opflash.at(n) )
            entrydata.append( [trktrace] + vis_opflash )

    return entrydata
    
    
EVENT_DATA = None
UNMATCHED_CLUSTERS = None
CURRENT_EVENT = None
    
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

axis_template = {
    "showbackground": True,
    #"backgroundcolor": "#141414", # black
    #"gridcolor": "rgba(255, 255, 255)",
    #"zerolinecolor": "rgba(255, 255, 255)",    
    "backgroundcolor": "rgba(100, 100, 100,0.5)",    
    "gridcolor": "rgb(50, 50, 50)",
    "zerolinecolor": "rgb(0, 0, 0)",
}

plot_layout = {
    "title": "",
    "height":800,
    "margin": {"t": 0, "b": 0, "l": 0, "r": 0},
    "font": {"size": 12, "color": "black"},
    "showlegend": False,
    #"plot_bgcolor": "#141414",
    #"paper_bgcolor": "#141414",
    "plot_bgcolor": "#ffffff",
    "paper_bgcolor": "#ffffff",
    "scene": {
        "xaxis": axis_template,
        "yaxis": axis_template,
        "zaxis": axis_template,
        "aspectratio": {"x": 1, "y": 1, "z": 1.2},
        "camera": {"eye": {"x": 1, "y": 1, "z": 1},
                   "up":dict(x=0, y=1, z=0)},
        "annotations": [],
    },
}

eventinput = dcc.Input(
    id="input_event",
    type="number",
    placeholder="Input Event")

plottrack = dcc.Dropdown(
    options=[
        {'label':'notloaded','value':'Event not loaded'},
    ],
    value='notloaded',
    id='plottrack',
)
        

app.layout = html.Div( [
    html.Div( [ eventinput,
                plottrack,
                html.Button("Plot",id="plot")
    ] ),
    html.Hr(),
    html.Div( [
        dcc.Graph(
            id="det3d",
            figure={
                "data": [],
                "layout": plot_layout,
            },
            config={"editable": True, "scrollZoom": False},
        )],
              className="graph__container"),
    html.Div(id="out")
] )


@app.callback(
    [Output("plottrack","options"),Output("plottrack","value"),Output("det3d","figure")],
    [Input("plot","n_clicks")],
    [State("input_event","value"),State("plottrack","value"),State("det3d","figure")])
def load_entry(*vals):
    global EVENT_DATA
    global UNMATCHED_CLUSTERS
    global io
    global CURRENT_EVENT
    
    if vals[1] is None:
        print("Input event is none")
        raise PreventUpdate
    if vals[1]>=NENTRIES or vals[1]<0:
        print("Input event is out of range")
        raise PreventUpdate

    entry = int(vals[1])

    opt = vals[2]
    
    # update entry, if needed

    if CURRENT_EVENT is None or entry!=CURRENT_EVENT:
        print("load new entry %d"%(int(vals[1])))
        EVENT_DATA = load_event_data( io, int(vals[1]) )
        print("loaded new entry %d: num tracks=%d"%(int(vals[1]),len(EVENT_DATA)) )
        CURRENT_EVENT = entry
        opt = "all"

    # reset data load
    options = []
    for n in xrange(len(EVENT_DATA)):
        options.append( {'label':n,'value':n} )
    options.append( {'label':"all",'value':"all"} )

    if len(EVENT_DATA)>0:
        print("set figure data")
        if opt in ["notloaded","noevents"]:
            vals[-1]["data"] = detdata+crtdata+EVENT_DATA[0]
            return options,0,vals[-1]
        elif opt in ["all"]:
            traces = detdata+crtdata
            for ev in EVENT_DATA:
                for tr in ev:
                    # pass along all but opflash
                    if "opflash" in tr["name"]:
                        continue
                    else:
                        traces.append( tr )
            vals[-1]["data"] = traces
            return options,"all",vals[-1]
        else:
            vals[-1]["data"] = detdata+crtdata+EVENT_DATA[ int(opt) ] 
            return options,int(opt),vals[-1]
    else:
        return options,"noevents",vals[-1]


if __name__ == "__main__":
    app.run_server(debug=True)

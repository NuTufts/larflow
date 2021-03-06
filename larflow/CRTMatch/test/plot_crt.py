from __future__ import print_function
import os,sys,argparse

parser = argparse.ArgumentParser("test_3d lardly viewer")
parser.add_argument("-ll","--larlite",required=True,type=str,help="larlite file with dltagger_allreco tracks")
parser.add_argument("-e","--entry",required=True,type=int,help="Entry to load")
parser.add_argument("-ns","--no-timeshift",action="store_true",default=False,help="Do not apply time-shift")

args = parser.parse_args(sys.argv[1:])

import os
import json
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from larlite import larlite
from larcv import larcv
import lardly


input_larlite = args.larlite
ientry        = args.entry

# LARLITE
io_ll = larlite.storage_manager(larlite.storage_manager.kREAD)
io_ll.add_in_filename( input_larlite )
io_ll.open()
io_ll.go_to(ientry)

# OPFLASH
evopflash_beam   = io_ll.get_data(larlite.data.kOpFlash,"simpleFlashBeam")
evopflash_cosmic = io_ll.get_data(larlite.data.kOpFlash,"simpleFlashCosmic")

traces_v = []

# TRACK
#evtrack = io_ll.get_data(larlite.data.kTrack,"dltagger_allreco")
#print("number of tracks: ",evtrack.size())
#track_v = [ lardly.data.visualize_larlite_track( evtrack[i] ) for i in range(evtrack.size())  ]
#traces_v += track_v

# CRT HITS
evhits = io_ll.get_data(larlite.data.kCRTHit,"crthitcorr")
crthit_v = [ lardly.data.visualize_larlite_event_crthit( evhits, "crthitcorr", notimeshift=args.no_timeshift) ]
filtered_crthit_v = lardly.ubdl.filter_crthits_wopreco( evopflash_beam, evopflash_cosmic, evhits )
vis_filtered_crthit_v = [ lardly.data.visualize_larlite_crthit( x, notimeshift=args.no_timeshift ) for x in filtered_crthit_v ]
traces_v += vis_filtered_crthit_v

# CRT TRACKS
evtracks   = io_ll.get_data(larlite.data.kCRTTrack,"crttrack")
crttrack_v = lardly.data.visualize_larlite_event_crttrack( evtracks, "crttrack", notimeshift=args.no_timeshift)
traces_v += crttrack_v

detdata = lardly.DetectorOutline()
crtdet  = lardly.CRTOutline()

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

axis_template = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(255, 255, 255)",
    "zerolinecolor": "rgb(255, 255, 255)",
}

plot_layout = {
    "title": "",
    "height":800,
    "margin": {"t": 0, "b": 0, "l": 0, "r": 0},
    "font": {"size": 12, "color": "white"},
    "showlegend": False,
    "plot_bgcolor": "#141414",
    "paper_bgcolor": "#141414",
    "scene": {
        "xaxis": axis_template,
        "yaxis": axis_template,
        "zaxis": axis_template,
        "aspectratio": {"x": 1, "y": 1, "z": 1},
        "camera": {"eye": {"x": 1, "y": 1, "z": 1},
                   "up":dict(x=0, y=1, z=0)},
        "annotations": [],
    },
}

testline = {
    "type":"scattergl",
    "x":[200,400,400,800],
    "y":[3200,3400,3800,4400],
    "mode":"markers",
    #"line":{"color":"rgb(255,255,255)","width":4},
    "marker":dict(size=10, symbol="triangle-up",color="rgb(255,255,255)"),
    }

app.layout = html.Div( [
    html.Div( [
        dcc.Graph(
            id="det3d",
            figure={
                "data": detdata.getlines()+crtdet.getlines()+traces_v,
                "layout": plot_layout,
            },
            config={"editable": True, "scrollZoom": False},
        )],
              className="graph__container"),
    ] )

if __name__ == "__main__":
    app.run_server(debug=True)

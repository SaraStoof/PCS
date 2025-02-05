from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import numpy as np
from .dla_3d_for_gui import *

app = Dash(__name__)

app.layout = html.Div([
    html.H4('3D Mold Growth Simulation'),

    # Main container for side-by-side layout
    html.Div([
        # Left side - Graph
        html.Div([
            dcc.Loading(
                id="loading",
                type="circle",
                children=[
                    dcc.Graph(id="graph", style={"height": "70vh"}),  # Adjust height
                    html.P(id="mold-coverage", style={"textAlign": "center", "fontSize": "16px", "marginTop": "10px"})
                ]
            )
        ], style={"flex": "2", "padding": "20px"}),

        # Right side - Input fields
        html.Div([
            html.P("Enter grid dimensions x, y, z:"),
            dcc.Input(id="x-index", type="number", min=1, value=100),
            dcc.Input(id="y-index", type="number", min=1, value=100),
            dcc.Input(id="z-index", type="number", min=1, value=100),

            html.P("Enter a spawn point:"),
            dcc.Input(id="x-coord", type="number", min=0, value=50),
            dcc.Input(id="y-coord", type="number", min=0, value=50),
            dcc.Input(id="z-coord", type="number", min=0, value=100),

            html.P("Number of simulations:"),
            dcc.Input(id="num-sims", type="number", min=1, value=5),

            html.P("Days:"),
            dcc.Input(id="days", type="number", min=1, value=60),

            html.P("Temperature:"),
            dcc.Input(id="temp", type="number", value=30),

            html.P("Relative humidity:"),
            dcc.Input(id="rh", type="number", min=1, value=97),

            dcc.Checklist(
                id="gradient-toggle",
                options=[{"label": "Gradient On", "value": "on"}],
                value=[],
            ),

            html.Br(),
            html.Button("Update Plot", id="update-button", n_clicks=0)
        ], style={"flex": "1", "padding": "20px", "border-left": "2px solid #ddd"})
    ], style={"display": "flex"})
])

@app.callback(
    [Output("graph", "figure"), Output("mold-coverage", "children")],
    Input("update-button", "n_clicks"),
    State("x-index", "value"), State("y-index", "value"), State("z-index", "value"),
    State("x-coord", "value"), State("y-coord", "value"), State("z-coord", "value"),
    State("num-sims", "value"),
    State("days", "value"),
    State("temp", "value"),
    State("rh", "value"),
    State("gradient-toggle", "value")
)
def update_bar_chart(n_clicks, grid_x, grid_y, grid_z, spawn_x, spawn_y, spawn_z,
                     num_sims, days, temp, rh, gradient_toggle):
    if n_clicks == 0:
        return px.scatter_3d(title="Waiting for input..."), "Mold coverage: N/A"

    if any(v is None or v <= 0 for v in [grid_x, grid_y, grid_z, num_sims, days, rh]):
        return px.scatter_3d(title="Error: All values must be positive integers!"), "Mold coverage: Error"

    if not (0 <= spawn_x <= grid_x and 0 <= spawn_y <= grid_y and 0 <= spawn_z <= grid_z):
        return px.scatter_3d(title="Error: Spawn point must be within the grid!"), "Mold coverage: Error"

    # Run the simulation
    grid, mold_cov = run_for_webgui(grid_x, grid_y, grid_z,
                                    spawn_x, spawn_y, spawn_z,
                                    num_sims, days, temp, rh)

    x, y, z = np.where(grid >= 1 / num_sims)
    color = grid[x, y, z]

    fig = px.scatter_3d(
        x=x, y=y, z=z,
        color=color,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        title="3D Mold Growth Simulation",
        scene=dict(
            xaxis=dict(range=[0, grid_x]),
            yaxis=dict(range=[0, grid_y]),
            zaxis=dict(range=[0, grid_z]),
            aspectmode='data'
        )
    )

    if "on" in gradient_toggle:
        fig.update_traces(marker=dict(size=2, color=color))
    else:
        fig.update_traces(marker=dict(size=2, color='green', line=dict(width=0.5, color='black')))

    return fig, f"3D Mold Coverage: {mold_cov:.2f}"

app.run_server(debug=True)

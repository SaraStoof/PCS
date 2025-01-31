from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import numpy as np
from dla_3d_for_gui import *

app = Dash(__name__)

grid = None  # Global variable to store grid
current_radius = 5  # Global variable to store current radius
reached_edge = False  # Global variable to store if the mold reached the edge

app.layout = html.Div([
    html.H4('3D Mold Growth Simulation'),

    html.Div([
        html.Div([
            dcc.Graph(id="graph", style={"height": "80vh"}),
            html.P("Current Radius: ", style={"font-weight": "bold"}),
            html.P(id="radius-text", children="0"),
        ], style={"flex": "2", "padding": "20px"}),

        html.Div([
            html.P("Grid dimensions (x, y, z):"),
            dcc.Input(id="x-index", type="number", min=1, value=100),
            dcc.Input(id="y-index", type="number", min=1, value=100),
            dcc.Input(id="z-index", type="number", min=1, value=100),

            html.P("Spawn point (x, y, z):"),
            dcc.Input(id="x-coord", type="number", min=0, value=50),
            dcc.Input(id="y-coord", type="number", min=0, value=50),
            dcc.Input(id="z-coord", type="number", min=0, value=100),

            html.P("Simulations:"),
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
            html.Button("Reset", id="reset-button", n_clicks=0),
            html.Button("Step", id="animate-button", n_clicks=0),
            html.Button("Start Animation", id="start-animation", n_clicks=0),
            html.Button("Stop Animation", id="stop-animation", n_clicks=0),

            dcc.Interval(
                id="animation-interval",
                interval=100,
                n_intervals=0,
                disabled=True
            )
        ], style={"flex": "1", "padding": "20px", "border-left": "2px solid #ddd"})
    ], style={"display": "flex"})
])


@app.callback(
    Output("animation-interval", "disabled"),
    [Input("start-animation", "n_clicks"), Input("stop-animation", "n_clicks")]
)
def control_animation(start_clicks, stop_clicks):
    """ Start/stop animation based on button clicks. """
    if start_clicks > stop_clicks:
        return False  # Enable interval (start animation)
    return True  # Disable interval (stop animation)


@app.callback(
    [Output("graph", "figure"), Output("radius-text", "children")],
    [Input("animate-button", "n_clicks"), Input("animation-interval", "n_intervals")],
    [State("x-index", "value"), State("y-index", "value"), State("z-index", "value"),
     State("x-coord", "value"), State("y-coord", "value"), State("z-coord", "value"),
     State("num-sims", "value"), State("days", "value"), State("temp", "value"),
     State("rh", "value"), State("gradient-toggle", "value")]
)
def call_animation(step_clicks, n_intervals, grid_x, grid_y, grid_z,
                   spawn_x, spawn_y, spawn_z, num_sims, days, temp, rh, gradient_toggle):
    """ Advances simulation one step while keeping camera position. """
    global grid, current_radius, reached_edge

    if step_clicks == 0 and n_intervals == 0:
        return px.scatter_3d(), "0"

    if grid is None:
        grid = np.zeros((grid_x + 1, grid_y + 1, grid_z + 1))
        grid[spawn_x, spawn_y, spawn_z] = 1

    spawn_on_x_edge = (spawn_x == 0, spawn_x == grid_x)
    spawn_on_y_edge = (spawn_y == 0, spawn_y == grid_y)
    spawn_on_z_edge = (spawn_z == 0, spawn_z == grid_z)

    current_radius, reached_edge, grid = one_batch_step(n_intervals + step_clicks, grid, temp, rh,
                                                        grid_x, grid_y, grid_z,
                                                        spawn_x, spawn_y, spawn_z,
                                                        spawn_on_x_edge, spawn_on_y_edge, spawn_on_z_edge,
                                                        current_radius, reached_edge)

    x, y, z = np.where(grid >= 1 / num_sims)
    color = grid[x, y, z]

    fig = px.scatter_3d(x=x, y=y, z=z, color=color, color_continuous_scale='Viridis')


    if "on" in gradient_toggle:
        fig.update_traces(marker=dict(size=2, color=color))
    else:
        fig.update_traces(marker=dict(
            size=2,
            color='green',
            line=dict(width=0.5, color='black')
        ))

    return fig, f"{current_radius}"


@app.callback(
    Input("reset-button", "n_clicks"),
    State("x-index", "value"), State("y-index", "value"), State("z-index", "value"),
    State("x-coord", "value"), State("y-coord", "value"), State("z-coord", "value")
)
def reset_grid(n_clicks, grid_x, grid_y, grid_z, spawn_x, spawn_y, spawn_z):
    """ Resets the simulation grid and radius. """
    global grid, current_radius, reached_edge
    grid = np.zeros((grid_x + 1, grid_y + 1, grid_z + 1))
    grid[spawn_x, spawn_y, spawn_z] = 1
    current_radius = 5
    reached_edge = False
    return



# @app.callback(
#     Output("graph", "figure"),
#     Input("update-button", "n_clicks"),
#     State("x-index", "value"), State("y-index", "value"), State("z-index", "value"),
#     State("x-coord", "value"), State("y-coord", "value"), State("z-coord", "value"),
#     State("num-sims", "value"),
#     State("days", "value"),
#     State("temp", "value"),
#     State("rh", "value"),
#     State("gradient-toggle", "value")
# )
# def update_bar_chart(n_clicks, grid_x, grid_y, grid_z, spawn_x, spawn_y, spawn_z,
#                      num_sims, days, temp, rh, gradient_toggle):
#     if n_clicks == 0:
#         return px.scatter_3d()

#     if any(v is None or v <= 0 for v in [grid_x, grid_y, grid_z, num_sims, days, rh]):
#         return px.scatter_3d(title="Error: All values must be positive integers!")

#     if not (0 <= spawn_x <= grid_x and 0 <= spawn_y <= grid_y and 0 <= spawn_z <= grid_z):
#         return px.scatter_3d(title="Error: Spawn point must be within the grid!")

#     grid, mold_cov = run_for_webgui(grid_x, grid_y, grid_z,
#                                     spawn_x, spawn_y, spawn_z,
#                                     num_sims, days, temp, rh)

#     x, y, z = np.where(grid >= 1 / num_sims)
#     color = grid[x, y, z]

#     fig = px.scatter_3d(
#         x=x, y=y, z=z,
#         color=color,
#         color_continuous_scale='Viridis'
#     )

#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(range=[0, grid_x]),
#             yaxis=dict(range=[0, grid_y]),
#             zaxis=dict(range=[0, grid_z]),
#             aspectmode='data'
#         )
#     )

#     if "on" in gradient_toggle:
#         fig.update_traces(marker=dict(size=2, color=color))
#     else:
#         fig.update_traces(marker=dict(
#             size=2,
#             color='green',
#             line=dict(
#                 width=0.5,
#                 color='black'
#             )
#         ))

#     return fig


app.run_server(debug=True)

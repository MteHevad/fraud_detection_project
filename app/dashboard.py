import dash
from dash import dcc, html
import plotly.express as px

app = dash.Dash(__name__)

fig = px.bar(merged_data, x='browser', y='class', color='class', barmode='group')

app.layout = html.Div([
    html.H1("Fraud Detection Insights"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)

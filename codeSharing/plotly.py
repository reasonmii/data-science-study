!pip install plotly

# =======================================

import pandas as pd

import plotly.express as px # 훨씬 가벼움 : 간단하고 빠르게 그리고 싶을 때
import plotly.graph_objects as go # detail 한 customize 하고 싶을 때
from plotly.subplots import make_subplots

# =======================================

df = pd.read_csv('Rainfall_data.csv')
df['time'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# =======================================

# Scatter
# fig = px.scatter(df, x='time', y='Temperature', title='Rainfall',
#                  labels={'time':'Time', 'Temperature':'Temperature'},
#                  width=1300, height=500,
#                  color_discrete_sequence=['blue'])

# Line
fig = px.line(df, x='time', y='Temperature', title='Rainfall',
                 labels={'time':'Time', 'Temperature':'Temperature'},
                 width=1300, height=500,
                 line_shape = 'spline',
                 color_discrete_sequence=['blue'])

fig.show()

# =======================================

layout = dict(title_text='Rainfall',
              xaxis=dict(title='time'),
              yaxis=dict(title='temperature', showgrid=False),
              width=1300, height=500)

fig = go.Figure(layout=layout)

# Line
fig.add_trace(go.Scatter(x=df.time, y=df.Temperature, name='trend', line=dict(color='yellow')))

# Scatter
fig.add_trace(go.Scatter(x=df.time, y=df.Temperature, mode='markers', name='temperature', marker=dict(size=5, color='blue')))

# fig.update_layout(showlegend=True)
fig.show()

# =======================================

# df_1 = df[df.Year <= 2010]
# df_2 = df[df.Year > 2010]

layout = dict(title_text='Rainfall', xaxis=dict(title='time'), yaxis=dict(title='temperature'), width=1300, height=500)
fig = go.Figure(layout=layout)

# fig.add_trace(go.Scatter(x=df_1.time, y=df_1.Temperature, mode='markers', name='before 2010', marker=dict(size=5, color='blue')))
# fig.add_trace(go.Scatter(x=df_2.time, y=df_2.Temperature, mode='markers', name='after 2010', marker=dict(size=5, color='red')))

fig.add_trace(go.Scatter(x=df.time, y=df.Temperature, mode='markers', name='temperature',
                         marker=dict(size=5, color=['rgba(255,0,0,1)' if y < 2007 else ('rgba(255,265,0,1)' if y < 2014 else 'rgba(0,0,255,0.5)') for y in df.Year])))

fig.show()

# =======================================

layout = go.Layout(
    title='Rainfall',
    xaxis=dict(title='Time'),
    yaxis=dict(title='Temp', side='left', showgrid=True),
    yaxis2=dict(title='Humid', side='right', showgrid=True, overlaying='y'),
    showlegend=True
)

plot1 = go.Scatter(x=df.time, y=df.Temperature, name='temp', yaxis='y1',
                   mode='markers', marker=dict(color=['blue' if p < 1000 else 'red' for p in df.Precipitation],
                                               size=[5 if p < 1000 else 10 for p in df.Precipitation]))

plot2 = go.Scatter(x=df.time, y=df['Specific Humidity'], name='humid', yaxis='y2',
                   mode='markers', marker=dict(color=['green' if p < 1000 else 'yellow' for p in df.Precipitation],
                                               size=[5 if p < 1000 else 10 for p in df.Precipitation]))

plot3 = go.Scatter(x=df.time, y=df.Temperature, name='temp', yaxis='y1', marker=dict(color='blue'))
plot4 = go.Scatter(x=df.time, y=df['Specific Humidity'], name='humid', yaxis='y2', marker=dict(color='green'))

fig = go.Figure(data=[plot1, plot2, plot3, plot4], layout=layout)
fig.show()

# =======================================

def hoverText(df, col1='Year', col2='Precipitation', col3='Specific Humidity', col4='Relative Humidity'):

    if df[col1] < 2010:
        if df[col2] < 1000:
            return f"{df['time']}<br>Prep : normal<br>Humid dff: {df[col3] - df[col4]}"
        else:
            return f"{df['time']}<br>Prep : {df[col2]/1000:.2f}K"
    else:
        if df[col2] < 1000:
            return f"{df['time']}<br>Prep : normal<br>Humid : {df[col3]}"
        else:
            return f"{df['time']}"

layout = go.Layout(
    title='Rainfall',
    xaxis=dict(title='Time'),
    yaxis=dict(title='Temp', side='left', showgrid=True),
    yaxis2=dict(title='Humid', side='right', showgrid=True, overlaying='y'),
    showlegend=True
)

hover_text = df.apply(hoverText, axis=1, col1='Year', col2='Precipitation', col3='Specific Humidity', col4='Relative Humidity')

plot1 = go.Scatter(x=df.time, y=df.Temperature, name='temp', yaxis='y1',
                   mode='markers', marker=dict(color=['blue' if p < 1000 else 'red' for p in df.Precipitation],
                                               size=[5 if p < 1000 else 10 for p in df.Precipitation]),
                   hovertemplate="<b>%{text}</b>", text=hover_text)

plot2 = go.Scatter(x=df.time, y=df['Specific Humidity'], name='humid', yaxis='y2',
                   mode='markers', marker=dict(color=['green' if p < 1000 else 'yellow' for p in df.Precipitation],
                                               size=[5 if p < 1000 else 10 for p in df.Precipitation]),
                   hovertemplate="<b>%{text}</b>", text=hover_text)

plot3 = go.Scatter(x=df.time, y=df.Temperature, name='temp', yaxis='y1', marker=dict(color='blue'))
plot4 = go.Scatter(x=df.time, y=df['Specific Humidity'], name='humid', yaxis='y2', marker=dict(color='green'))

plot5 = go.Scatter(x=df.time, y=[25] * len(df), name='limit', yaxis='y1', marker=dict(color='red'))

fig = go.Figure(data=[plot1, plot2, plot3, plot4, plot5], layout=layout)
fig.show()

fig.write_html('plot.html')
`

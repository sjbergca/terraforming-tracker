import os
import base64
import io
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px

# --- Load and transform the data ---
def load_game_data():
    path = "games/games.xlsx"
    df = pd.read_excel(path, sheet_name="restart_Mar_2025", engine="openpyxl")
    df = df.dropna(subset=['Game #', 'SB Score', 'AV Score'])

    records = []
    for _, row in df.iterrows():
        records.append({
            'Game': row['Game #'],
            'Date': row['Date'],
            'Map': row['Map'],
            'Player': 'SB',
            'Corporation': row['SB Corp.'],
            'Score': row['SB Score'],
            'Winner': row['Winner'] == 'SB'
        })
        records.append({
            'Game': row['Game #'],
            'Date': row['Date'],
            'Map': row['Map'],
            'Player': 'AV',
            'Corporation': row['AV. Corp.'],
            'Score': row['AV Score'],
            'Winner': row['Winner'] == 'AV'
        })

    return pd.DataFrame(records)


def compute_pdf_cdf(series, bins=30):
    counts, bin_edges = np.histogram(series, bins=bins, density=True)
    cdf = np.cumsum(counts) * np.diff(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, counts, cdf

# --- Build the app ---
app = Dash(__name__)
app.title = "Terraforming Mars Tracker"

app.layout = html.Div([
    html.H1("Terraforming Mars Game Tracker"),

    dcc.Upload(
        id='upload-data',
        children=html.Div(['📄 Drag and Drop or ', html.A('Select a file')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),

    dcc.Store(id='data-refresh-flag', data=True),
    html.Div(id='latest-date-display', style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Hr(),

    dcc.Tabs([
        dcc.Tab(label='Player Summary', children=[
            dcc.Graph(id='cumulative-winrate-graph'),
            html.Div(id='summary-stats', style={'marginTop': '20px'})
        ]),
        dcc.Tab(label='Score Distributions', children=[
            html.Label("Select number of bins:"),
            dcc.Slider(id='bin-slider', min=10, max=100, step=5, value=30),
            dcc.Graph(id='score-distribution'),
            dcc.Graph(id='score-cdf'),
            dcc.Graph(id='score-diff-pdf'),
            dcc.Graph(id='score-diff-cdf'),
            dcc.Graph(id='combined-total-pdf'),
            dcc.Graph(id='combined-total-cdf'),
        ]),
        dcc.Tab(label='Corp vs Corp Matchups', children=[
            dcc.Graph(id='corp-matchup-count'),
            dcc.Graph(id='corp-matchup-winrate'),
            dcc.Graph(id='corp-boxplot')
        ]),
        dcc.Tab(label='Corporation Summary Table', children=[
            html.Div(id='corp-summary-table')
        ]),
        dcc.Tab(label='Map Insights', children=[
            dcc.Graph(id='map-game-count'),
            dcc.Graph(id='map-winrate'),
            dcc.Graph(id='map-avg-score'),
            html.H4("Map Summary Table", style={'marginTop': '30px'}),
            dash_table.DataTable(
                id='map-summary-table',
                columns=[],  # filled in by callback
                data=[],     # filled in by callback
                style_table={'overflowX': 'auto'},
                sort_action='native',
                style_cell={'textAlign': 'center'},
                style_header={'fontWeight': 'bold'}
            ),
            html.Br(),
            html.H4("Corporation Performance by Map (Win %)", style={'marginTop': '40px'}),
            html.Div(id='corp-map-summary-table')         
        ]),

        dcc.Tab(label='Game Results', children=[
            html.Div(id='raw-data-table')
        ])        
    ])
])

@app.callback(
    Output('data-refresh-flag', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def save_uploaded_file(contents, filename):
    if contents and filename:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        os.makedirs("games", exist_ok=True)
        with open("games/games.xlsx", "wb") as f:
            f.write(decoded)
    return True

@app.callback(
    Output('latest-date-display', 'children'),
    Input('data-refresh-flag', 'data')
)
def display_latest_game_date(_):
    df = load_game_data()
    latest_date = pd.to_datetime(df['Date']).max()
    return html.Div([
        html.Strong(f"Most Recent Game Recorded: {latest_date.strftime('%B %d, %Y')}",
                    style={'fontSize': '18px'})
    ])

@app.callback(
    Output('raw-data-table', 'children'),
    Input('data-refresh-flag', 'data')
)
def update_raw_data_table(_):
    df = pd.read_excel("games/games.xlsx", sheet_name="restart_Mar_2025", engine="openpyxl")
    df = df.dropna(subset=['Date', 'Map', 'SB Corp.', 'AV. Corp.', 'SB Score', 'AV Score'])
    df['Date'] = pd.to_datetime(df['Date']).dt.date  #strime time from datetime
    df_display = df[['Date', 'Map', 'SB Corp.', 'AV. Corp.', 'SB Score', 'AV Score']]

    return dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df_display.columns],
        data=df_display.to_dict('records'),
        sort_action="native",
        filter_action="native",
        page_action="none",
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'minWidth': '100px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
    )

@app.callback(
    Output('cumulative-winrate-graph', 'figure'),
    Output('summary-stats', 'children'),
    Input('data-refresh-flag', 'data')
)
def update_summary_panel(_):
    df = load_game_data()
    df = df.sort_values('Game')

    summary = []
    fig_data = []

    df_sb = df[df['Player'] == 'SB'].sort_values('Game')
    df_av = df[df['Player'] == 'AV'].sort_values('Game')
    sb_scores = df_sb['Score'].values
    av_scores = df_av['Score'].values
    margins = sb_scores - av_scores

    for player in ['SB', 'AV']:
        player_df = df[df['Player'] == player].sort_values('Game')
        player_df['Win Count'] = player_df['Winner'].astype(int).cumsum()
        player_df['Game #'] = player_df['Game']
        player_df['Cumulative Win %'] = player_df['Win Count'] / range(1, len(player_df) + 1)

        fig_data.append({
            'x': player_df['Game #'],
            'y': player_df['Cumulative Win %'],
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': player
        })

        total_games = len(player_df)
        wins = player_df['Winner'].sum()
        win_rate = round((wins / total_games) * 100, 1)
        avg_score = round(player_df['Score'].mean(), 2)
        median_score = round(player_df['Score'].median(), 2)
        min_score = round(player_df['Score'].min(), 2)
        max_score = round(player_df['Score'].max(), 2)

        # Compute current win/loss streak
        recent_results = player_df['Winner'].values[::-1]  # reverse the order to start from latest
        streak_type = 'Win' if recent_results[0] else 'Loss'
        streak_count = 0
        for result in recent_results:
            if result == recent_results[0]:
                streak_count += 1
            else:
                break


        if player == 'SB':
            win_margins = margins[df_sb['Winner'].values]
            loss_margins = margins[~df_sb['Winner'].values]
        else:
            win_margins = -margins[df_av['Winner'].values]
            loss_margins = -margins[~df_av['Winner'].values]

        max_win_margin = round(win_margins.max(), 2) if len(win_margins) else None
        max_loss_margin = round(loss_margins.min(), 2) if len(loss_margins) else None

        summary.append(html.Div([
            html.H4(f"{player} Summary"),
            html.P(f"Total Games: {total_games}"),
            html.P(f"Wins: {wins} ({win_rate}%)"),
            html.P(f"Average Score: {avg_score}"),
            html.P(f"Median Score: {median_score}"),
            html.P(f"Min Score: {min_score}"),
            html.P(f"Max Score: {max_score}"),
            html.P(f"Largest Win Margin: {max_win_margin}"),
            html.P(f"Worst Loss Margin: {max_loss_margin}"),
            html.P(f"Current Streak: {streak_count} {streak_type}{'s' if streak_count > 1 else ''}")
        ], style={'marginRight': '40px', 'display': 'inline-block'}))

    fig = {
        'data': fig_data,
        'layout': {
            'title': 'Cumulative Win % Over Time',
            'yaxis': {'tickformat': '.0%', 'range': [0, 1]},
            'xaxis': {'title': 'Game Number'}
        }
    }

    return fig, summary

@app.callback(
    Output('score-distribution', 'figure'),
    Output('score-cdf', 'figure'),
    Output('score-diff-pdf', 'figure'),
    Output('score-diff-cdf', 'figure'),
    Output('combined-total-pdf', 'figure'),
    Output('combined-total-cdf', 'figure'),
    Input('bin-slider', 'value'),
    Input('data-refresh-flag', 'data')
)
def update_score_graphs(bins, _):
    df = load_game_data()
    df_sb = df[df['Player'] == 'SB']
    df_av = df[df['Player'] == 'AV']

    sb_centers, sb_pdf, sb_cdf = compute_pdf_cdf(df_sb['Score'], bins)
    av_centers, av_pdf, av_cdf = compute_pdf_cdf(df_av['Score'], bins)

    pdf_fig = go.Figure()
    pdf_fig.add_trace(go.Scatter(x=sb_centers, y=sb_pdf, mode='lines', name='SB', line=dict(color='blue')))
    pdf_fig.add_trace(go.Scatter(x=av_centers, y=av_pdf, mode='lines', name='AV', line=dict(color='orange')))
    pdf_fig.update_layout(title='PDF of Player Scores', xaxis_title='Score', yaxis_title='Density')

    cdf_fig = go.Figure()
    cdf_fig.add_trace(go.Scatter(x=sb_centers, y=sb_cdf, mode='lines', name='SB', line=dict(color='blue')))
    cdf_fig.add_trace(go.Scatter(x=av_centers, y=av_cdf, mode='lines', name='AV', line=dict(color='orange')))
    cdf_fig.update_layout(title='CDF of Player Scores', xaxis_title='Score', yaxis_title='Cumulative Density')

    diffs = df_sb['Score'].values - df_av['Score'].values
    d_centers, d_pdf, d_cdf = compute_pdf_cdf(diffs, bins)

    diff_pdf_fig = go.Figure([go.Scatter(x=d_centers, y=d_pdf, mode='lines', name='Diff PDF')])
    diff_pdf_fig.update_layout(title='PDF of Score Difference (SB - AV)', xaxis_title='Score Diff', yaxis_title='Density')

    diff_cdf_fig = go.Figure([go.Scatter(x=d_centers, y=d_cdf, mode='lines', name='Diff CDF')])
    diff_cdf_fig.update_layout(title='CDF of Score Difference (SB - AV)', xaxis_title='Score Diff', yaxis_title='Cumulative Density')

    totals = df_sb['Score'].values + df_av['Score'].values
    t_centers, t_pdf, t_cdf = compute_pdf_cdf(totals, bins)

    total_pdf_fig = go.Figure([go.Scatter(x=t_centers, y=t_pdf, mode='lines', name='Total PDF')])
    total_pdf_fig.update_layout(title='PDF of Combined Total Score', xaxis_title='Total Score', yaxis_title='Density')

    total_cdf_fig = go.Figure([go.Scatter(x=t_centers, y=t_cdf, mode='lines', name='Total CDF')])
    total_cdf_fig.update_layout(title='CDF of Combined Total Score', xaxis_title='Total Score', yaxis_title='Cumulative Density')

    return pdf_fig, cdf_fig, diff_pdf_fig, diff_cdf_fig, total_pdf_fig, total_cdf_fig

@app.callback(
    Output('corp-matchup-count', 'figure'),
    Output('corp-matchup-winrate', 'figure'),
    Output('corp-boxplot', 'figure'),
    Input('data-refresh-flag', 'data')
)
def update_corp_vs_corp(_):
    df = pd.read_excel("games/games.xlsx", sheet_name="restart_Mar_2025", engine="openpyxl")
    df = df.dropna(subset=['SB Corp.', 'AV. Corp.', 'Winner'])

    count_matrix = df.groupby(['SB Corp.', 'AV. Corp.']).size().unstack(fill_value=0)
    fig_count = px.imshow(count_matrix,
                          labels=dict(x="AV Corp.", y="SB Corp.", color="Game Count"),
                          title="Corp vs Corp Matchup Count",
                          height=800, width=800)

    win_df = df.copy()
    win_df['SB Win'] = (df['Winner'] == 'SB').astype(int)
    winrate_matrix = win_df.pivot_table(index='SB Corp.', columns='AV. Corp.', values='SB Win', aggfunc='mean')
    fig_winrate = px.imshow(winrate_matrix,
                            labels=dict(x="AV Corp.", y="SB Corp.", color="SB Win Rate"),
                            title="Corp vs Corp Win Rate (SB Perspective)",
                            color_continuous_scale='RdBu', zmin=0, zmax=1,
                            height=800, width=800)

    player_df = load_game_data()
    box_fig = px.box(player_df, x='Corporation', y='Score', color='Player', points='all',
                    color_discrete_map={"SB": "blue", "AV": "orange"},
                    title='Score Distribution by Corporation and Player')

    box_fig.update_layout(
        height=800,
        width=1200,
        xaxis={'categoryorder': 'category ascending'}
    )

    return fig_count, fig_winrate, box_fig

@app.callback(
    Output('corp-summary-table', 'children'),
    Input('data-refresh-flag', 'data')
)
def update_corp_summary_table(_):
    df = load_game_data()

    corp_summary = []
    for corp in sorted(df['Corporation'].unique()):
        row = {'Corporation': corp}
        for player in ['SB', 'AV']:
            player_df = df[(df['Corporation'] == corp) & (df['Player'] == player)]
            games = len(player_df)
            wins = player_df['Winner'].sum()
            win_pct = round((wins / games) * 100, 1) if games else 0
            row[f'{player} Games'] = games
            row[f'{player} Wins'] = wins
            row[f'{player} Win %'] = win_pct

        both_df = df[df['Corporation'] == corp]
        total_games = len(both_df)
        total_wins = both_df['Winner'].sum()
        overall_win_pct = round((total_wins / total_games) * 100, 1) if total_games else 0
        row['Total Games'] = total_games
        row['Total Wins'] = total_wins
        row['Total Win %'] = overall_win_pct
        corp_summary.append(row)

    return dash_table.DataTable(
        columns=[
            {'name': col, 'id': col} for col in corp_summary[0].keys()
        ],
        data=corp_summary,
        sort_action='native',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )

@app.callback(
    Output('map-game-count', 'figure'),
    Input('data-refresh-flag', 'data')
)
def update_map_game_count(_):
    df = load_game_data()
    map_counts = df[['Game', 'Map']].drop_duplicates().groupby('Map').count().reset_index()
    fig = px.bar(map_counts, x='Map', y='Game', orientation='v', title='Game Count per Map')
    fig.update_layout(yaxis_title='Number of Games', xaxis_title='Map')
    return fig

@app.callback(
    Output('map-winrate', 'figure'),
    Input('data-refresh-flag', 'data')
)
def update_map_winrate(_):
    df = load_game_data()

    # Only one row per player per game
    win_data = df.groupby(['Map', 'Player'])['Winner'].agg(['sum', 'count']).reset_index()
    win_data['Win %'] = 100 * win_data['sum'] / win_data['count']

    fig = px.bar(win_data, x='Map', y='Win %', color='Player',
                 barmode='group',
                 title='Win Rate by Map and Player',
                 color_discrete_map={'SB': 'blue', 'AV': 'orange'})

    fig.update_layout(yaxis_title='Win %', xaxis_title='Map')
    return fig

@app.callback(
    Output('map-avg-score', 'figure'),
    Input('data-refresh-flag', 'data')
)
def update_map_avg_score(_):
    df = load_game_data()

    fig = px.box(df, x='Map', y='Score', color='Player',
                 points='all',
                 title='Score Distribution by Map and Player',
                 color_discrete_map={'SB': 'blue', 'AV': 'orange'})

    fig.update_layout(yaxis_title='Score', xaxis_title='Map')
    return fig

@app.callback(
    Output('map-summary-table', 'data'),
    Output('map-summary-table', 'columns'),
    Input('data-refresh-flag', 'data')
)
def update_map_summary_table(_):
    df = load_game_data()
    summary = []

    for map_name in sorted(df['Map'].unique()):
        map_df = df[df['Map'] == map_name]
        games = map_df['Game'].nunique()

        sb_df = map_df[map_df['Player'] == 'SB']
        av_df = map_df[map_df['Player'] == 'AV']

        sb_wins = sb_df['Winner'].sum()
        av_wins = av_df['Winner'].sum()

        sb_win_pct = round(100 * sb_wins / len(sb_df), 1) if len(sb_df) else 0
        av_win_pct = round(100 * av_wins / len(av_df), 1) if len(av_df) else 0

        sb_avg = round(sb_df['Score'].mean(), 1) if len(sb_df) else 0
        av_avg = round(av_df['Score'].mean(), 1) if len(av_df) else 0
        total_avg = round(map_df['Score'].mean(), 1)

        summary.append({
            'Map': map_name,
            'Games': games,
            'SB Wins': int(sb_wins),
            'AV Wins': int(av_wins),
            'SB Win %': sb_win_pct,
            'AV Win %': av_win_pct,
            'SB Avg Score': sb_avg,
            'AV Avg Score': av_avg,
            'Avg Total Score': total_avg
        })

    columns = [{"name": col, "id": col} for col in summary[0].keys()] if summary else []
    return summary, columns

@app.callback(
    Output('corp-map-summary-table', 'children'),
    Input('data-refresh-flag', 'data')
)
def update_corp_map_summary(_):
    df = load_game_data()
    
    grouped = df.groupby(['Map', 'Corporation']).agg(
        Games=('Score', 'count'),
        Wins=('Winner', 'sum'),
        Avg_Score=('Score', 'mean')
    ).reset_index()
    
    grouped['Win %'] = (grouped['Wins'] / grouped['Games'] * 100).round(1)
    grouped['Avg_Score'] = grouped['Avg_Score'].round(1)

    pivot = grouped.pivot(index='Corporation', columns='Map', values='Win %').fillna('-')
    pivot = pivot.sort_index()

    return dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in pivot.reset_index().columns],
        data=pivot.reset_index().to_dict('records'),
        style_table={'overflowX': 'auto'},
        sort_action='native',
        style_cell={'textAlign': 'center'},
        style_header={'fontWeight': 'bold'}
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)

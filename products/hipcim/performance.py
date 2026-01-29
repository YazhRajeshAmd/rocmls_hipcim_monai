# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import streamlit as st
import plotly.graph_objects as go

from components.state import session_state_get

# Display performance graph as a tile
def display_performance_graph():
    tile_display_size_offset = 50 # Arbitrary magic number
    graph_height = (session_state_get('tile_display_size') 
                    - 
                    tile_display_size_offset)

    # Initialize graph display parameters
    labels = ['']
    data_values = []
    colors = ['orange']
    x_title = 'No Transformations Applied!'

    # If there are transformations applied, include them in the graph
    pipeline = session_state_get('pipeline')
    if pipeline:
        n_tr = len(session_state_get('pipeline'))
        if n_tr:
            x_title = f"Performance of {n_tr} transform(s)"

            # Get latest times for the last operation
            latest_gpu = session_state_get('gpu_times')[-1]
            latest_cpu = session_state_get('cpu_times')[-1]
            last_ratio = latest_cpu / latest_gpu
        
            # Create a bar chart
            data_values = [last_ratio]

    # Initialize the performance graph with improved styling
    fig = go.Figure()

    # Add performance bars with gradient-like effect
    fig.add_trace(go.Bar(
        x=labels,
        y=data_values,
        name='GPU Speedup',
        orientation='v',
        marker=dict(
            color=data_values if data_values else [0],
            colorscale=[[0, '#fbbf24'], [0.5, '#f97316'], [1, '#ef4444']],
            line=dict(width=0)
        ),
        text=[f'{v:.1f}x' for v in data_values] if data_values else None,
        textposition='outside',
        textfont=dict(size=14, color='#374151', family='Inter, sans-serif')
    ))

    # Update layout with modern styling
    fig.update_layout(
        barmode='group',
        height=graph_height,
        margin=dict(l=20, r=20, t=30, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,1)',
        xaxis=dict(
            title=dict(
                text=x_title,
                font=dict(family="Inter, sans-serif", size=11, color='#6b7280')
            ),
            showgrid=False,
            showline=True,
            linecolor='#e5e7eb'
        ),
        yaxis=dict(
            title=dict(
                text='Speedup (x)',
                font=dict(family="Inter, sans-serif", size=11, color='#6b7280')
            ),
            showgrid=True,
            gridcolor='#f1f5f9',
            showline=True,
            linecolor='#e5e7eb'
        ),
        showlegend=False
    )

    # Render the chart
    st.plotly_chart(fig)


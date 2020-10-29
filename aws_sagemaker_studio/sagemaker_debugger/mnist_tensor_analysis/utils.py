import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams.update({'font.size': 8})

#create slider and updatemenues for each training step
def create_slider(steps):
    updatemenus = [dict(type='buttons',
                        direction= 'left', 
                        pad=dict(r= 10, t=85), 
                        showactive = True, 
                        x= 0.1, 
                        y= 0, 
                        xanchor= 'right', 
                        yanchor= 'top',
                        buttons=[dict(label='Play',
                                  method='animate',
                                  args=[[f'{k}' for k in range(steps)], 
                                         dict(frame=dict(duration=100, redraw=True), 
                                              transition=dict(duration=300),
                                              easing='linear',
                                              fromcurrent=True,
                                              mode='immediate')])])
                    ]

    sliders = [{'yanchor': 'top',
                'xanchor': 'left', 
                'currentvalue': {'font': {'size': 16}, 
                                 'prefix': 'Step: ', 
                                 'visible': True, 
                                 'xanchor': 'right'},
                'transition': {'duration': 500.0, 
                               'easing': 'linear'},
                'pad': {'b': 10, 't': 50}, 
                'len': 0.9, 'x': 0.1, 'y': 0, 
                'steps': [{'args': [[k], {'frame': {'duration': 500.0, 
                                                    'easing': 'linear', 
                                                    'redraw': True},
                                          'transition': {'duration': 0, 
                                                         'easing': 'linear'}}], 
                           'label': k, 
                           'method': 'animate'} 
                          for k in range(steps)       
                    ]}]
    
    return updatemenus, sliders

# create animated histograms with plotly
def create_interactive_plotly_histogram(tensors):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    tname = list(tensors.keys())[0]
    steps = list(tensors[tname].keys())
    nrows = math.ceil(len(tensors.keys())/2)
   
    #create plot for each layer in the neural network
    fig = make_subplots(rows=nrows, 
                        cols=2, 
                        horizontal_spacing = 0.05, 
                        vertical_spacing = 0.1, 
                        subplot_titles = (list(tensors.keys())))
    
    #plot histograms for training step 0
    row,col = 1,1
    for tname in tensors:  
        x = tensors[tname][steps[0]].flatten()
        
        fig.add_trace(go.Histogram(x = x, nbinsx = 100), row, col)
        if col >= 2:
            row += 1
            col = 0
        col += 1
    
    # Set frames for each training step
    frames = []
    for idx,step in enumerate(steps):
        frame = {'data': [], 
                 'name': str(idx), 
                 'traces': np.arange(len(tensors.keys()))}
        for tname in tensors:
            x = tensors[tname][step].flatten()
            print(np.min(x), np.max(x), )
            frame['data'].append(go.Histogram(x = x, nbinsx=100))
        frames.append(frame)

    #create slider and updatemenue    
    updatemenus, sliders = create_slider(len(steps))
    
    #set frames and update layout
    fig.update(frames=frames)
    fig.update_layout(width=1000, height=nrows*400, 
                      showlegend=False,
                      plot_bgcolor='rgba(0,0,0,0)',
                      updatemenus=updatemenus,
                      sliders=sliders)

    return fig.show(renderer="iframe")



# create animated histograms with matplotlib
def create_interactive_matplotlib_histogram(tensors, filename="data/animation.gif"):
    nrows = math.ceil(len(tensors.keys())/2) 
    if nrows == 1:
        nrows = 2
    fig, axes = plt.subplots(nrows, 2, figsize=(15, nrows*5))
    plt.subplots_adjust(wspace = 0.5, hspace = 0.3)
    tname = list(tensors.keys())[0]
    steps = list(tensors[tname].keys())

    #function that defines the data for the different frames
    def animate(frame):
        row,col = 0,0
        for tname in tensors:
            
            #get new data for histogram
            z = tensors[tname][steps[frame]]
            neg_values = np.where(z <= 0)[0]
            if col > 1:
                row += 1
                col = 0
               
            #clear previous histogram data
            axes[row,col].clear()
            
            #set title and new histogram data
            axes[row,col].set_title( "{} \n Step {} : {:.0f}% of values below 0 - variance : {:.2f}".format(
                                  tname, steps[frame], (len(neg_values)/z.size)*100, np.var(z)))
            axes[row,col].hist(z.flatten(),bins=100)
            col += 1
            
    simulation = FuncAnimation(fig, animate, frames=len(steps), interval=1, repeat=False)
    simulation.save(filename, writer='pillow', fps=5)
    fig.tight_layout()
    plt.close()
   
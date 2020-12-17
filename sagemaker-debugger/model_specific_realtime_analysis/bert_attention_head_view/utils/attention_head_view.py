from bokeh.plotting import show, figure
from bokeh.models.annotations import Title
from bokeh.models import  ColumnDataSource, Label, Range1d
from bokeh.io import  show, output_notebook, push_notebook
from bokeh.models.glyphs import Line
import numpy as np

output_notebook()

class AttentionHeadView():
    def __init__(self, 
                 input_tokens=None, 
                 tensors=None, 
                 layer='bertencoder0_transformer0_multiheadattentioncell0_output_1', 
                 step=0, 
                 n_tokens=20):
        self.head = 0
        self.step = step
        self.input_tokens = input_tokens[:n_tokens]
        self.n_tokens = n_tokens
        self.tensors = tensors
        self.p = None
        self.layer = layer
        self.sources = []
        self.create()
        
    def update(self):
        
        tensor =  self.tensors[self.layer][self.step][0, self.head, :, :]
        
        counter = 0
        for x in range(self.n_tokens):
            for y in range(self.n_tokens):
                source = self.sources[counter]
                source.line_width = tensor[x, y] * 2
                counter += 1
               
    def select_layer(self, layer):
        self.layer = layer
        self.update()
        push_notebook()
    
    def select_head(self, head):
        self.head = head
        self.update()
        push_notebook()
        
    def select_step(self, step):
        self.step = step
        self.update()
        push_notebook()
        
    def create(self):

        # set size of figure
        self.p = figure(width = 450, 
                   plot_height = 50 * self.n_tokens, 
                   x_range=Range1d(0, self.n_tokens + 2), 
                   y_range=Range1d(0, self.n_tokens))
        
        self.p.xgrid.visible = False
        self.p.ygrid.visible = False
        self.p.axis.visible = False
        
        x = np.zeros(self.n_tokens) + 2 
        y = np.flip(np.arange(0, self.n_tokens), axis=0)
        
        # set input tokens in plot
        for token, x_i, y_i in zip(self.input_tokens, x, y):
            text1 = Label(x = x_i - 1, 
                          y = y_i, 
                          text = token, 
                          text_font_size = '10pt')
            text2 = Label(x = x_i + 10, 
                          y = y_i, 
                          text = token, 
                          text_font_size = '10pt')
            self.p.add_layout(text2)
            self.p.add_layout(text1)
    
        tensor =  self.tensors[self.layer][self.step][0, self.head, :, :]

        #plot attention weights
        for x in range(self.n_tokens):
            for y in range(self.n_tokens):
                source = ColumnDataSource(data=dict(x=[2, 12], 
                                                    y=[self.n_tokens - x - 1, self.n_tokens - y - 1]))
                line = Line(x="x", y="y", line_width=tensor[x, y], line_color = "blue")
                self.p.add_glyph(source, line)
                self.sources.append(line)

        show(self.p, notebook_handle=True)

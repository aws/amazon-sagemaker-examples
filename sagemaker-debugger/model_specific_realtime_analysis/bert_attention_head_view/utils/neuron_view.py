from bokeh.plotting import show, figure, gridplot
from bokeh.models.annotations import Title
from bokeh.models import  ColumnDataSource, Label, Range1d
from bokeh.io import  show, output_notebook, push_notebook
from bokeh.models.glyphs import Line, Image
from bokeh.models.mappers import LinearColorMapper
import numpy as np
output_notebook()

class NeuronView():
    def __init__(self, 
                 input_tokens=None, 
                 keys=None, 
                 queries=None, 
                 layers=None, 
                 step=0, 
                 head=0, 
                 n_tokens=20, 
                 layer_names=None):
        self.layers = layers
        self.head = head
        self.step = step
        self.query = 0
        self.p = None
        self.input_tokens = input_tokens[:n_tokens]
        self.n_tokens = n_tokens
        self.keys = keys
        self.queries = queries
        self.layer_names = layer_names
        self.key_name = layers[0][0]
        self.query_name = layers[0][1]
        self.source_key = None
        self.source_query = None
        self.product = None
        self.create()
        
    def update(self):
        key =  self.keys[self.key_name][self.step][0, self.head, :, :]
        self.source_key.data["image"] = [key[:self.n_tokens,:]]
       
        query = self.queries[self.query_name][self.step][0, self.head, :self.n_tokens, :]
        query_input = np.zeros((self.n_tokens,query.shape[1])) 
        query_input[:,:] = np.nan 
        
        query_input[self.query,:] = query[self.n_tokens - self.query - 1, :]
        self.source_query.data["image"] = [query_input]
        
        product = np.multiply(query[self.n_tokens - self.query - 1, :], key)
        self.product.data["image"] = [product[:self.n_tokens,:]]
        
        dot_product = np.dot(key, query[self.query,:])
        dot_product = dot_product.reshape((dot_product.shape[0],1))
        self.dot_product.data["image"] = [dot_product[:self.n_tokens,:]]
        
    def select_query(self, query):
        self.query = query
        self.update()
        push_notebook()      
    
    def select_layer(self, layer):
        layer_id = self.layer_names[layer]
        self.key_name = self.layers[layer_id][0]
        self.query_name = self.layers[layer_id][1]
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
        self.p = figure(width = 900, 
                   plot_height = 35 * self.n_tokens, 
                   x_range=Range1d(0, self.n_tokens + 100), 
                   y_range=Range1d(-1, self.n_tokens))
        
        self.p.xgrid.visible = False
        self.p.ygrid.visible = False
        self.p.axis.visible = False
        
        x = np.zeros(self.n_tokens) + 2 
        y = np.flip(np.arange(0, self.n_tokens), axis=0) + 0.25
        
        # set input tokens in plot
        for token, x_i, y_i in zip(self.input_tokens, x, y):
            text1 = Label(x = x_i - 1, 
                          y = y_i, 
                          text = token, 
                          text_font_size = '10pt')
            text2 = Label(x = x_i + 105, 
                          y = y_i, 
                          text = token, 
                          text_font_size = '10pt')
            self.p.add_layout(text2)
            self.p.add_layout(text1)
        
        # set plot labels
        text = Label(x=17, y=-1, text="query", text_font_size = '15pt')
        self.p.add_layout(text)
        text = Label(x=50, y=-1, text="key", text_font_size = '15pt')
        self.p.add_layout(text)
        text = Label(x=80, y=-1, text="q x k", text_font_size = '15pt')
        self.p.add_layout(text)  
        text = Label(x=98, y=-1, text="q * k", text_font_size = '15pt')
        self.p.add_layout(text)  
        
        color_mapper = LinearColorMapper(palette="Blues8", nan_color='white')
        
        #get key matrix and query vector
        key =  self.keys[self.key_name][self.step][0, self.head, :, :] 
        query = self.queries[self.query_name][self.step][0, self.head, :self.n_tokens, :]
        
        #plot key matrix
        self.source_key = ColumnDataSource(data=dict(image=[key[:self.n_tokens,:]], x=[40],y=[0], dw=[25], dh=[self.n_tokens]))
        img = Image(image="image", x="x", y="y", dw="dw", dh="dh", color_mapper=color_mapper)
        self.p.add_glyph(self.source_key, img)

        #create an empty query matrix where only one vector is set
        query_input = np.zeros((self.n_tokens, query.shape[-1])) 
        query_input[:,:] = np.nan   
        query_input[self.query,:] = query[self.n_tokens - self.query - 1, :]
        self.source_query = ColumnDataSource(data=dict(image=[query_input], x=[10], y=[0], dw=[25], dh=[self.n_tokens]))
        img = Image(image="image", x="x", y="y", dw="dw", dh="dh", color_mapper=color_mapper)
        self.p.add_glyph(self.source_query, img)
        
        #compute elementwise product between a query vector and key matrix
        product = np.multiply(query[self.n_tokens - self.query - 1, :], key)
        self.product = ColumnDataSource(data=dict(image=[product[:self.n_tokens,:]], x=[70], y=[0], dw=[25], dh=[self.n_tokens]))
        img = Image(image="image", x="x", y="y", dw="dw", dh="dh", color_mapper=color_mapper)
        self.p.add_glyph(self.product, img)
        
        #compute dot product between query vector and key matrix
        dot_product = np.dot(key, query[self.n_tokens - self.query - 1, :])
        dot_product = dot_product.reshape((dot_product.shape[0], 1))
        self.dot_product = ColumnDataSource(data=dict(image=[dot_product[:self.n_tokens,:]], x=[100], y=[0], dw=[2], dh=[self.n_tokens]))
        img = Image(image="image", x="x", y="y", dw="dw", dh="dh", color_mapper=color_mapper)
        self.p.add_glyph(self.dot_product, img)

        
        show(self.p, notebook_handle=True)

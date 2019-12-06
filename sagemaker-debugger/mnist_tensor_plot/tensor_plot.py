# Third Party
import numpy as np
import plotly.graph_objects as go
import plotly.offline as py

# First Party
from smdebug.trials import create_trial

py.init_notebook_mode(connected=True)

# This class provides methods to plot tensors as 3 dimensional objects. It is intended for plotting convolutional
# neural networks and expects that inputs are images and that outputs are class labels or images.
class TensorPlot:
    def __init__(
        self,
        regex,
        path,
        steps=10,
        batch_sample_id=None,
        color_channel=1,
        title="",
        label=None,
        prediction=None,
    ):
        """

        :param regex: tensor regex
        :param path:
        :param steps:
        :param batch_sample_id:
        :param color_channel:
        :param title:
        :param label:
        :param prediction:
        """
        self.trial = create_trial(path)
        self.regex = regex
        self.steps = steps
        self.batch_sample_id = batch_sample_id
        self.color_channel = color_channel
        self.title = title
        self.label = label
        self.prediction = prediction
        self.max_dim = 0
        self.dist = 0
        self.tensors = {}
        self.output = {}
        self.input = {}
        self.load_tensors()
        self.set_figure()
        self.plot_network()
        self.set_frames()

    # Loads all tensors into a dict where the key is the step.
    # If batch_sample_id is None then batch dimension is plotted as a seperate dimension
    # if batch_sample_id is -1 then tensors are summed over batch dimension. Otherwise
    # the corresponding sample is plotted in the figure, and all the remaining samples
    # in the batch are dropped.
    def load_tensors(self):
        available_steps = self.trial.steps()
        for step in available_steps[0 : self.steps]:
            self.tensors[step] = []

            # input image into the neural network
            if self.label is not None:
                for tname in self.trial.tensor_names(regex=self.label):
                    tensor = self.trial.tensor(tname).value(step)
                    if self.color_channel == 1:
                        self.input[step] = tensor[0, 0, :, :]
                    elif self.color_channel == 3:
                        self.input[step] = tensor[0, :, :, 3]

            # iterate over tensors that match the regex
            for tname in self.trial.tensor_names(regex=self.regex):
                tensor = self.trial.tensor(tname).value(step)
                # get max value of tensors to set axis dimension accordingly
                for dim in tensor.shape:
                    if dim > self.max_dim:
                        self.max_dim = dim

                # layer inputs/outputs have as first dimension batch size
                if self.batch_sample_id != None:
                    # sum over batch dimension
                    if self.batch_sample_id == -1:
                        tensor = np.sum(tensor, axis=0) / tensor.shape[0]
                    # plot item from batch
                    elif self.batch_sample_id >= 0 and self.batch_sample_id <= tensor.shape[0]:
                        tensor = tensor[self.batch_sample_id]
                    # plot first item from batch
                    else:
                        tensor = tensor[0]

                    # normalize tensor values between 0 and 1 so that all tensors have same colorscheme
                    tensor = tensor - np.min(tensor)
                    if np.max(tensor) != 0:
                        tensor = tensor / np.max(tensor)
                    if len(tensor.shape) == 3:
                        for l in range(tensor.shape[self.color_channel - 1]):
                            if self.color_channel == 1:
                                self.tensors[step].append([tname, tensor[l, :, :]])
                            elif self.color_channel == 3:
                                self.tensors[step].append([tname, tensor[:, :, l]])
                    elif len(tensor.shape) == 1:
                        self.tensors[step].append([tname, tensor])
                else:
                    # normalize tensor values between 0 and 1 so that all tensors have same colorscheme
                    tensor = tensor - np.min(tensor)
                    if np.max(tensor) != 0:
                        tensor = tensor / np.max(tensor)
                    if len(tensor.shape) == 4:
                        for i in range(tensor.shape[0]):
                            for l in range(tensor.shape[1]):
                                if self.color_channel == 1:
                                    self.tensors[step].append([tname, tensor[i, l, :, :]])
                                elif self.color_channel == 3:
                                    self.tensors[step].append([tname, tensor[i, :, :, l]])
                    elif len(tensor.shape) == 2:
                        self.tensors[step].append([tname, tensor])

            # model output
            if self.prediction is not None:
                for tname in self.trial.tensor_names(regex=self.prediction):
                    tensor = self.trial.tensor(tname).value(step)
                    # predicted class (batch size, propabilities per clas)
                    if len(tensor.shape) == 2:
                        self.output[step] = np.array([np.argmax(tensor, axis=1)[0]])
                    # predict an image (batch size, color channel, weidth, height)
                    elif len(tensor.shape) == 4:
                        # MXNet has color channel in dim1
                        if self.color_channel == 1:
                            self.output[step] = tensor[0, 0, :, :]
                        # TF has color channel in dim 3
                        elif self.color_channel == 3:
                            self.output[step] = tensor[0, :, :, 0]

    # Configure the plot layout
    def set_figure(self):
        self.fig = go.Figure(
            layout=go.Layout(
                autosize=False,
                title=self.title,
                width=1000,
                height=800,
                template="plotly_dark",
                font=dict(color="gray"),
                showlegend=False,
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": 1, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 1},
                                    },
                                ],
                            )
                        ],
                    )
                ],
                scene=dict(
                    xaxis=dict(
                        range=[-self.max_dim / 2, self.max_dim / 2],
                        autorange=False,
                        gridcolor="black",
                        zerolinecolor="black",
                        showgrid=False,
                        showline=False,
                        showticklabels=False,
                        showspikes=False,
                    ),
                    yaxis=dict(
                        range=[-self.max_dim / 2, self.max_dim / 2],
                        autorange=False,
                        gridcolor="black",
                        zerolinecolor="black",
                        showgrid=False,
                        showline=False,
                        showticklabels=False,
                        showspikes=False,
                    ),
                    zaxis=dict(
                        gridcolor="black",
                        zerolinecolor="black",
                        showgrid=False,
                        showline=False,
                        showticklabels=False,
                        showspikes=False,
                    ),
                ),
            )
        )

    # Create a sequence of frames: tensors from same step will be stored in the same frame
    def set_frames(self):
        frames = []
        available_steps = self.trial.steps()
        for step in available_steps[0 : self.steps]:
            layers = []
            if self.label is not None:
                if len(self.input[step].shape) == 2:
                    # plot predicted image
                    layers.append({"type": "surface", "surfacecolor": self.input[step]})
            for i in range(len(self.tensors[step])):
                if len(self.tensors[step][i][1].shape) == 1:
                    # set color of fully connected layer for corresponding step
                    layers.append(
                        {"type": "scatter3d", "marker": {"color": self.tensors[step][i][1]}}
                    )
                elif len(self.tensors[step][i][1].shape) == 2:
                    # set color of convolutional/pooling  layer for corresponding step
                    layers.append({"type": "surface", "surfacecolor": self.tensors[step][i][1]})
            if self.prediction is not None:
                if len(self.output[step].shape) == 1:
                    # plot predicted class for first input in batch
                    layers.append(
                        {
                            "type": "scatter3d",
                            "text": "Predicted class " + str(self.output[step][0]),
                            "textfont": {"size": 40},
                        }
                    )
                elif len(self.output[step].shape) == 2:
                    # plot predicted image
                    layers.append({"type": "surface", "surfacecolor": self.output[step]})
            frames.append(go.Frame(data=layers))

        self.fig.frames = frames

    # Plot the different neural network layers
    # if ignore_batch_dimension is True then convolutions are plotted as
    # Surface and dense layers are plotted as Scatter3D
    # if ignore_batch_dimension is False then convolutions and dense layers
    # are plotted as Surface. We don't plot biases.
    # If convolution has shape [batch_size, 10, 24, 24] and ignore_batch_dimension==True
    # then this function will plot 10 Surface layers in the size of 24x24
    def plot_network(self):
        tensors = []
        dist = 0
        counter = 0

        first_step = self.trial.steps()[0]
        if self.label is not None:
            tensor = self.input[first_step].shape
            if len(tensor) == 2:
                tensors.append(
                    go.Surface(
                        z=np.zeros((tensor[0], tensor[1])) + self.dist,
                        y=np.arange(-tensor[0] / 2, tensor[0] / 2),
                        x=np.arange(-tensor[1] / 2, tensor[1] / 2),
                        surfacecolor=self.input[first_step],
                        showscale=False,
                        colorscale="gray",
                        opacity=0.7,
                    )
                )
            self.dist += 2
        prev_name = None
        for tname, layer in self.tensors[first_step]:
            tensor = layer.shape

            if len(tensor) == 2:
                tensors.append(
                    go.Surface(
                        z=np.zeros((tensor[0], tensor[1])) + self.dist,
                        y=np.arange(-tensor[0] / 2, tensor[0] / 2),
                        x=np.arange(-tensor[1] / 2, tensor[1] / 2),
                        text=tname,
                        surfacecolor=layer,
                        showscale=False,
                        # colorscale='gray',
                        opacity=0.7,
                    )
                )

            elif len(tensor) == 1:
                tensors.append(
                    go.Scatter3d(
                        z=np.zeros(tensor[0]) + self.dist,
                        y=np.zeros(tensor[0]),
                        x=np.arange(-tensor[0] / 2, tensor[0] / 2),
                        text=tname,
                        mode="markers",
                        marker=dict(size=3, opacity=0.7, color=layer),
                    )
                )
            if tname == prev_name:
                self.dist += 0.2
            else:
                self.dist += 1
            counter += 1
            prev_name = tname
        # plot model output
        if self.prediction is not None:
            # model predicts a class label (batch_size, class propabilities)
            if len(self.output[first_step].shape) == 1:
                tensors.append(
                    go.Scatter3d(
                        z=np.array([self.dist + 0.2]),
                        x=np.array([0]),
                        y=np.array([0]),
                        text="Predicted class",
                        mode="markers+text",
                        marker=dict(size=3, color="black"),
                        textfont=dict(size=18),
                        opacity=0.7,
                    )
                )
                # model predicts an output image  (batch size, color channel, width, height)
            elif len(self.output[first_step].shape) == 2:
                tensor = self.output[first_step].shape
                tensors.append(
                    go.Surface(
                        z=np.zeros((tensor[0], tensor[1])) + self.dist + 3,
                        y=np.arange(-tensor[0] / 2, tensor[0] / 2),
                        x=np.arange(-tensor[1] / 2, tensor[1] / 2),
                        text="Predicted image",
                        surfacecolor=self.output[first_step],
                        showscale=False,
                        colorscale="gray",
                        opacity=0.7,
                    )
                )

        # add list of tensors to figure
        self.fig.add_traces(tensors)

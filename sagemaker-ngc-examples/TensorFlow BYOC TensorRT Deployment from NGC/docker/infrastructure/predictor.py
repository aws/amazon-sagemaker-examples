"""
Based on Amazon SageMaker's 'bring your own scikit container' 'serve' example
https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own
This is the file that implements a flask server to do inferences. Modify this file for your own inference.
"""

import flask
import os
import numpy as np
import io
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
import ctypes
import os
ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
import pycuda.driver as cuda
import pycuda.autoinit
import time

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

import helpers.data_processing as dp
import helpers.tokenization as tokenization
vocab_file_path = os.path.join(model_path, "vocab.txt")
engine_path = os.path.join(model_path, "bert_large_384.engine")
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path, do_lower_case=True)
# The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
max_query_length = 64
# When splitting up a long document into chunks, how much stride to take between chunks.
doc_stride = 128
# The maximum total input sequence length after WordPiece tokenization. 
# Sequences longer than this will be truncated, and sequences shorter 
max_seq_length = 384



# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        try:
            if (cls.model == None):
                
                cls.model = open(engine_path, "rb")
                
        except Exception as e:
            # Don't return any extra information
            cls.model = None

        return cls.model

    @classmethod
    def predict(cls, short_paragraph_text, question_text):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        
        # Extract tokens from the paragraph
        doc_tokens = dp.convert_doc_tokens(short_paragraph_text)
        # Extract features from the paragraph and question
        features = dp.convert_examples_to_features(doc_tokens, question_text, tokenizer, max_seq_length, doc_stride, max_query_length)
        
        # Load the BERT-Large Engine
        with open(engine_path, "rb") as f, \
            trt.Runtime(TRT_LOGGER) as runtime, \
            runtime.deserialize_cuda_engine(f.read()) as engine, \
            engine.create_execution_context() as context:
            
            # We use batch size 1.
            input_shape = (max_seq_length, 1)
            input_nbytes = trt.volume(input_shape) * trt.int32.itemsize
            
            # Allocate device memory for inputs.
            d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]
            # Create a stream in which to copy inputs/outputs and run inference.
            stream = cuda.Stream()
            
            # Specify input shapes. These must be within the min/max bounds of the active profile (0th profile in this case)
            # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
            for binding in range(3):
                context.set_binding_shape(binding, input_shape)
            assert context.all_binding_shapes_specified
            
            # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
            h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32)
            d_output = cuda.mem_alloc(h_output.nbytes)
           
            print("\nRunning Inference now...")
            eval_start_time = time.time()

            # Copy inputs
            cuda.memcpy_htod_async(d_inputs[0], features["input_ids"], stream)
            cuda.memcpy_htod_async(d_inputs[1], features["segment_ids"], stream)
            cuda.memcpy_htod_async(d_inputs[2], features["input_mask"], stream)
            
            # Run inference
            context.execute_async_v2(bindings=[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
            
            # Transfer predictions back from GPU
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            
            # Synchronize the stream
            stream.synchronize()
            
            eval_time_elapsed = time.time() - eval_start_time
            h_output = h_output.transpose((1,0,2,3,4))
            
            print("-----------------------------")
            print("Running Inference at {:.3f} Sentences/Sec".format(1.0/eval_time_elapsed))
            print("-----------------------------")
        return h_output, doc_tokens, features, 1.0/eval_time_elapsed


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """

    # Convert from json
    if (not flask.request.content_type == 'application/json'):
        return flask.Response(response='This predictor only supports json data. We have a request of type '+flask.request.content_type, status=415, mimetype='text/plain')
    print("Getting request.")
    json_data=flask.request.get_json(force = True)
    short_paragraph_text = json_data["short_paragraph_text"]
    question_text = json_data["question_text"]
    
    
    print("Got request, starting prediction.")
    # Do prediction
    h_output, doc_tokens, features, sentences_sec = ScoringService.predict(short_paragraph_text, question_text)
    print("Finished prediction.")
    result = ""
    for index, batch in enumerate(h_output):
        start_logits = batch[:, 0]
        end_logits = batch[:, 1]

        # The total number of n-best predictions to generate in the nbest_predictions.json output file
        n_best_size = 20

        # The maximum length of an answer that can be generated. This is needed 
        #  because the start and end predictions are not conditioned on one another
        max_answer_length = 30


        (prediction, nbest_json, scores_diff_json) = \
            dp.get_predictions(doc_tokens, features, start_logits, end_logits, n_best_size, max_answer_length)

        result += "Answer: '{}'".format(prediction) + " with prob: {:.3f}% ".format(nbest_json[0]['probability'] * 100.0) + "at {:.3f} Sentences/Sec.".format(sentences_sec)
        

    return flask.Response(response=result, status=200, mimetype='text/plain')

import argparse
import time
import numpy as np
import mxnet as mx

import gluonnlp as nlp
from gluonnlp.data import SQuAD
from model import BertForQALoss, BertForQA
from data import SQuADTransform, preprocess_dataset

import smdebug.mxnet as smd
from smdebug import modes

def get_dataloaders(batch_size, vocab, train_dataset_size, val_dataset_size):
    
    batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Stack(),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack(),
    )

    train_data = SQuAD("train", version='2.0')[:train_dataset_size]

    train_data_transform, _ = preprocess_dataset(
        train_data, SQuADTransform(
            nlp.data.BERTTokenizer(vocab=vocab, lower=True),
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_pad=True,
            is_training=True))

    train_dataloader = mx.gluon.data.DataLoader(
        train_data_transform, batchify_fn=batchify_fn,
        batch_size=batch_size, num_workers=4, shuffle=True)

    #we only get 4 validation samples
    dev_data = SQuAD("dev", version='2.0')[:val_dataset_size]
    dev_data = mx.gluon.data.SimpleDataset(dev_data)
    
    dev_dataset = dev_data.transform(
        SQuADTransform(
            nlp.data.BERTTokenizer(vocab=vocab, lower=True),
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_pad=False,
            is_training=False)._transform, lazy=False)
    
    dev_data_transform, _ = preprocess_dataset(
        dev_data, SQuADTransform(
            nlp.data.BERTTokenizer(vocab=vocab, lower=True),
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_pad=False,
            is_training=False))
    
    dev_dataloader = mx.gluon.data.DataLoader(
        dev_data_transform,
        batchify_fn=batchify_fn,
        num_workers=1, batch_size=batch_size,
        shuffle=False, last_batch='keep')
    
    return train_dataloader, dev_dataloader, dev_dataset

def train_model(epochs, batch_size, learning_rate, train_dataset_size, val_dataset_size):
    
    #Check if GPU available
    ctx = mx.gpu() 

    #load petrained BERT model weights (trained on wiki dataset)
    bert, vocab = nlp.model.get_model(
        name='bert_12_768_12',
        dataset_name='book_corpus_wiki_en_uncased',
        vocab=None,
        pretrained='true',
        ctx=ctx,
        use_pooler=False,
        use_decoder=False,
        use_classifier=False,
        output_attention=True)

    #create BERT model for Question Answering
    net = BertForQA(bert=bert)
    net.span_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)

    #create smdebug hook
    hook = smd.Hook.create_from_json_file() 
    
    hook.register_block(net)
    
    #loss function for BERT model training
    loss_function = BertForQALoss()
    
    #trainer
    trainer = mx.gluon.Trainer(net.collect_params(), 
                               'bertadam',
                               {'learning_rate': learning_rate}, 
                               update_on_kvstore=False)

    #create dataloader
    train_dataloader, dev_dataloader, dev_dataset = get_dataloaders(batch_size, vocab, train_dataset_size, val_dataset_size)

    #initialize model parameters
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
  
    params = [p for p in net.collect_params().values()
              if p.grad_req != 'null']

    #start training loop
    for epoch_id in range(epochs):
        
        for batch_id, data in enumerate(train_dataloader):
            hook.set_mode(modes.TRAIN)
            with mx.autograd.record():
                _, inputs, token_types, valid_length, start_label, end_label = data

                # forward pass
                out = net(inputs.astype('float32').as_in_context(ctx),
                          token_types.astype('float32').as_in_context(ctx),
                          valid_length.astype('float32').as_in_context(ctx))

                #compute loss
                ls = loss_function(out, [
                    start_label.astype('float32').as_in_context(ctx),
                    end_label.astype('float32').as_in_context(ctx)]).mean()

            #backpropagation
            ls.backward()
            nlp.utils.clip_grad_global_norm(params, 1)
            
            #update model parameters
            trainer.update(1)
                
        #validation loop
        hook.set_mode(modes.EVAL)
        for data in dev_dataloader:

            example_ids, inputs, token_types, valid_length, _, _ = data

            #forward pass
            out = net(inputs.astype('float32').as_in_context(ctx),
                      token_types.astype('float32').as_in_context(ctx),
                      valid_length.astype('float32').as_in_context(ctx))

            #record input tokens
            input_tokens = np.array([])
            for example_id in example_ids.asnumpy().tolist():
                array = np.array(dev_dataset[example_id][0].tokens, dtype=np.str)
                array = array.reshape(1, array.shape[0])
                input_tokens = np.append(input_tokens, array)

            if hook.get_collections()['all'].save_config.should_save_step(modes.EVAL, hook.mode_steps[modes.EVAL]):  
                hook._write_raw_tensor_simple("input_tokens", input_tokens)
            

            
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)  
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--val_dataset_size', type=int, default=64) 
    parser.add_argument('--train_dataset_size', type=int, default=1024)  
    parser.add_argument('--smdebug_dir', type=str, default=None)

    #parse arguments
    args, _ = parser.parse_known_args()
    
    #train model
    model = train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, train_dataset_size=args.train_dataset_size, val_dataset_size=args.val_dataset_size)


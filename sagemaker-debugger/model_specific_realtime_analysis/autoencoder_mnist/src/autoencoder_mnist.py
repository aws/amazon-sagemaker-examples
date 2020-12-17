import mxnet as mx
from mxnet import autograd, gluon
import argparse
import smdebug.mxnet as smd
from smdebug.mxnet import modes

def parse_args():
    parser = argparse.ArgumentParser(description="Train an autoencoder model") 
    parser.add_argument("--out_dir", type=str, default="/opt/ml/output/tensors", help="Variable to change path to tensors")
    parser.add_argument("--batch_size", type=int, default=64, help="Variable to change batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Variable to change number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Variable to change learning rate")  
    parser.add_argument("--wd", type=float, default=1e-5, help="Variable to change weight decay")
    opt = parser.parse_args()
    return opt

class ConvolutionalAutoencoder(gluon.nn.HybridBlock):
    
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        
        with self.name_scope():
        #encoder
            self.encoder = gluon.nn.HybridSequential()
            with self.encoder.name_scope():
                self.encoder.add(gluon.nn.Conv2D(32, 5, activation='relu'))
                self.encoder.add(gluon.nn.MaxPool2D(2))
                self.encoder.add(gluon.nn.Conv2D(32, 5, activation='relu'))
                self.encoder.add(gluon.nn.MaxPool2D(2))

        #bottleneck
                self.encoder.add(gluon.nn.Dense(20))

        #decoder
            self.decoder = gluon.nn.HybridSequential()
            with self.decoder.name_scope():
                self.decoder.add(gluon.nn.Dense(32*4*4, activation='relu'))
                self.decoder.add(gluon.nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')))
                self.decoder.add(gluon.nn.Conv2DTranspose(32, 5, activation='relu'))
                self.decoder.add(gluon.nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest')))
                self.decoder.add(gluon.nn.Conv2DTranspose(1, kernel_size=5, activation='sigmoid'))

    def hybrid_forward(self, F, x, label):
        x = self.encoder(x)
        x = self.decoder[0](x)
        x = x.reshape((-1,32,4,4))
        x = self.decoder[1:](x)

        return x, label


def train(batch_size, num_epochs, lr, wd, out_dir):
    
    # define dataloader for MNIST dataset
    transform = lambda x,y: (x.transpose((2,0,1)).astype('float32')/255., y)

    train_dataset = gluon.data.vision.MNIST(train=True)
    test_dataset = gluon.data.vision.MNIST(train=False)

    train_dataset_t = train_dataset.transform(transform)
    test_dataset_t = test_dataset.transform(transform)

    train_dataloader = gluon.data.DataLoader(train_dataset_t, batch_size=batch_size, last_batch='rollover', shuffle=True)
    test_dataloader = gluon.data.DataLoader(test_dataset_t, batch_size=100, last_batch='discard', shuffle=False)
    
    #check if GPU available
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    
    #create autoencoder model
    model = ConvolutionalAutoencoder()
    model.collect_params().initialize(mx.init.Xavier('gaussian'), ctx=ctx)
    #model.hybridize()
    
    #reconstruction loss
    loss_function = gluon.loss.L2Loss()
    
    #trainer
    optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    
    hook = smd.Hook.create_from_json_file()
    
    #training loop
    for epoch in range(num_epochs):
        
        #train
        hook.set_mode(modes.TRAIN)
        for image_batch, label in train_dataloader:
            image = image_batch.as_in_context(ctx)
            with mx.autograd.record():
                output,_ = model(image_batch, label)
                loss = loss_function(output, image_batch)
            loss.backward()
            optimizer.step(image.shape[0])
         
        #test
        hook.set_mode(modes.EVAL)
        for image_batch, label in test_dataloader:
            image = image_batch.as_in_context(ctx)
            output,_ = model(image_batch, label)
            loss = loss_function(output, image_batch)
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, mx.nd.mean(loss).asscalar()))

def main():
    opt = parse_args()
    train(opt.batch_size, opt.num_epochs, opt.lr, opt.wd, opt.out_dir)

if __name__ == "__main__":
    main()

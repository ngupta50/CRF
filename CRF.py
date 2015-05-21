#!/usr/bin/python
import pickle
import gzip
import os
import re
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano import sandbox, Out
cdir= os.path.dirname(os.path.abspath(__file__))
cdir=re.sub("/bin$","",cdir)
sys.path.append(cdir+"/../../utilities")
sys.path.append(cdir+"/../../DeepLearning/lib")
from CRFLayer import CRFLayer
from HiddenLayer import HiddenLayer
from SaveLoadModel import SaveLoadModel
import argparse
theano.config.floatX = 'float32'
theano.config.mode='FAST_RUN'

class CRF(SaveLoadModel):
    def  __init__(self,hiddenLayerSizes="5",inputSize=3, wordVecLen=10, CCLayerSize=0,CVectorSize=2,CCContext=1,MaxWordLen=10,n_out=2 ):
        self.hiddenLayerSizes=hiddenLayerSizes
        self.wordVecLen=wordVecLen
        self.inputSize=inputSize
        self.CVectorSize=CVectorSize
        self.CCLayerSize=CCLayerSize
        self.CCContext=CCContext 
        self.MaxWordLen=MaxWordLen
        self.n_out=n_out
        self.CWV=None
        self.WV=None
        
    def ready(self, wordVecFile=None):
        self.printNetPara()
        self.index = T.lscalar('ind')           # index to a [mini]batch
        self.input = T.matrix('x')              # the data is presented as sequence if index of word in a dictionary
        self.y = T.vector('y')                  # the labels are presented as 1D vector of [int] labels   
        self.cha= T.tensor4('cha')
        self.lr = T.scalar('l_r', dtype=theano.config.floatX)
        self.l1 = T.scalar('l1', dtype=theano.config.floatX)
        self.l2 = T.scalar('l2', dtype=theano.config.floatX)
        self.mom = T.scalar('mom', dtype=theano.config.floatX)
        self.L1 =0 
        self.L2_sqr =0
        self.params=[]
        self.params1=[]
        rng = numpy.random.RandomState(23455)
        
        self.n_hidden=[]
        HL=self.hiddenLayerSizes.split("_")
        for h in HL:
            self.n_hidden.append(int(h))
            
        self.hiddenLayer=[]
        self.hiddenLayer.append(self.FirstLayer(rng, wordVecFile))
        
        if(len(self.n_hidden) >1):
            for i in range(1,len(self.n_hidden)):
                self.hiddenLayer.append(HiddenLayer(rng, self.hiddenLayer[i-1].output, self.n_hidden[i-1] , self.n_hidden[i]))        
        self.CRFLayer = CRFLayer(rng, self.hiddenLayer[len(self.n_hidden)-1].output, self.n_hidden[len(self.n_hidden)-1], self.n_out)
        self.L1+=abs(self.CRFLayer.W).sum()
        self.L2_sqr+=(self.CRFLayer.W ** 2).sum()
        self.params+=self.CRFLayer.params
        self.params1+=self.CRFLayer.params
        for layer in self.hiddenLayer:
            self.L1 += abs(layer.W).sum() 
            self.L2_sqr += (layer.W ** 2).sum()
            self.params += layer.params
            self.params1 += layer.params
        self.negative_log_likelihood = self.CRFLayer.negative_log_likelihood
        self.negative_log_likelihood_Y_Y = self.CRFLayer.negative_log_likelihood_Y_Y
        self.errors = self.CRFLayer.errors
        self.pred=self.CRFLayer.y_pred
        self.params.append(self.CRFLayer.YY)
        
    def printNetPara(self):
        netval=[self.wordVecLen, (self.inputSize-1)/2, self.CCContext, self.MaxWordLen, self.CVectorSize, self.CCLayerSize,  
            self.hiddenLayerSizes, self.n_out]
        netSizes=['Word vector size', 'Word context size', 'Charecter context size in conv. window', 'Maximum word length used for charecter conv.',
              'Charecter vector size', 'Charecter conv. layer size', 'Hidden layer sizes', 'Output size']
        for i, s in   enumerate(netSizes):
            sys.stderr.write(s+"="+str(netval[i])+"\n")
    
    def FirstLayer(self,rng, vecfile):
        if(vecfile!=None):
            f = gzip.open(vecfile, 'rb')
            wv=pickle.load(f)
            self.wordVecLen=wv.shape[1]
            self.WV=theano.shared(numpy.asarray(wv, dtype=numpy.float32), name='WV',borrow=True)
            f.close()
        else:         
            self.WV= theano.shared(value=numpy.asarray(rng.uniform(low=-1, high=1, size=(10000, self.wordVecLen)), dtype=theano.config.floatX),name='WV', borrow=True)    
        self.params.append(self.WV)
        self.params1.append(self.WV)
        self.combinedInp=self.WV[T.cast(self.input,'int32'),]               #The input the layer are indicies in the word/char vector perameters
        self.ConvInSize=self.inputSize*(self.wordVecLen+self.CCLayerSize)
        if(self.CCLayerSize>0):                                                 # Set up the char convolution layer
            charDictSize=100
            self.CWV=theano.shared(numpy.asarray( rng.uniform(low=-1.25, high=1.7, size=(charDictSize,self.CVectorSize)), dtype=theano.config.floatX), name='CWV', borrow=True)
            self.params.append(self.CWV)
            self.params1.append(self.CWV)
            cha_in=(2*self.CCContext+1)*self.CVectorSize
            lm=numpy.sqrt(6. / (cha_in + self.CCLayerSize))
            Cha_W = theano.shared(value=numpy.asarray(rng.uniform(low=-lm, high=lm, size=(cha_in, self.CCLayerSize)), dtype=theano.config.floatX),name='Cha_W', borrow=True)
            Cha_b = theano.shared(value=numpy.zeros((self.CCLayerSize,), dtype=theano.config.floatX), name='Cha_b', borrow=True)
            Cha_output = T.tanh(T.dot(T.reshape(self.CWV[T.cast(self.cha,'int32'),], [self.cha.shape[0],self.cha.shape[1],self.cha.shape[2], cha_in]), Cha_W) + Cha_b)
            CharConvOutput= T.max(Cha_output,axis=2)
            self.params.append(Cha_W)
            self.params.append(Cha_b)
            self.params1.append(Cha_W)
            self.params1.append(Cha_b)
            self.L1 = abs(self.CWV).sum() + abs(Cha_W).sum()
            self.L2_sqr = (self.CWV ** 2).sum()+(Cha_W ** 2).sum()
            self.combinedInp=T.concatenate([self.combinedInp, CharConvOutput], axis=2)
        self.combinedInp=T.reshape(self.combinedInp, [self.input.shape[0], self.ConvInSize])
        return HiddenLayer(rng, self.combinedInp, self.ConvInSize, self.n_hidden[0])
       
    def cost(self):
        return (self.negative_log_likelihood(T.cast(self.y,'int32'))+ self.l1 * self.L1  + self.l2 * self.L2_sqr)
    
    def costYY(self):
        return (T.sum(self.negative_log_likelihood_Y_Y(T.cast(self.y,'int32')))+ self.l1 * self.L1  + self.l2 * self.L2_sqr)
    
    def CompileTraining(self, train_set_x, train_set_char_x, train_set_y, train_segments ):   
        
        cost=self.cost()
        costYY=self.costYY()
        if(theano.config.device=='gpu'):
            cost=theano.sandbox.cuda.basic_ops.gpu_from_host(cost)
            costYY=theano.sandbox.cuda.basic_ops.gpu_from_host(costYY)
        gparams = []
        for param in self.params1:
            gparam = T.grad(cost, param)
            gparams.append(gparam)
        updates = []
        for param, gparam in zip(self.params1, gparams):
            updates.append((param, param - self.lr* gparam))
    
        self.train_model = theano.function(inputs=[self.index, self.lr, self.l2, self.l1],
                outputs=cost,
                updates=updates,
                on_unused_input='ignore',
                givens={self.input: train_set_x[train_segments[self.index]: train_segments[self.index+1]],
                        self.cha: train_set_char_x[T.cast(train_set_x[train_segments[self.index] : train_segments[self.index + 1]],'int32')],
                        self.y: train_set_y[train_segments[self.index]:train_segments[self.index+1]]})
        
        updatesYY = []
        updates.append((self.CRFLayer.YY, self.CRFLayer.YY - self.lr* T.grad(cost, self.CRFLayer.YY)))
    
        self.train_modelYY = theano.function(inputs=[self.index, self.lr, self.l2, self.l1],
                outputs=costYY,
                updates=updatesYY,
                on_unused_input='ignore',
                givens={self.input: train_set_x[train_segments[self.index]: train_segments[self.index+1]],
                        self.cha: train_set_char_x[T.cast(train_set_x[train_segments[self.index] : train_segments[self.index + 1]],'int32')],
                        self.y: train_set_y[train_segments[self.index]:train_segments[self.index+1]]})

    def CompileTesting(self, test_set_x,  test_set_char_x, test_set_y, test_segments):   
        self.test_model = theano.function(inputs=[self.index],
            outputs=self.errors(T.cast(self.y,'int32')),
            on_unused_input='ignore',
            givens={self.input: test_set_x[test_segments[self.index]: test_segments[self.index+1]],
                        self.cha: test_set_char_x[test_segments[self.index]: test_segments[self.index+1],],
                        self.y: test_set_y[test_segments[self.index]:test_segments[self.index+1]]})
        
        self.pred_model = theano.function(inputs=[self.index],
            outputs=self.pred,
            on_unused_input='ignore',
            givens={self.input: test_set_x[test_segments[self.index]: test_segments[self.index+1]],
                        self.cha: test_set_char_x[test_segments[self.index]: test_segments[self.index+1],],
                        self.y: test_set_y[test_segments[self.index]:test_segments[self.index+1]]})
            
    def RunTrainingEpoch(self, n_train, train_len, learning_rate, l2_reg, l1_reg,Sap=False ):
        score=[]
        sys.stderr.write(" ")
        for i in range(n_train):
            if(train_len[i]>1):   # GRADIENT CALCULATIONS SOMEHOW DO NOT WORK IF LENGTH OF THE EXAMPLE IS LESS THAN 2
                loss=self.train_model(i,learning_rate, l2_reg, l1_reg)
                loss+=self.train_modelYY(i,learning_rate, l2_reg, l1_reg)
                score.append(numpy.asarray(loss))
            if(i%5000==0 and i>0):
                sys.stderr.write(" %i:%f" % (i, numpy.mean(score)))
        sys.stderr.write("\n")
        return numpy.mean(score)
    
    def TestClassifier(self, n_test, test_segments):
        errors=[]
        for i in xrange(n_test):
            err=self.test_model(i)
            errors.extend(err)
        WordAcc=(1.0-numpy.mean(errors))*100.0
        
        seg=test_segments.get_value()
        count=seg.shape[0]-1
        corrSeg=numpy.sum([numpy.sum(errors[seg[i]:seg[i+1]])==0 for i in range(count)])
        SegAcc=corrSeg*100.0/count
        return(WordAcc, SegAcc)
    
    def testModel(self,dataFile):
        sys.stderr.write('... Loading Data\n')
        n_test, test_text, test_set_char_x, test_labs, test_segments, test_len = self.shared_dataset(dataFile)
        sys.stderr.write('... Compiling test code\n')
        self.CompileTesting( test_text,  test_set_char_x, test_labs, test_segments)
                               # Test the model
        (WordAcc, SegAcc)=self.TestClassifier(n_test,test_segments)
        sys.stderr.write(('     Word Accuracy = %f%% Segment Accuracy= %f%%\n') % (WordAcc, SegAcc))
        
    def Classify(self,dataFile, dictfile):
        print '... Loading Data'
        n_test, test_text, test_set_char_x, test_labs, test_segments, test_len = self.shared_dataset(dataFile)
        print '... Compiling test code'
        self.CompileTesting( test_text,  test_set_char_x, test_labs, test_segments)
        f = gzip.open(dictfile, 'rb')
        word_dict=  pickle.load(f)
        lab_dict=  pickle.load(f)
        f.close()
        Wrds=numpy.asarray(test_text.get_value(borrow=True)[:,2], dtype=numpy.int32)
        words=numpy.asarray(map(lambda (k,v): k, sorted(word_dict.items(),key=lambda (k, v): v)))[Wrds]
        labDic=numpy.asarray(map(lambda (k,v): k, sorted(lab_dict.items(),key=lambda (k, v): v)))
        seg=test_segments.get_value(borrow=True)
        labs=test_labs.get_value(borrow=True)
        for i in range(n_test):
            pred=self.pred_model(i)
            ref=labDic[numpy.asarray(labs[seg[i]:seg[i+1]], dtype=numpy.int32)]
            hyp=labDic[pred]
            if(ref!=hyp):
                print "Sentance "+str(i)
                for i, r in enumerate(ref):
                    if(r==hyp[i]):
                        print words[i]+" "+r+" "+hyp[i]
                    else:
                        print words[i]+" "+r+" "+hyp[i]+" <=="
                print '\n'
        
    def fit(self, trainFile, n_epochs, LearningRate, LearningRateDecay, l2=0, l1=0, testFile=None):
        sys.stderr.write( '... Loading Data\n')
        n_test=0
        if testFile is not None:
            n_test, test_set_x, test_set_char_x,  test_set_y,  test_segments, test_len= self.shared_dataset(testFile)
        n_train, train_set_x, train_set_char_x, train_set_y, train_segments, train_len = self.shared_dataset(trainFile)
        sys.stderr.write( '... Compiling networks\n')
        self.CompileTraining(train_set_x, train_set_char_x, train_set_y, train_segments)
        if (n_test>0):
            self.CompileTesting(test_set_x, test_set_char_x, test_set_y, test_segments) 
        
        epoch=0
        while (epoch < n_epochs) :
            epoch = epoch + 1
            lr=LearningRate/epoch
            sys.stderr.write( '... Running training epoch %i at lr %f ' % (epoch, lr ))
            avg_train_cost = self.RunTrainingEpoch(n_train, train_len, lr,l2 , l1)
            if(n_test>0):
                (WordAcc, SegAcc) = self.TestClassifier(n_test,test_segments)
                avg_test_error=100.0-WordAcc
                sys.stderr.write(' tr loss %f vld errors %f%% \n' % (avg_train_cost, avg_test_error))
            else:
                sys.stderr.write(' tr loss %f\n' % avg_train_cost) 
            
            
    def shared_dataset(self, datafile):
        f = gzip.open(datafile, 'rb')
        data= pickle.load(f)
        segments = pickle.load(f)
        labels = pickle.load(f)
        chardata=pickle.load(f)
        f.close()
        numExample=segments.shape[0]-1
        seg_len=[]
        for i in range(numExample):
            seg_len.append(segments[i+1]-segments[i])   
        shared_x = theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=True)
        shared_char = theano.shared(numpy.asarray(chardata,dtype=theano.config.floatX),borrow=True)
        shared_y = theano.shared(numpy.asarray(labels,dtype=theano.config.floatX),borrow=True)
        shared_segments = theano.shared(numpy.asarray(segments,dtype=numpy.int32),borrow=True)
        return numExample, shared_x, shared_char, shared_y, shared_segments, seg_len
     
def ParseArguments():
    parser = argparse.ArgumentParser(description="Trains/ test a Multilayer Percetpron network, using fixed length word strings.")
    parser.add_argument("-WD","--WorkDir", type=str, default=".",
                        help="Working Directory relative to which all other files are specified. Default is current directory. ")
    parser.add_argument("-TR","--TrainingDataFile", type=str, help="Name of the pre-processed Training data file in pickle format. Default is None")
    parser.add_argument("-TS","--TestDataFile", type=str, help="Name of the pre-processed test data file in pickle format. Default is None")
    parser.add_argument("-LM","--LoadModelFile", type=str, help="Name of the model file from which model must be loaded. Default is None")
    parser.add_argument("-SM","--SaveModelfile", type=str, help="A Stem used to contruct the file name where trained model is stored.")
    
    parser.add_argument("-WV","--WordVecFile", type=str, help="Name of the word vector file.")
    parser.add_argument("-WVS","--WordVecSize", type=int, default=100, help="Word embedding vector size, default=100")
    parser.add_argument("-C","--ContextSize", type=int, default=2, help="Word context on left and right. \
                                                                    This gives fixed string size of(2*ContextSize+1),Default=2")
    parser.add_argument("-CC","--CCContext", type=int, default=2, help="Charecter convolution context on left and right. default=2")
    parser.add_argument("-MW","--MaximumWordLen", type=int, default=10, help="For efficiency reason charecter convolution is designed to \
                        run on a fixed word length only. Therefore a mximum word length is used to pre-process the data.\
                        Even though this parameter is used only during pre-processing the training and testing data and is not used by the network,\
                        it is stored in the classifier model so that it can be recovered for runtime pre-processing of the data.\n")
    parser.add_argument("-CV","--CVecSize", type=int, default=10, help="Size of the embedded charecter vectors. default=10")
    parser.add_argument("-CL","--CCLayerSize", type=int, default=50, help="Charecter Convolution layer output size. default=50.\
                                        No char convolution is performed if this is set to 0.")
    parser.add_argument("-HL","--HiddenLayer", type=str, default="300", help="Sizes of the hidden layers conjoined by _. Default=300")
    parser.add_argument("-O","--OutputSize", type=int, default=47, help="Number of output. Default=47")
    
    parser.add_argument("-L","--LearningRate", type=float, default=0.0075, help="Initial learning rate. Default=0.0075")
    parser.add_argument("-LD","--LearningRateDecay", type=float, default=0.80, help="Learning rate decay after each epoch. Default=0.80")
    parser.add_argument("-L2","--L2Reg", type=float, default=0.001, help="L2 Regularization weight. Default =.001")
    parser.add_argument("-L1","--L1Reg", type=float, default=0.000, help="L1 Regularization Parameter. Default =.00")
    
    parser.add_argument("-IM","--Initial_momentum", type=float, default=0.5, help="Initial gradient momentum =.5")
    parser.add_argument("-FM","--Final_momentum", type=float, default=0.9, help="Final gradient momentum =.9")
    parser.add_argument("-MS","--Momentum_switchover", type=int, default=5, help="Number of epoch after which momentum is switched default=5")
    parser.add_argument("-N","--NumEpoch", type=int, default=20, help="Number of epoch to be run for training. Default=20")
    
    parser.add_argument("-D","--DictionaryStem", type=str, help="Stem of a .dic file and a .vec file. .dic file contain indecies of the words \
                        in the in the training data. and the .vec contains their vector representation. Needed only when computing errors.")
    
    parser.add_argument("-ER","--Errors", action="store_true", help="Print word index, word, ref tag, hypothsis tag only for Error cases")
    
    args = parser.parse_args()
    
    argval=[args.WorkDir , args.TrainingDataFile, args.TestDataFile, args.LoadModelFile, args.SaveModelfile, args.WordVecFile,
            args.LearningRate, args.LearningRateDecay, args.L2Reg, args.L1Reg, args.NumEpoch]
    arg=['Working directory','Training data file', 'Test data File', 'Load model file', 'Save model file', 'Word vector file',
         'Learning rate', 'Learning rate decay','L2 regularization weight','L1 regularization weight','Number of training Epoch']
    
    sys.stderr.write("\n Parameters\n")
    sys.stderr.write(" ----------\n")
    for i, a in enumerate(arg):
        sys.stderr.write(a+"="+str(argval[i])+"\n")
    
    args.WorkDir+='/'
    if(args.TrainingDataFile!=None):
        args.TrainingDataFile=args.WorkDir+args.TrainingDataFile
    if(args.TestDataFile!=None):
        args.TestDataFile=args.WorkDir+args.TestDataFile
    if(args.LoadModelFile!=None):
        args.LoadModelFile=args.WorkDir+args.LoadModelFile
    if(args.SaveModelfile!=None):
        args.SaveModelfile=args.WorkDir+args.SaveModelfile
    if(args.WordVecFile!=None):
        args.WordVecFile=args.WorkDir+args.WordVecFile
    if(args.DictionaryStem!=None):
        args.DictionaryStem=args.WorkDir+args.DictionaryStem+".dic.pcl.gz"
    return args

if __name__ == '__main__': 
    args=ParseArguments()
        
    Classifier=None
    if(args.LoadModelFile!=None):
        Classifier=CRF()
        Classifier.load(args.LoadModelFile)
    if(args.TrainingDataFile!=None):
        if(args.LoadModelFile==None):
            if(args.WordVecFile==None):
                sys.stderr.write("No word embedding provided. Random embedding will be used for vocabulary size of 10000\n")
            Classifier=CRF(args.HiddenLayer,2*args.ContextSize+1,args.WordVecSize, args.CCLayerSize,args.CVecSize, args.CCContext,args.MaximumWordLen,args.OutputSize)
            Classifier.ready(args.WordVecFile)    
        Classifier.fit(args.TrainingDataFile,args.NumEpoch,args.LearningRate,args.LearningRateDecay,args.L2Reg, args.L1Reg, args.TestDataFile)
        Classifier.testModel(re.sub("valid","test",args.TestDataFile))
        modelSpecs=str(args.HiddenLayer)+"_"+ str(args.CCLayerSize)+"_"+str(args.LearningRate)+"_"+str(args.LearningRateDecay)+"_"\
                +str(args.L2Reg)+"_"+str(args.NumEpoch)+".plk"
        Classifier.save(args.SaveModelfile+modelSpecs)
    elif(args.TestDataFile!=None and args.LoadModelFile!=None):
        if(args.Errors):
            if(args.DictionaryStem!=None):
                Classifier.Classify(args.TestDataFile, args.DictionaryStem)
            else:
                sys.stderr.write("Dictionary file is needed for error identification. Please specify it with -D option")
        else:
            Classifier.testModel(args.TestDataFile)


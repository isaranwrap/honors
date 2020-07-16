import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnParamsFormatConverterLSTM
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.python.ops import array_ops
from tensorflow.contrib import seq2seq
import numpy as np
from copy import copy,deepcopy
from random import randint,sample

class BehavioralStater:
    def __init__(self,params,mode='const',train_init=False,init_h=0.1,init_c=0.1):
        self.mode=mode
        self.name=params['name']
        self.type = params['type']
        self.epochs = params['epochs']
        self.nlayers = params['nlayer']
        self.width = params['width']
        self.dim = params['dim']
        self.batch_size = params['batch_size']
        self.seq_len = params['seq_len']
        self.lr = params['lr']
        self.eps = params['eps']
        self.cap = params['cap']
        self.weight_decay = params['decay']


        if mode == 'schedule':
            self.n = params['n']
            self.red = params['red'] 
        if train_init:
            self.train_init=1
        #self.train_graph = tf.Graph()
        
        self.sess = tf.Session()
  
        #with self.train_graph.as_default():
        self.changer = CudnnParamsFormatConverterLSTM(1,params['width'],params['dim'])
        self.input_data=array_ops.placeholder(tf.float32,[None,None,self.dim],name='input')
        self.target=array_ops.placeholder(tf.float32,[None,None,self.dim],name='target')
        self.tr_target=tf.transpose(self.target,perm = [1,0,2])
        input_shape = tf.shape(self.input_data)
        if train_init:
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                self.h = tf.get_variable('h',[self.width,],dtype=tf.float32,initializer=tf.initializers.random_uniform(minval=-init_h,maxval=init_h))#initializer=tf.keras.initializers.glorot_normal()
                self.c =tf.get_variable('c',[self.width,],dtype=tf.float32,initializer=tf.initializers.random_uniform(minval=-init_c,maxval=init_c))
            x1 = tf.expand_dims(tf.reshape(  tf.tile(self.h,input_shape[1:2]), [input_shape[1],self.width]),0)
            x2 = tf.expand_dims(tf.reshape( tf.tile(self.c,input_shape[1:2]), [input_shape[1],self.width] ),0)
            self.initial_state = (x1,x2)
        else:
            h = array_ops.placeholder(tf.float32,[1,None,self.width])
            c = array_ops.placeholder(tf.float32,[1,None,self.width])
            self.initial_state = (h,c)
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.cell =       tf.contrib.cudnn_rnn.CudnnLSTM(1,self.width,dtype=tf.dtypes.float32,kernel_initializer=params['initializer'])
            self.cell.build([None,None,self.dim])            
            self.outputs,self.output_state = self.cell(self.input_data,initial_state = self.initial_state,training=True)
            self.logits = tf.contrib.layers.fully_connected(self.outputs,self.dim,activation_fn = None, weights_regularizer=l2_regularizer(self.weight_decay))
            self.probs = tf.nn.softmax(self.logits,name='probs')
            self.labels = tf.reshape(tf.where(tf.equal(self.tr_target,1))[:,2],[input_shape[1],input_shape[0]])
            self.cost = seq2seq.sequence_loss(tf.transpose(self.logits,perm=  [1,0,2]),self.labels,tf.ones([input_shape[1],input_shape[0]]))
            if mode=='const':
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                self.gradients = self.optimizer.compute_gradients(self.cost)
                self.capped_gradients = [(tf.clip_by_value(grad, -self.cap, self.cap), var) for grad, var in self.gradients if grad                                        is not None]
                self.train_op = self.optimizer.apply_gradients(self.capped_gradients)
            if mode =='schedule':
                self.global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = self.lr
                self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                           self.n, self.red, staircase=True) #(750,0.9): 0.1998
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.gradients = self.optimizer.compute_gradients(self.cost)#,global_step=self.global_step)
                self.capped_gradients = [(tf.clip_by_value(grad, -self.cap, self.cap), var) for grad, var in self.gradients if grad                                        is not None]
                self.train_op = self.optimizer.apply_gradients(self.capped_gradients,global_step=self.global_step)
            

       
         #
    def ZeroState(self,batch_size=None):
        if batch_size==None:
            batch_size = self.batch_size
        input_h = np.zeros((1,batch_size,self.width))
        input_c = np.zeros((1,batch_size,self.width))
        initial_state = (input_h,input_c)
        return initial_state
    
    def RandomState(self,batch_size=None,loc_c=0,loc_h=0,scale_c=0.01,scale_h=0.01):
        if batch_size==None:
            batch_size = self.batch_size
        input_h = np.random.normal(loc=loc_h,scale = scale_h,size=(1,batch_size,self.width))
        input_c = np.random.normal(loc=loc_c,scale = scale_c,size=(1,batch_size,self.width))
        initial_state = (input_h,input_c)
        return initial_state
  
    def get_batches(self,data,time=None):
        # this function will batch data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], batch_size=3,seq_len=2
        # it will turn it into: 1st batch - [[1,2],[7,8],[13,14]], 2nd batch - [[3,4],[9,10],[15,16]], 3rd batch [[5,6],                   [11,12],[17,18]]
        # and targets - [[2,3],[8,9],[14,15]] ,            [[4,5],[10,11],[16,17]],          [[6,7],[12,13],[18,19]]
        
        #data_h = np.vstack(data)
        X = []
        Y = []
        if time==None:
            t = self.seq_len
        else:
            t = time
        num_batch = int(len(data)/(self.batch_size*t))
        for i in range(num_batch):
            X.append([])
            Y.append([])
            m=i*t
            for k in range(self.batch_size): 
                X[i].append(copy(data[m:m+t].astype(int)))
                Y[i].append(copy(data[m+1:m+t+1].astype(int)))
                m+=num_batch*t
            X[i] = np.transpose(np.array(X[i]),axes=(1,0,2))
            Y[i] = np.transpose(np.array(Y[i]),axes=(1,0,2))
        return [X, Y]    
    
    
    def train(self,data_tr,data_val,saver,path,init_state='Zero',stateful=True,dropout=None,loc_c=0,loc_h=0,scale_c=0.01,scale_h=0.01,ver=1):
        X_train = data_tr[0]
        Y_train = data_tr[1]
        if len(data_val[0])!=0:
            X_val = np.hstack(data_val[0])
            Y_val = np.hstack(data_val[1])
        epoch=0
        val_losses=[]
        loss_min = np.inf
        with self.sess.as_default():
            tf_weights = self.sess.run(self.changer.opaque_to_tf_canonical(self.cell.get_weights()[0]))
            to_op = []
            for i in range(len(tf_weights[0])):
                to_op.append(tf_weights[0][i])
            for i in range(len(tf_weights[1])):
                tf_weights[1][i][:]=0
                tf_weights[1][i][2*self.width:3*self.width] = 1
                to_op.append(tf_weights[1][i])
            to_op = tuple(to_op)
            op_weights = self.sess.run(self.changer.tf_canonical_to_opaque(to_op))
#             tf_weights[1][0][:]=0
#             tf_weights[1][0][2*self.width:3*self.width] = 1
#             op_weights = self.sess.run(self.changer.tf_canonical_to_opaque((tf_weights[0][0],tf_weights[1][0])))
            self.cell.set_weights([op_weights])
            while True:
                if len(data_val[0])!=0:
                    if init_state!='var':
                        if init_state=='Zero':
                            initial_state = self.ZeroState(X_val.shape[1])
                        else:
                            initial_state = self.RandomState(X_val.shape[1],loc_c=loc_c,loc_h=loc_h,scale_c=scale_c,scale_h=scale_h)

                        feed = {self.input_data: X_val, self.target: Y_val,self.initial_state:initial_state}
                    else:
                        feed = {self.input_data: X_val, self.target: Y_val}
                else:
                    if init_state!='var':
                        if init_state=='Zero':
                            initial_state = self.ZeroState(X_tr.shape[1])
                        else:
                            initial_state = self.RandomState(X_tr.shape[1],loc_c=loc_c,loc_h=loc_h,scale_c=scale_c,scale_h=scale_h) 
                        feed = {self.input_data: X_tr, self.target: Y_tr,self.initial_state:initial_state}
                    else:
                        feed = {self.input_data: X_tr, self.target: Y_tr}
                val_loss = self.sess.run(self.cost,feed)   
                if val_loss<loss_min:
                    loss_min = val_loss
                    save_path = saver.save(self.sess, path)

                if len(val_losses)==20 or epoch==self.epochs:
                    val_losses = val_losses[1:]
                    if loss_min==val_losses[0] or epoch == self.epochs:
                        break
                val_losses.append(val_loss)
                if ver:
                    if init_state!='var':
                        if init_state=='Zero':
                            initial_state = self.ZeroState()
                        else:
                            initial_state = self.RandomState(loc_c=loc_c,loc_h=loc_h,scale_c=scale_c,scale_h=scale_h) 
                        feed = {self.input_data: X_train[0], self.target: Y_train[0],self.initial_state:initial_state}
                    else:
                        feed = {self.input_data: X_train[0], self.target: Y_train[0]}
                    train_loss = self.sess.run(self.cost,feed)
                    print('Epoch {:>3} train_loss = {:.6f} val_loss = {:.6f}'.format(
                       epoch,
                       train_loss,
                       val_losses[-1]
                      ))
          
            #if (epoch>2 and (val_losses[-2] - val_losses[-1])<self.eps) or epoch>=self.epochs:
            #if epoch>=self.epochs:
                #break
                if init_state=='Zero':
                    last_state = self.ZeroState()
                if init_state=='Random':
                    last_state = self.RandomState(loc_c=loc_c,loc_h=loc_h,scale_c=scale_c,scale_h=scale_h)
            
            #state = 
            #h_ev = np.zeros((self.batch_size,self.width))
            #c_ev = np.zeros((self.batch_size,self.width))
                for i, b in enumerate(X_train):
                    if init_state=='Zero' or init_state=='Random':
                        feed = {self.input_data: b, self.target: Y_train[i],self.initial_state:last_state}
                    else:
                        feed = {self.input_data: b, self.target: Y_train[i]}
                    train_loss,state,_ = self.sess.run([self.cost,self.output_state,self.train_op],feed)
                    if stateful:
                        last_state = state
                    else:
                        if init_state=='Zero':
                            last_state = self.ZeroState()
                        if init_state=='Random':
                            last_state = self.RandomState(loc_c=loc_c,loc_h=loc_h,scale_c=scale_c,scale_h=scale_h)                       
                if self.mode=='schedule':
                    print(self.sess.run(self.learning_rate,feed))
                #h_ev = state.h
                #c_ev = state.c
                epoch+=1
                self.global_step = epoch
        return val_losses,loss_min            
                    
                    
    def predict_states(self,data,initial_state=[],train_init = False):
        input_shape = tf.shape(self.input_data)
        if not train_init:
            if len(initial_state)==0: 
                initial_state=self.ZeroState(batch_size=data.shape[1])
            feed = {self.input_data:data,self.initial_state:initial_state}
        else:
            feed = {self.input_data:data}
        
        states,full_state=self.sess.run([self.outputs,self.output_state],feed)

        return states,full_state
        
    def generate(self,time,mode = 'classical',slim = False,p_initial = None,batch=1,h_initial=[],c_initial=[],loc_h=0,loc_c=0,scale_h=0.2,scale_c=0.2,train_init=False,scale_noise=-1,inds_to_ignore=[None]):
        if not isinstance(p_initial,(np.ndarray,list)):
            index = np.random.randint(0,self.dim-1,size = batch)
            behavior = np.array([np.eye(self.dim)[index]])
        elif len(p_initial.shape)==1:
            index = p_initial
            if index[0]==-1:
                behavior = np.zeros((1,batch,self.dim))
            else:
                behavior = np.array([np.eye(self.dim)[index]])
        elif len(p_initial.shape)==3:
                behavior = p_initial
        if not train_init:
            if len(h_initial)==0: 
                h_initial=np.random.normal(loc_h,scale_h,size=(1,batch,self.width))
            if len(c_initial)==0:
                c_initial=np.random.normal(loc_c,scale_c,size=(1,batch,self.width))
            state_ev = (h_initial,c_initial)
        else:
            state_ev = self.sess.run(self.initial_state,{self.input_data:behavior})
            
            
        probs_ev_list = np.zeros((time,batch,self.dim))
        if not slim:
            state_ev_list = np.zeros((time,batch,self.width))
            c_ev_list = np.zeros((time,batch,self.width))
        if mode == 'quantum':
            behavior_list = np.zeros((batch,time))
            elements = range(self.dim)
            
        if inds_to_ignore[0]:
            behavior[:,:,inds_to_ignore]=0
        for t in range(time):
            probs_ev,state_ev = self.sess.run([self.probs,self.output_state],{self.input_data:behavior,self.initial_state:state_ev})
            probs_ev_list[t,:,:] = probs_ev[0]
            if not slim:
                state_ev_list[t,:,:] = state_ev[0][0]
                c_ev_list[t,:,:] = state_ev[1][0]
            
            if mode == 'classical':
                behavior = deepcopy(probs_ev)
                if scale_noise>0:
                    noise = np.random.normal(scale = scale_noise,size=behavior.shape)
                    behavior = behavior + noise
            if mode == 'quantum':
                index = np.zeros(batch).astype(int)
                if inds_to_ignore[0]:
                    probs_ev[:,:,inds_to_ignore]=0
                    probs_ev = probs_ev/np.expand_dims(probs_ev.sum(axis=2),axis=-1)
                for b in range(batch):
                    index[b] = np.random.choice(elements, 1, p=probs_ev[0,b])[0]
                behavior = np.array([np.eye(self.dim)[index]])
                behavior_list[:,t] = index
        if mode == 'classical':
            if not slim:
                return probs_ev_list, state_ev_list,c_ev_list
            else:
                return probs_ev_list,state_ev
        if mode == 'quantum':
            if not slim:
                return behavior_list,probs_ev_list, state_ev_list,c_ev_list
            else:
                return behavior_list,probs_ev_list, state_ev


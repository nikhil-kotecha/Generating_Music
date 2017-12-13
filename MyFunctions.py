import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple

#def Input_Kernel_numpy(input_data, Midi_low=24, Midi_high=102):
"""
Arguments:
Note_State_Batch: numpy size = [batch_size x num_notes x num_timesteps]
Midi_low: integer
Midi_high: integer
    
With numpy, 
"""    
#return Note_State_Expand, prev_input


def Input_Kernel(input_data, Midi_low, Midi_high, time_init=0):
    """
    Arguments:
        input_data: size = [batch_size x num_notes x num_timesteps x 2] 
        Midi_low: integer
        Midi_high: integer
    the input data represents that at the previous timestep of what we are trying to predict
    """    

    
    # batch_size and num_timesteps are variable length
    batch_size = tf.shape(input_data)[0]
    num_notes = input_data.get_shape()[1].value
    num_timesteps = tf.shape(input_data)[2]
    
    
    
    # MIDI note number
    Midi_indices = tf.squeeze(tf.range(start=Midi_low, limit = Midi_high+1, delta=1))
    x_Midi = tf.ones((batch_size, num_timesteps, 1, num_notes))*tf.cast(Midi_indices, dtype=tf.float32)
    x_Midi = tf.transpose(x_Midi, perm=[0,3,1,2]) # shape = batch_size, num_notes, num_timesteps, 1
    #print('x_Midi shape = ', x_Midi.get_shape())

    
    # part_pitchclass
    Midi_pitchclasses = tf.squeeze(x_Midi % 12, axis=3)
    #Midi_pitchclasses = tf.squeeze(x_Midi % 12, axis = 3)
    x_pitch_class = tf.one_hot(tf.cast(Midi_pitchclasses, dtype=tf.uint8), depth=12)
    #print('x_pitch_class shape = ', x_pitch_class.get_shape())
    
   
    # part_prev_vicinity (1 time step in past)
    input_flatten = tf.transpose(input_data, perm=[0,2,1,3])
    input_flatten = tf.reshape(input_flatten, [batch_size*num_timesteps, num_notes, 2]) # channel for play and channel for articulate
    input_flatten_p = tf.slice(input_flatten, [0,0,0],size=[-1, -1, 1])
    input_flatten_a = tf.slice(input_flatten, [0,0,1],size=[-1, -1, 1])
    
    # reverse identity kernel
    filt_vicinity = tf.expand_dims(tf.eye(25), axis=1)

   
    #1D convolutional filter for each play and articulate arrays 
    vicinity_p = tf.nn.conv1d(input_flatten_p, filt_vicinity, stride=1, padding='SAME')
    vicinity_a = tf.nn.conv1d(input_flatten_a, filt_vicinity, stride=1, padding='SAME')    
    
    #concatenate back together and restack such that play-articulate numbers alternate
    vicinity = tf.stack([vicinity_p, vicinity_a], axis=3)
    vicinity = tf.unstack(vicinity, axis=2)
    vicinity = tf.concat(vicinity, axis=2)
    
    #reshape by major dimensions, THEN swap axes
    x_vicinity = tf.reshape(vicinity, shape=[batch_size, num_timesteps, num_notes, 50])
    x_vicinity = tf.transpose(x_vicinity, perm=[0,2,1,3])
    #print('x_prev vicinity shape = ', x_vicinity.get_shape())  
   

    #part_prev_context (1 time step in past)
    input_flatten_p_bool = tf.minimum(input_flatten_p,1) 
    # 1 if note is played, 0 if not played.  Don't care about articulation
    
    #kernel
    filt_context = tf.expand_dims(tf.tile(tf.eye(12), multiples=[(num_notes // 12)*2,1]), axis=1)
    #print('filt_context size = ', filt_context.get_shape())
       
    context = tf.nn.conv1d(input_flatten_p_bool, filt_context, stride=1, padding='SAME')
    x_context = tf.reshape(context, shape=[batch_size, num_timesteps, num_notes, 12])
    x_context = tf.transpose(x_context, perm=[0,2,1,3])
    #print('x_prev context shape = ', x_prev_context.get_shape())    
    
    
    
    #beat - for music generation, need to keep track of 't' from batch to batch (timestep to timestep)
    Time_indices = tf.range(time_init, num_timesteps + time_init)
    x_Time = tf.reshape(tf.tile(Time_indices, multiples=[batch_size*num_notes]), shape=[batch_size, num_notes, num_timesteps,1])
    x_beat = tf.cast(tf.concat([x_Time%2, x_Time//2%2, x_Time//4%2, x_Time//8%2], axis=-1), dtype=tf.float32)
    #print('x_beat shape = ', x_beat.get_shape()) 
    
    #zero
    x_zero = tf.zeros([batch_size, num_notes, num_timesteps,1])


    #Final Vector
    Note_State_Expand = tf.concat([x_Midi, x_pitch_class, x_vicinity, x_context, x_beat, x_zero], axis=-1)
    
    
    return Note_State_Expand, time_init




def LSTM_TimeWise_Training_Layer(input_data, state_in):
    """
    Arguments:
        input_data: Tensor with size = [batch_size x num_notes x num_timesteps x input_size]
        initial_state: Tuple of 2 Tensors each with size =  [batch_size x num_notes x num_units]
        
    # LSTM time-wise 
    # This section is the 'Model LSTM-TimeAxis' block and will run a number of LSTM cells over the time axis.
    # Every note and sample in the batch will be run in parallel with the same LSTM weights
    # The input data is 'Note_State_Filt' with dimensions batch_size x num_notes x num_timesteps x input_size
    # The output will be 'Hid_State_Final' with dimensions batch_size x num_notes x num_timesteps x num_units

    
    # Reshape the input
    # batch_size and num_notes dimensions of input are flattened to treat as single 'batch' dimension for LSTM cell
    # will be reshaped at the end of this block for the next stage
  """  
    
    # batch_size and num_timesteps are variable length
    batch_size = tf.shape(input_data)[0]
    num_notes = input_data.get_shape()[1].value
    num_timesteps = tf.shape(input_data)[2]
    input_size = input_data.get_shape()[3].value
    
    c_in, h_in = state_in
    num_units = c_in.get_shape()[2].value
    
    
    #Flatten input_data and state_in to allow note steps to be run in parallel as separate batches for this segment
    input_flatten = tf.reshape(input_data, shape=[batch_size*num_notes, num_timesteps, input_size])
    c_in_flatten = tf.reshape(c_in, shape=[batch_size*num_notes, num_units])
    h_in_flatten = tf.reshape(h_in, shape=[batch_size*num_notes, num_units])    
    state_in_flatten = LSTMStateTuple(c_in_flatten, h_in_flatten)
    
    #Instantiate Time-Wise Cell
    lstmcell_time = BasicLSTMCell(num_units=num_units, forget_bias=1.0, state_is_tuple=True,activation=math_ops.tanh, reuse=None)


    #Run through LSTM time steps and generate time-wise sequence of outputs
    out_flat, (c_flat, h_flat) = tf.nn.dynamic_rnn(lstmcell_time, input_flatten, initial_state=state_in_flatten, dtype=tf.float32)


    #Unflatten the 1st 2 dimensions [Lbatch, Nnotes, num_timesteps, num_units]
    output = tf.reshape(out_flat, shape=[batch_size, num_notes, num_timesteps, num_units])
    h = tf.reshape(h_flat, shape=[batch_size, num_notes, num_units])    
    c = tf.reshape(c_flat, shape=[batch_size, num_notes, num_units])     
    
    state_out = LSTMStateTuple(c, h)
    
    return output, state_out


def LSTM_NoteWise_Layer(input_data, num_class=2):
    """
    Arguments:
        input_data: size = [batch_size x num_notes x num_timesteps x num_hidden_units]
        num_class: number of possible integer values for Note_State_Batch entries
          
    # LSTM note-wise
    # This section is the 'Model LSTM-Note Axis' block and runs a number of LSTM cells from low note to high note
    # A batches and time steps are run in parallel in
    # The input sequence to the LSTM cell is the hidden state output from the previous block for each note
    #  concatenated with a sampled output from the previous note step
    # The input data is 'Hid_State_Final' with dimensions batch_size x num_notes x num_timesteps x num_units
    # The output will be:
    #    - LogP with dimensions batch_size x num_notes x num_timesteps x 3
    #    - note_gen with dimensions batch_size x num_notes x num_timesteps x 1

    # number of outputs is number of note+articulation combinations
    #    - 0: note is not played
    #    - 1: note is played and held
    #    - 2: note is played and articulated
    """
    
    # batch_size and num_timesteps are variable length
    batch_size = tf.shape(input_data)[0]
    num_notes = input_data.get_shape()[1].value
    num_timesteps = tf.shape(input_data)[2]
    num_units = input_data.get_shape()[3].value

    # Reshape the input
    # batch_size and num_timesteps dimensions of input are flattened to treat as single 'batch' dimension for LSTM cell
    notewise_in = tf.reshape(input_data, [batch_size*num_timesteps, num_notes, num_units])


    # For this LSTM cell, can't use tf.nn.dynamic_rnn because samples have to be generated and fed back to for subsequent notes
    # need to feed the generated output for note 'n-1' into the generation of note 'n'
    # Will use 'for' loop and call the LSTM cell each time


    #Instantiate Note-Wise Cell
    lstmcell_note = BasicLSTMCell(num_units=4, forget_bias=1.0, state_is_tuple=True,activation=math_ops.tanh, reuse=None)


    #Set values for initial LSTM state and sampled note.  Zero notes always played below bottom note
    logP_state_initial = tf.zeros([batch_size*num_timesteps, 4])
    pa_gen_initial = tf.zeros([batch_size*num_timesteps,2, 1])

    logP_n_state = LSTMStateTuple(logP_state_initial, logP_state_initial) #(c, h)
    pa_gen_n_unflat = pa_gen_initial

    logP_out_list=[]
    pa_gen_out_list=[]
    #print('notewise shape = ', notewise_in.get_shape())
    #print('pa_gen_n shape = ', pa_gen_n_unflat.get_shape())
    
    
    
    #Run through notes for note-wise LSTM to obtain P(va(n) | va(<n))
    for n in range(num_notes):    
        #concatenate previously sampled note play-articulate-combo with timewise output
        p_gen = tf.squeeze(tf.slice(pa_gen_n_unflat, [0,0,0],[-1,1,-1]), axis=2)
        a_gen = tf.squeeze(tf.slice(pa_gen_n_unflat, [0,1,0],[-1,1,-1]), axis=2)
        cell_inputs = tf.concat([notewise_in[:,n,:], tf.cast(p_gen, dtype=tf.float32),  tf.cast(a_gen, dtype=tf.float32)], axis=1)
        #print('Cell inputs shape = ', cell_inputs.get_shape())
        #print('logP_n_state shape = ', logP_n_state[0].get_shape())
        
        # Run single note step of LSTM cell and reshape output from [batch_size*num_timesteps, 4] [batch_size*num_timesteps*2, 2]
        logP_n_out, logP_n_state = lstmcell_note(cell_inputs, logP_n_state)       
        logP_n_out_unflat = tf.reshape(logP_n_out, shape=[batch_size*num_timesteps*2, 2])
        
        # Sample play and articulate from logPout distribution and reshape 
        pa_gen_n = tf.multinomial(logits=logP_n_out_unflat, num_samples=1)          
        pa_gen_n_unflat = tf.reshape(pa_gen_n, shape=[batch_size*num_timesteps, 2, 1])
        
        #Append to notewise list
        logP_out_list.append(logP_n_out_unflat)
        pa_gen_out_list.append(pa_gen_n_unflat)
    
    # Convert output list to a Tensor
    logP_out = tf.reshape(tf.stack(logP_out_list, axis=1), [batch_size, num_notes, num_timesteps, 2, 2])
    pa_gen_out = tf.reshape(tf.stack(pa_gen_out_list, axis=1),  [batch_size, num_notes, num_timesteps, 2])


    
    return logP_out, pa_gen_out



def Loss_Function(Note_State_Batch, logP):
    """
    Arguments:
        Note State Batch: shape = [batch_size x num_notes x num_timesteps x 2]
        batch of log probabilities: shape = [batch_size x num_notes x num_timesteps x 2 x 2]
        
    # This section is the Loss Function Block
    # logP out is the 3x play-articulate log probabilities for each note, at every time step, for every batch
    # Input Note_State_Batch contains the actual class played for each note, at every time step, for every batch
    # The Loss Function should match up the logP log probabilities at time 't-1' to the ground truth class at time 't'
    # Remove the following:
    #    - 1st element of Note_State Batch in 't' dimension.  This is irrelevant as a label, anyways.
    #    - last element of logP_out in 't' dimension.  There is no corresponding future Note_State_Batch element , anyways
    # y_out elements will now correspond to the Note_State_Batch elements that it is trying to predict.  
    """   

    
    # batch_size and num_timesteps are variable length
    batch_size = tf.shape(logP)[0]
    num_notes = logP.get_shape()[1].value
    num_timesteps = tf.shape(logP)[2]
    
    
    #assert Note_State_Batch.get_shape()[0].value == logP.get_shape()[0].value
    #assert Note_State_Batch.get_shape()[1].value == logP.get_shape()[1].value
    #assert Note_State_Batch.get_shape()[2].value == logP.get_shape()[2].value


    # Line up logP with future input data
    logP_align = tf.slice(logP, [0,0,0,0,0],[batch_size, num_notes, num_timesteps-1, 2, 2])
    #print('logP : ', logP)
    #print('logP align: ', logP_align)

    Note_State_Batch_align = tf.cast(tf.slice(Note_State_Batch, [0,0,1, 0],[batch_size, num_notes, num_timesteps-1, 2]), dtype=tf.int64)
    #print('Note_State_Batch: ', Note_State_Batch)
    #print('Note_State_Batch_align: ', Note_State_Batch_align)


    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logP_align,labels=Note_State_Batch_align) 
    Loss = tf.reduce_mean(cross_entropy)
   
    
    return Loss

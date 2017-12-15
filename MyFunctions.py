import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import LSTMStateTuple



def Input_Kernel(input_data, Midi_low, Midi_high, time_init):
    """
    Arguments:
        input_data: size = [batch_size x num_notes x num_timesteps x 2] 
            (the input data represents that at the previous timestep of what we are trying to predict)
        Midi_low: integer
        Midi_high: integer
        time_init: integer representing where the 'beat' component begins for the batch.
    Returns:
        Note_State_Expand: size = [batch_size x num_notes x num_timesteps x 80]
    """    

    
    # Capture input_data dimensions (batch_size and num_timesteps are variable length)
    batch_size = tf.shape(input_data)[0]
    num_notes = input_data.get_shape()[1].value
    num_timesteps = tf.shape(input_data)[2]
    
    
    
    # MIDI note number (only a function of the note index)
    Midi_indices = tf.squeeze(tf.range(start=Midi_low, limit = Midi_high+1, delta=1))
    x_Midi = tf.ones((batch_size, num_timesteps, 1, num_notes))*tf.cast(Midi_indices, dtype=tf.float32)
    x_Midi = tf.transpose(x_Midi, perm=[0,3,1,2]) # shape = batch_size, num_notes, num_timesteps, 1
    #print('x_Midi shape = ', x_Midi.get_shape())

    
    # part_pitchclass (only a function of the note index)
    Midi_pitchclasses = tf.squeeze(x_Midi % 12, axis=3)
    x_pitch_class = tf.one_hot(tf.cast(Midi_pitchclasses, dtype=tf.uint8), depth=12)
    #print('x_pitch_class shape = ', x_pitch_class.get_shape())
    
   
    # part_prev_vicinity
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
   

    #part_prev_context
    input_flatten_p_bool = tf.minimum(input_flatten_p,1) 
    # 1 if note is played, 0 if not played.  Don't care about articulation
    
    #kernel
    filt_context = tf.expand_dims(tf.tile(tf.eye(12), multiples=[(num_notes // 12)*2,1]), axis=1)
    #print('filt_context size = ', filt_context.get_shape())
       
    context = tf.nn.conv1d(input_flatten_p_bool, filt_context, stride=1, padding='SAME')
    x_context = tf.reshape(context, shape=[batch_size, num_timesteps, num_notes, 12])
    x_context = tf.transpose(x_context, perm=[0,2,1,3])
    #print('x_prev context shape = ', x_prev_context.get_shape())    
    
    
    
    #beat (only a function of the time axis index plus the time_init value
    Time_indices = tf.range(time_init, num_timesteps + time_init)
    x_Time = tf.reshape(tf.tile(Time_indices, multiples=[batch_size*num_notes]), shape=[batch_size, num_notes, num_timesteps,1])
    x_beat = tf.cast(tf.concat([x_Time%2, x_Time//2%2, x_Time//4%2, x_Time//8%2], axis=-1), dtype=tf.float32)
    #print('x_beat shape = ', x_beat.get_shape()) 
    
    #zero
    x_zero = tf.zeros([batch_size, num_notes, num_timesteps,1])


    #Final Vector
    Note_State_Expand = tf.concat([x_Midi, x_pitch_class, x_vicinity, x_context, x_beat, x_zero], axis=-1)
    
    
    return Note_State_Expand




def LSTM_TimeWise_Training_Layer(input_data, state_init, output_keep_prob=1.0):
    """
    Arguments:
        input_data: Tensor with size = [batch_size x num_notes x num_timesteps x input_size]
        state_init: List of LSTMTuples([batch_size*num_notes x num_units[layer]], [batch_size*num_notes x num_units[layer]])
        
    Returns:
        output: tensor with size = [batch_size*num_notes x num_timesteps x num_units_final
        state: List of LSTMTuples([batch_size*num_notes x num_units[layer]], [batch_size*num_notes x num_units[layer]])
        
    # LSTM time-wise 
    # This section is the 'Model LSTM-TimeAxis' block and will run a number of LSTM cells over the time axis.
    # Every note and sample in the batch will be run in parallel with the same LSTM weights


    
    # Reshape the input
    # batch_size and num_notes dimensions of input are flattened to treat as single 'batch' dimension for LSTM cell
    # will be reshaped at the end of this block for the next stage
    # state_init is already flat for convenience
  """  
    
    # batch_size and num_timesteps are variable length
    batch_size = tf.shape(input_data)[0]
    num_notes = input_data.get_shape()[1].value
    num_timesteps = tf.shape(input_data)[2]
    input_size = input_data.get_shape()[3].value
    
    num_layers = len(state_init)    
    
    # Flatten input
    input_flatten = tf.reshape(input_data, shape=[batch_size*num_notes, num_timesteps, input_size])
    
    # generate cell list of length specified by initial state
    cell_list=[]
    num_states=[]
    for h in range(num_layers):
        num_states.append(state_init[h][0].get_shape()[1].value)
        lstm_cell = BasicLSTMCell(num_units=num_states[h], forget_bias=1.0, state_is_tuple=True,activation=math_ops.tanh, reuse=None)
        lstm_cell = DropoutWrapper(lstm_cell, output_keep_prob=output_keep_prob)
        cell_list.append(lstm_cell)
    

    #Instantiate multi layer Time-Wise Cell
    multi_lstm_cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)


    #Run through LSTM time steps and generate time-wise sequence of outputs
    output_flat, state_out = tf.nn.dynamic_rnn(cell=multi_lstm_cell, inputs=input_flatten, initial_state=state_init, dtype=tf.float32)

    output = tf.reshape(output_flat, shape=[batch_size, num_notes, num_timesteps, num_states[-1]])
    
    return output, state_out


def LSTM_NoteWise_Layer(input_data, state_init, output_keep_prob=1.0, num_class=2):
    """
    Arguments:
        input_data: size = [batch_size x num_notes x num_timesteps x size_input]
        state_init: List of LSTMTuples([batch_size*num_time_steps x num_units[layer]], [batch_size*num_timesteps x num_units[layer]]) 
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
    input_size = input_data.get_shape()[3].value
    
    num_layers = len(state_init) 

    # Reshape the input
    # batch_size and num_timesteps dimensions of input are flattened to treat as single 'batch' dimension for LSTM cell
    notewise_in = tf.transpose(input_data, perm=[0,2,1,3])
    notewise_in = tf.reshape(notewise_in, shape=[batch_size*num_timesteps, num_notes, input_size])
    
    # generate LSTM cell list of length specified by initial state
    cell_list=[]
    num_states=[]
    for h in range(num_layers):
        num_states.append(state_init[h][0].get_shape()[1].value)
        lstm_cell = BasicLSTMCell(num_units=num_states[h], forget_bias=1.0, state_is_tuple=True,activation=math_ops.tanh, reuse=None)
        lstm_cell = DropoutWrapper(lstm_cell, output_keep_prob=output_keep_prob)
        cell_list.append(lstm_cell)

    #Instantiate multi layer Time-Wise Cell
    multi_lstm_cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True) 
    
   
    # For this LSTM cell, can't use tf.nn.dynamic_rnn because samples have to be generated and fed back to for subsequent notes
    # need to feed the generated output for note 'n-1' into the generation of note 'n'
    # Will use 'for' loop and call the LSTM cell each time

   

    #Set values for initial LSTM state and sampled note.  Zero notes always played below bottom note
    h_state = state_init
    note_gen_n = tf.zeros([batch_size*num_timesteps, 2])

    y_list=[]
    note_gen_list=[]
  
   
    #Run through notes for note-wise LSTM to obtain P(va(n) | va(<n))
    for n in range(num_notes):    
        #concatenate previously sampled note play-articulate-combo with timewise output
        p_gen = tf.cast(tf.slice(note_gen_n, [0,0],[-1,1]), tf.float32)
        #a_gen = tf.cast(tf.slice(note_gen_n, [0,1],[-1,1]), tf.float32) # only feed back 'play' component, NOT 'articulate'
        cell_inputs = tf.concat([notewise_in[:,n,:], p_gen], axis=-1)

        #print('Cell inputs shape = ', cell_inputs.get_shape())
        
        # output shape = [batch_size*num_timesteps, Nfinal] 
        h_final_out, h_state = multi_lstm_cell(inputs=cell_inputs, state=h_state)       
        #print('h_final_out shape = ', h_final_out.get_shape())
        #print('h_state len = ', len(h_state))
        
        # Fully Connected Layer to generate 4 outputs: [logP(p=0), logP(p=1), logP(a=0), logP(a=1)]
        y_n = tf.layers.dense(inputs=h_final_out, units=4, activation=None)
        #print('y_n shape original shape = ', y_n.get_shape())
        
        # slice output of full connected layer into play and articulate probabilities   
        y_p_n = tf.slice(y_n, [0,0], [-1,2]) #[logP(p=0), logP(p=1)]
        y_a_n = tf.slice(y_n, [0,2], [-1,2]) #[logP(a=0), logP(a=1)]
        #print('y_p_n shape = ', y_p_n.get_shape())
        #print('y_a_n shape = ', y_a_n.get_shape())
       
        # Sample the 'play' and 'articulate' values  from y_p_n and y_a_n (unscaled) log probabilities, respectively
        p_gen_n = tf.multinomial(logits=y_p_n, num_samples=1) 
        a_gen_n = tf.multinomial(logits=y_a_n, num_samples=1) 
      
        """
         Network should never generate a 'no play' with an 'articulate'.  
         This unnecessarily penalizes notes that are (correctly) rested but perceived by the network to require 'articulation'
         In addition, the Midi-to-Matrix function never generates this condition, so during music generation, feeding this condition into              the next batch creates inputs that the model has never seen.
        """        
        
        a_mask_gen_n = tf.multiply(p_gen_n, a_gen_n) # if a given note is not played, automatically set articulate to zero.
        note_gen_n = tf.concat([p_gen_n, a_mask_gen_n], axis=1) # concatenate
        
        #print('note_gen_n final shape = ', note_gen_n.get_shape())
        
        # Concatenate y_n back together such that the logp=1/0 is the last dimension
        y_n = tf.stack([y_p_n, y_a_n], axis=2) # now y_n point is 2 x 2 with last dimension being [play, articulate]
        y_n = tf.transpose(y_n, perm=[0, 2, 1])# now last dimension is [prob = 1, prob = 0]
        #dimensions of y_n are: [batch_size*num_timesteps, 2, 2]
        
        # Reshape the 1st dimension back into batch and timesteps dimensions                        
        y_n_unflat = tf.reshape(y_n, shape=[batch_size, num_timesteps, 2, 2])
        note_gen_n_unflat = tf.reshape(note_gen_n, shape=[batch_size, num_timesteps, 2])
        
        #print('note_gen_n shape = ', note_gen_n.get_shape())
        
        #Append to notewise list
        y_list.append(y_n_unflat)
        note_gen_list.append(note_gen_n_unflat)
    
    # Convert output list to a Tensor
    #y_out = tf.reshape(tf.stack(y_list, axis=1), [batch_size, num_notes, num_timesteps, 2, 2])
    #note_gen_out = tf.reshape(tf.stack(note_gen_list, axis=1),  [batch_size, num_notes, num_timesteps, 2])
    y_out = tf.stack(y_list, axis=1)
    note_gen_out = tf.stack(note_gen_list, axis=1)



    
    return y_out, note_gen_out



def Loss_Function(Note_State_Batch, y_in):
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
    batch_size = tf.shape(y_in)[0]
    num_notes = y_in.get_shape()[1].value
    num_timesteps = tf.shape(y_in)[2]
    
    
    #assert Note_State_Batch.get_shape()[0].value == logP.get_shape()[0].value
    #assert Note_State_Batch.get_shape()[1].value == logP.get_shape()[1].value
    #assert Note_State_Batch.get_shape()[2].value == logP.get_shape()[2].value


    # Line up logP with future input data
    y_align = tf.slice(y_in, [0,0,0,0,0],[batch_size, num_notes, num_timesteps-1, 2, 2])
    #print('logP : ', logP)
    print('y_align shape = : ', y_align.get_shape())

    Note_State_Batch_align = tf.cast(tf.slice(Note_State_Batch, [0,0,1, 0],[batch_size, num_notes, num_timesteps-1, 2]), dtype=tf.int64)
    #print('Note_State_Batch: ', Note_State_Batch)
    print('Note_State_Batch_align shape = : ', Note_State_Batch_align.get_shape())


    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_align,labels=Note_State_Batch_align)
    Loss = tf.reduce_mean(cross_entropy)
   
    
    return Loss, cross_entropy

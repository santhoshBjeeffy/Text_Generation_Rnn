# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 19:22:10 2018

@author: santhob
"""
import temp
tf.train.latest_checkpoint(checkpoint_dir)

model=build_model(vocab_size,embedding_dim,rnn_units,batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1,None]))

model.summary()


# the prediction loop

def generate_text(model,start_string):
    #evaluation step (generating text using learned model)
    num_generate=1000
    #you can change the start string
    start_string='ROMEO'
    #vectorizing(converting string to nos)
    input_eval=[char2idx[s] for s in start_string]
    input_eval=tf.expand_dims(input_eval,0)
    
    #empty string to store our results
    text_generated=[]
    
    # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
    temperature=1.0
    
    #here batch size ==1
    model.reset_states()
    for i in range(num_generate):
        predictions=model(input_eval)
        #remove the batch dimension
        predictions=tf.squeeze(predictions,0)
        predictions=predictions/temperature
        predicted_id=tf.multinomial(predictions,num_samples=1)[-1,0].numpy()
        
        input_eval=tf.expand_dims([predicted_id,0])
        text_generated.append(idx2char[predicted_id])
    
    return (start_string + ''.join(text_generated))

    # using a multinomial distribution to predict the word returned by the model
    

    print(generate_text(model, start_string="ROMEO: "))
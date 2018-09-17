import tensorflow as tf
import sys
sys.path.insert(0,'./pylibs')
import tfutils

def get_error_map(x,gt):
	weights= tf.cast(tf.greater(gt,0),tf.float32)
	est = tf.multiply(x,weights)
	res = tf.cast( tf.abs(tf.subtract(est,gt)) ,tf.float32)
	error_map = tf.cast( tf.greater(res,3),tf.float32)
	return error_map

def gt_compare(x,gt):
	weights= tf.cast(tf.greater(gt,0),tf.float32)
	valid = tf.cast(tf.count_nonzero(gt),tf.float32)
	est = tf.multiply(x,weights) 
	res = tf.cast( tf.abs(tf.subtract(est,gt)) ,tf.float32)
	error = tf.reduce_sum(tf.cast( tf.greater(res,3),tf.float32))
	return tf.divide(error,valid)

def res_block(inputs,kernel,features,scope,dropout,bn,reuse,is_training):
	conv1 = tfutils.conv2d(inputs,
       features,
       kernel,
       scope+'_block_conv1',
       stride=[1, 1],
       padding='SAME',
       use_xavier=True,
       stddev=1e-3,
       weight_decay=0.0,
       activation_fn=tf.nn.relu,
       bn=bn,
       bn_decay=None,
       is_training=is_training,
       dropout=dropout,
       reuse=reuse)

	conv2 = tfutils.conv2d(conv1,
       features,
       kernel,
       scope+'_block_conv2',
       stride=[1, 1],
       padding='SAME',
       use_xavier=True,
       stddev=1e-3,
       weight_decay=0.0,
       activation_fn=tf.nn.relu,
       bn=bn,
       bn_decay=None,
       is_training=is_training,
       dropout=None,
       reuse=reuse)

	conv2 = tf.concat([inputs,conv2],3)

	conv3 = tfutils.conv2d(conv2,
       features,
       kernel,
       scope+'_block_conv3',
       stride=[1, 1],
       padding='SAME',
       use_xavier=True,
       stddev=1e-3,
       weight_decay=0.0,
       activation_fn=tf.nn.relu,
       bn=bn,
       bn_decay=None,
       is_training=is_training,
       dropout=dropout,
       reuse=reuse)

	conv4 = tfutils.conv2d(conv3,
       features,
       kernel,
       scope+'_block_conv4',
       stride=[1, 1],
       padding='SAME',
       use_xavier=True,
       stddev=1e-3,
       weight_decay=0.0,
       activation_fn=tf.nn.relu,
       bn=bn,
       bn_decay=None,
       is_training=is_training,
       dropout=None,
       reuse=reuse)   

	conv4 =  tf.concat([conv2,conv4],3)      	    

	return conv4


def h_net(inputs,height,width,is_training,reuse,keep_prob):

	bn=False

	conv1 = tfutils.conv2d_depth(inputs,
           32,
           [5,5],
           'First_Block_conv1',
           stride=[1, 1],
           padding='SAME',
           rate=[1],
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=is_training,
           dropout=None,
           reuse=reuse)

	skip1 = conv1

	conv1 = tfutils.conv2d(conv1,
       32,
       [5,5],
       'First_Block_conv1_down',
       stride=[2, 2],
       padding='SAME',
       use_xavier=True,
       stddev=1e-3,
       weight_decay=0.0,
       activation_fn=tf.nn.relu,
       bn=False,
       bn_decay=None,
       is_training=is_training,
       dropout=keep_prob,
       reuse=reuse)			


	conv1_block = res_block(conv1,[3,3],32,'First_Block_block1',1,bn,reuse,is_training)
	skip2 = conv1_block



	###############################################################################
	######################## Transpose Convolution 1_step ########################


	conv1_up_block = tfutils.conv2d_transpose(conv1_block,
	                     32,
	                     [5,5],
	                     'First_Block_inverted_conv1_up',
	                     stride=[2, 2],
	                     padding='SAME',
	                     use_xavier=True,
	                     stddev=1e-3,
	                     weight_decay=0.0,
	                     activation_fn=tf.nn.relu,
	                     bn=False,
	                     bn_decay=None,
	                     is_training=is_training,
	                     dropout=1,
	                     height=height//2,
	                     width=width//2,
	                     reuse=reuse)

	conv1_up_block = tf.concat([conv1_up_block,skip1],3)  
	conv1_up_block = res_block(conv1_up_block,[3,3],32,'First_Block_inverted_block',1,bn,reuse,is_training)


	down2_skip = tfutils.conv2d(conv1_up_block,
           1,
           [1,1],
           'First_Block_inverted_out_output',
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=None,
           bn=False,
           bn_decay=None,
           is_training=is_training,
           dropout=1,
           reuse=reuse)  	



	###############################################################################



	conv2_block = tfutils.conv2d(conv1_block,
       64,
       [5,5],
       'Second_Block_out_conv2',
       stride=[2, 2],
       padding='SAME',
       use_xavier=True,
       stddev=1e-3,
       weight_decay=0.0,
       activation_fn=tf.nn.relu,
       bn=False,
       bn_decay=None,
       is_training=is_training,
       dropout=keep_prob,
       reuse=reuse)	

	skip3 = conv2_block
	conv2_block = res_block(conv2_block,[3,3],64,"Second_Block_block2",1,bn,reuse,is_training)
	

	###############################################################################
	######################## Transpose Convolution 2_steps ########################

	conv2_up_block_1 = tfutils.conv2d_transpose(conv2_block,
	                     32,
	                     [5,5],
	                     'Second_Block_inverted_conv1_up',
	                     stride=[2, 2],
	                     padding='SAME',
	                     use_xavier=True,
	                     stddev=1e-3,
	                     weight_decay=0.0,
	                     activation_fn=tf.nn.relu,
	                     bn=False,
	                     bn_decay=None,
	                     is_training=is_training,
	                     dropout=1,
	                     height=height//4,
	                     width=width//4,
	                     reuse=reuse)

	conv2_up_block_1 = tf.concat([conv2_up_block_1,skip2],3)  
	conv2_up_block_1 = res_block(conv2_up_block_1,[3,3],32,'Second_Block_inverted_block1',1,bn,reuse,is_training)


	conv2_up_block_2 = tfutils.conv2d_transpose(conv2_up_block_1,
	                     32,
	                     [5,5],
	                     'Second_Block_inverted_conv2_up',
	                     stride=[2, 2],
	                     padding='SAME',
	                     use_xavier=True,
	                     stddev=1e-3,
	                     weight_decay=0.0,
	                     activation_fn=tf.nn.relu,
	                     bn=False,
	                     bn_decay=None,
	                     is_training=is_training,
	                     dropout=1,
	                     height=height//2,
	                     width=width//2,
	                     reuse=reuse)

	conv2_up_block_2 = tf.concat([conv2_up_block_2,skip1],3)  
	conv2_up_block_2 = res_block(conv2_up_block_2,[3,3],32,'Second_Block_inverted_block2',1,bn,reuse,is_training)		



	down4_skip = tfutils.conv2d(conv2_up_block_2,
           1,
           [1,1],
           'Second_Block_inverted_out_output',
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=None,
           bn=False,
           bn_decay=None,
           is_training=is_training,
           dropout=1,
           reuse=reuse)  




	###############################################################################



	conv3_block = tfutils.conv2d(conv2_block,
       128,
       [5,5],
       'Third_Block_out_conv3',
       stride=[2, 2],
       padding='SAME',
       use_xavier=True,
       stddev=1e-3,
       weight_decay=0.0,
       activation_fn=tf.nn.relu,
       bn=False,
       bn_decay=None,
       is_training=is_training,
       dropout=keep_prob,
       reuse=reuse)	

	conv3_block = res_block(conv3_block,[3,3],128,"Third_Block_block2",1,bn,reuse,is_training)
	

	conv3_up_block_1 = tfutils.conv2d_transpose(conv3_block,
	                     128,
	                     [5,5],
	                     'Third_Block_inverted_conv1_up',
	                     stride=[2, 2],
	                     padding='SAME',
	                     use_xavier=True,
	                     stddev=1e-3,
	                     weight_decay=0.0,
	                     activation_fn=tf.nn.relu,
	                     bn=False,
	                     bn_decay=None,
	                     is_training=is_training,
	                     dropout=1,
	                     height=height//8,
	                     width=width//8,
	                     reuse=reuse)

	conv3_up_block_1 = tf.concat([conv3_up_block_1,skip3],3)  
	conv3_up_block_1 = res_block(conv3_up_block_1,[3,3],64,'Third_Block_inverted_block1',1,bn,reuse,is_training)


	conv3_up_block_2 = tfutils.conv2d_transpose(conv3_up_block_1,
	                     64,
	                     [5,5],
	                     'Third_Block_inverted_conv2_up',
	                     stride=[2, 2],
	                     padding='SAME',
	                     use_xavier=True,
	                     stddev=1e-3,
	                     weight_decay=0.0,
	                     activation_fn=tf.nn.relu,
	                     bn=False,
	                     bn_decay=None,
	                     is_training=is_training,
	                     dropout=1,
	                     height=height//4,
	                     width=width//4,
	                     reuse=reuse)

	conv3_up_block_2 = tf.concat([conv3_up_block_2,skip2],3)  
	conv3_up_block_2 = res_block(conv3_up_block_2,[3,3],64,'Third_Block_inverted_block2',1,bn,reuse,is_training)	


	conv3_up_block_3 = tfutils.conv2d_transpose(conv3_up_block_2,
	                     32,
	                     [5,5],
	                     'Third_Block_inverted_conv3_up',
	                     stride=[2, 2],
	                     padding='SAME',
	                     use_xavier=True,
	                     stddev=1e-3,
	                     weight_decay=0.0,
	                     activation_fn=tf.nn.relu,
	                     bn=False,
	                     bn_decay=None,
	                     is_training=is_training,
	                     dropout=1,
	                     height=height//2,
	                     width=width//2,
	                     reuse=reuse)

	conv3_up_block_3 = tf.concat([conv3_up_block_3,skip1],3)  
	conv3_up_block_3 = res_block(conv3_up_block_3,[3,3],64,'Third_Block_inverted_block3',1,bn,reuse,is_training)				



	down8_skip = tfutils.conv2d(conv2_up_block_2,
           1,
           [1,1],
           'Third_Block_inverted_output',
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=None,
           bn=False,
           bn_decay=None,
           is_training=is_training,
           dropout=1,
           reuse=reuse)  

	all_preds = tf.concat( [ inputs[:,:,:,0:1],down2_skip,down4_skip,down8_skip ],3 )

	output = tf.add( inputs[:,:,:,0:1], down8_skip )
	output = tf.add( output ,down4_skip)
	output = tf.add( output ,down2_skip)


	output = tf.concat( [ output, skip1],3 )
	output = res_block(output,[3,3],64,'Combined_Res_Block',1,bn,reuse,is_training)

	output = tfutils.conv2d(output,
       1,
       [1,1],
       'output_layer',
       stride=[1, 1],
       padding='SAME',
       use_xavier=True,
       stddev=1e-3,
       weight_decay=0.0,
       activation_fn=tf.nn.relu,
       bn=False,
       bn_decay=None,
       is_training=is_training,
       dropout=1,
  	   reuse=reuse)  	




	return tf.concat([all_preds,output],3)
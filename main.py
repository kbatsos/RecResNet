from __future__ import division

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import scipy
from sklearn.feature_extraction import image

from collections import namedtuple

import sys
import math
import random
import os
import time

import argparse
import json

sys.path.insert(0,'./pylibs')
sys.path.insert(0,'.')
sys.path.insert(0,'./cpp')

import cpputils 
import pfmutil as pfm
import tfutils
import tfmodel
from kitti_dataloader import Dataloader as ktdt
from kitti2015_dataloader import Dataloader as kt2015dt
from freiburg_dataloader import Dataloader as frdt


#Tensorboard log file locations. Used for visualization
train_logs="/media/kbatsos/Data1/git/pretrain"
validation_logs="/media/kbatsos/Data1/git/pretrain_val"
validation_accuracy="/media/kbatsos/Data1/git/train_acc_val"

#Only used if no model is specified.
model_save="/media/kbatsos/Data1/model/model.ckpt"

parser = argparse.ArgumentParser(description='RecResNet TensorFlow Implementation.')
parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model',                     type=str,   help='model to load', default=None)
parser.add_argument('--params', 				   type=str,   help='loader parameters', default=None)

args = parser.parse_args()

with open(args.params) as f:
	loader_data=json.load(f)

parameters = namedtuple('parameters','left_path, disp_path,gt_path,gt_path_noc,down_sample_ratio,epochs')
fr_parameters = namedtuple('parameters','left_path,kitti_disp_path,kitti15_disp_path,gt_path,gt_path_noc,down_sample_ratio,epochs')

if loader_data["loader"] == "freiburg":
	params=fr_parameters(left_path=loader_data["left_path"],
				  kitti_disp_path=loader_data["kitti_disp_path"],
				  kitti15_disp_path=loader_data["kitti15_disp_path"],
				  gt_path=loader_data["gt_path"],
				  gt_path_noc=loader_data["gt_path_noc"],
				  down_sample_ratio=8,
				  epochs=400)
elif loader_data["loader"] == "pretrain":
	params=fr_parameters(left_path=loader_data["data"]["freiburg"]["left_path"],
				  kitti_disp_path=loader_data["data"]["freiburg"]["kitti_disp_path"],
				  kitti15_disp_path=loader_data["data"]["freiburg"]["kitti15_disp_path"],
				  gt_path=loader_data["data"]["freiburg"]["gt_path"],
				  gt_path_noc=loader_data["data"]["freiburg"]["gt_path_noc"],
				  down_sample_ratio=8,
				  epochs=400)
	kitti_params=parameters(left_path=loader_data["data"]["kitti"]["left_path"],
					  disp_path=loader_data["data"]["kitti"]["disp_path"],
					  gt_path=loader_data["data"]["kitti"]["gt_path"],
					  gt_path_noc=loader_data["data"]["kitti"]["gt_path_noc"],
					  down_sample_ratio=loader_data["data"]["kitti"]["down_sample_ratio"],
					  epochs=loader_data["data"]["kitti"]["epochs"])
	kitti15_params=parameters(left_path=loader_data["data"]["kitti15"]["left_path"],
					  disp_path=loader_data["data"]["kitti15"]["disp_path"],
					  gt_path=loader_data["data"]["kitti15"]["gt_path"],
					  gt_path_noc=loader_data["data"]["kitti15"]["gt_path_noc"],
					  down_sample_ratio=loader_data["data"]["kitti15"]["down_sample_ratio"],
					  epochs=loader_data["data"]["kitti15"]["epochs"])
else:
	params=parameters(left_path=loader_data["left_path"],
					  disp_path=loader_data["disp_path"],
					  gt_path=loader_data["gt_path"],
					  gt_path_noc=loader_data["gt_path_noc"],
					  down_sample_ratio=loader_data["down_sample_ratio"],
					  epochs=loader_data["epochs"])





with tf.Graph().as_default():

	global_step = tf.Variable(0, trainable=False,name="g_step")
	keep_prob = tf.placeholder(tf.float32,name="keep_prob")

	starter_learning_rate = 0.0001
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
	                                           2500, 1, staircase=True,name="learning_rate")



	if args.mode == 'train':
		if loader_data["loader"] == "kitti":
			dataloader = ktdt(params)
			kitti_dataloader = ktdt(params)
		elif loader_data["loader"] == "kitti15":
			dataloader = kt2015dt(params)  
			kitti15_dataloader = kt2015dt(params)
			
		else:
			dataloader = frdt(params)
			kitti_dataloader = ktdt(kitti_params)
			kitti15_dataloader = kt2015dt(kitti15_params)

		height,width,channels = dataloader.get_training_data_size();

		with tf.Session() as sess:
			
			x=tf.placeholder(tf.float32, [1,None,None,channels], name="x_p")
			y=tf.placeholder(tf.float32, [1,None,None,1], name="y_p")
			y_noc=tf.placeholder(tf.float32, [1,None,None,1], name="y_noc_p")
			keep_prob = tf.placeholder(tf.float32,name="keep_prob")

			input_height = tf.Variable(0, name="input_height",dtype=tf.int32)	
			input_width = tf.Variable(0, name="input_width",dtype=tf.int32)
			is_training = tf.Variable(True, name="is_training",dtype=tf.bool)
			max_disp = tf.Variable(0,name="max_disp",dtype=tf.int32)

			kt_val_err = tf.Variable(0.0, name="kt_validation_error",collections=[tf.GraphKeys.LOCAL_VARIABLES])
			kt15_val_err = tf.Variable(0.0, name="kt15_validation_error",collections=[tf.GraphKeys.LOCAL_VARIABLES])	
			train_err = tf.Variable(0.0, name="validation_error")	

			#Do not consider pixels without ground truth 
			weights = tf.cast(tf.greater(y,0),tf.float32)
			weights_noc = tf.cast(tf.greater(y_noc,0),tf.float32)
			weights = tf.add(weights,weights_noc)

			#perform 2 iterations over the network
			pred = tf.to_float(tfmodel.h_net(x,input_height,input_width,is_training,reuse=False,keep_prob=keep_prob))
			out_1=pred[:,:,:,4:5];
			loss_it1 = tf.losses.absolute_difference( y,out_1,weights=weights,scope='it1_loss_5' )			
			pred = tf.concat([pred[:,:,:,4:5],x[:,:,:,1:2]],3)
			pred = tf.to_float(tfmodel.h_net(pred,input_height,input_width,is_training,reuse=True,keep_prob=keep_prob))
			out_2=pred[:,:,:,4:5];
			loss_it2 = tf.losses.absolute_difference( y,out_2,weights=weights,scope='it2_loss_5' )

			#aggregate losses by iteration (helps the network converge faster)
			total_loss = .75*loss_it2 + .25*loss_it1

			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			total_opt = optimizer.minimize(total_loss,global_step=global_step,name="adam_opt_loss")

			error_map = tfmodel.get_error_map(x[:,:,:,0:1],y)
			error_map_it1 = tfmodel.get_error_map(out_1,y)
			error_map_it2 = tfmodel.get_error_map(out_2,y)

			error_it1 = tfmodel.gt_compare(out_1,y)
			error_it2 = tfmodel.gt_compare(out_2,y)
			init_error = tfmodel.gt_compare(x[:,:,:,0:1],y)
			#Tensorboard stats and visualizations
			t_ka=tf.summary.scalar('Training_error',error_it2)
			t_lr=tf.summary.scalar('learning_rate',learning_rate)
			t_tl=tf.summary.scalar('total_loss',total_loss)
			
			t_i=tf.summary.image("input",x[:,:,:,0:1],max_outputs=1)
			t_em=tf.summary.image("error_map",error_map,max_outputs=1)
			

			t_emp_it1=tf.summary.image("it1/error_map",error_map_it1,max_outputs=1)
			t_out_it1=tf.summary.image("it1/output",out_1,max_outputs=1)


			t_emp_it2=tf.summary.image("it2/error_map",error_map_it2,max_outputs=1)
			t_im5_it2=tf.summary.image("it2/output",out_2,max_outputs=1)
			t_im4_it2=tf.summary.image("it2/R4",pred[:,:,:,3:4],max_outputs=1)
			t_im3_it2=tf.summary.image("it2/R3",pred[:,:,:,2:3],max_outputs=1)
			t_im2_it2=tf.summary.image("it2/R2",pred[:,:,:,1:2],max_outputs=1)

			merged = tf.summary.merge([ t_ka, t_lr, t_tl,
										t_i, t_em, 
										t_emp_it1,
										t_out_it1,
										t_emp_it2,  t_im5_it2, t_im4_it2, t_im3_it2,t_im2_it2 ])
		
			kt_val_err_sum = tf.summary.scalar("kt_validation_error",kt_val_err)	
			kt_val_merge = tf.summary.merge([kt_val_err_sum])

			kt15_val_err_sum = tf.summary.scalar("kt15_validation_error",kt15_val_err)	
			kt15_val_merge = tf.summary.merge([kt15_val_err_sum])

			train_err_sum = tf.summary.scalar("training_error",train_err)	
			train_merge = tf.summary.merge([train_err_sum])
			
			train_writer = tf.summary.FileWriter(train_logs,
	                      sess.graph)	

			val_writer = tf.summary.FileWriter(validation_logs,
                  	sess.graph)

			train_acc_writer = tf.summary.FileWriter(validation_accuracy,
                  	sess.graph)	                      	
			
			init = tf.global_variables_initializer()
			# Saver will save all variables existing at the time of construction
			saver = tf.train.Saver(max_to_keep=10)
			if args.model != None:
				saver.restore(sess, args.model)
				print("Model restored")
			else:
				sess.run(init)		

			
			accumulate_training_error = np.empty([0])	
			current_epoch = dataloader.epoch
			while( dataloader.epoch <  params.epochs):
				is_train_phase = True
				if current_epoch < dataloader.epoch:
					training_error = np.mean(accumulate_training_error)
					t_e = sess.run( train_merge, {train_err:training_error} )
					print "Training error: " + str(training_error)
					train_acc_writer.add_summary(t_e,1)
					train_acc_writer.flush()
					accumulate_training_error = np.empty([0])
					current_epoch = dataloader.epoch
	

				data,gt,gt_noc,sindex = dataloader.load_training_sample(); 
				outp,opt,lossv,summary,err_it1,err_it2,ini_err = sess.run([pred,total_opt,total_loss,merged,error_it1,error_it2,init_error], feed_dict={x: data,y: gt, y_noc:gt_noc,
																																						 input_width:width,input_height:height,is_training:True,
																																						 max_disp:dataloader.max_disp,keep_prob:.8})
				train_writer.add_summary(summary, dataloader.epoch*params.epochs +sindex )
				accumulate_training_error = np.append( accumulate_training_error,[err_it2 ] )
				print("Epoch: "+ str(dataloader.epoch) + " Step: " + str(sindex) + " RMS error: " +str(lossv) + " Init Error " + str(ini_err) + " Error it1: " + str(err_it1) + " Error it2: "+ str(err_it2) )

				if(sindex == 1  and dataloader.epoch%2 == 0  and dataloader.epoch >0 ):#and dataloader.epoch> 0 

					print "########################### Running Validation ###########################################"

					def validate( validation_dataloader, v_height,v_width ):
						accumulate_pred = np.empty([0])
						accumulate_pred_1 = np.empty([0])
						accumulate_init = np.empty([0])
						validation_dataloader.init_sample_index(validation_dataloader.get_sample_size()-40)

						while( validation_dataloader.get_sample_index() < validation_dataloader.get_sample_size()):

							data,gt,gt_noc,sindex = validation_dataloader.load_validation_sample();
							outp,err_it1,err,ini_err,disp,errmp,errmi = sess.run([pred,error_it1,error_it2,init_error,x,error_map_it2,error_map], feed_dict={x: data,y: gt, y_noc:gt_noc, input_width:v_width,input_height:v_height,is_training:False,
																																					max_disp:validation_dataloader.max_disp,keep_prob:1})

							print("Sample: "+ str(validation_dataloader.get_sample_index()) + " Step: " + str(sindex) + " Init Error " + str(ini_err) + "Error it1: " + str(err_it1) + " Error: " + str(err) )
							accumulate_pred = np.append( accumulate_pred,[err ] )
							accumulate_pred_1 = np.append( accumulate_pred_1,[err_it1 ] )
							accumulate_init = np.append( accumulate_init,[ini_err])


							

						mean_error_1 = np.mean(accumulate_pred_1)	
						mean_error = np.mean(accumulate_pred)
						mean_init_error = np.mean(accumulate_init)
						print(' Validation Initial Error: '+ str(mean_init_error) + 'Validation Mean error 1: ' + str(mean_error_1) + 'Validation Mean error 2: ' + str(mean_error) )
						return mean_error


					if loader_data["loader"] == "kitti" or loader_data["loader"] == "pretrain":
						kt_height,kt_width,kt_channels = kitti_dataloader.get_data_size()
						mean_error=validate(kitti_dataloader, kt_height,kt_width )	
						v_e = sess.run( kt_val_merge, {kt_val_err:mean_error} )
						val_writer.add_summary(v_e,1)
						val_writer.flush()

					if loader_data["loader"] == "kitti15" or loader_data["loader"] == "pretrain":
						kt15_height,kt15_width,kt15_channels = kitti15_dataloader.get_data_size()	
						mean_error=validate(kitti15_dataloader, kt15_height,kt15_width )
						v_e = sess.run( kt15_val_merge, {kt15_val_err:mean_error} )
						val_writer.add_summary(v_e,1)
						val_writer.flush()					

					# if (dataloader.epoch%5 == 0) :
					save_path = saver.save(sess, model_save,global_step=global_step)
					print("Model saved in file: %s" % save_path)	
	


	elif args.mode == 'verify':
		with tf.Session() as sess:


			if loader_data["loader"] == "kitti":
				dataloader = ktdt(params)
			elif loader_data["loader"] == "kitti15":
				dataloader = kt2015dt(params)  
			else:
				dataloader = frdt(params)

			height,width,channels = dataloader.get_data_size();	

			x=tf.placeholder(tf.float32, [1,None,None,channels], name="x_p")
			y=tf.placeholder(tf.float32, [1,None,None,1], name="y_p")
			y_noc=tf.placeholder(tf.float32, [1,None,None,1], name="y_noc_p")

			input_height = tf.Variable(0, name="input_height",dtype=tf.int32)	
			input_width = tf.Variable(0, name="input_width",dtype=tf.int32)
			is_training = tf.Variable(False, name="is_training",dtype=tf.bool)
			keep_prob = tf.placeholder(tf.float32,name="keep_prob")
			error_map = tfmodel.get_error_map(x[:,:,:,0:1],y)
			init_error = tfmodel.gt_compare(x[:,:,:,0:1],y)

			pred = tf.to_float(tfmodel.h_net(x,input_height,input_width,is_training,reuse=False,keep_prob=keep_prob))
			out1 = pred[:,:,:,4:5]
			r1_it1 = pred[:,:,:,1:2]
			r2_it1 = pred[:,:,:,2:3]
			r3_it1 = pred[:,:,:,3:4]

			error_map_it1 = tfmodel.get_error_map(pred[:,:,:,4:5],y)
			error_it1 = tfmodel.gt_compare(pred[:,:,:,4:5],y)			
		
			pred = tf.concat([pred[:,:,:,4:5],x[:,:,:,1:2]],3)
			pred = tf.to_float(tfmodel.h_net(pred,input_height,input_width,is_training,reuse=True,keep_prob=keep_prob))	

			r1_it2 = pred[:,:,:,1:2]
			r2_it2 = pred[:,:,:,2:3]
			r3_it2 = pred[:,:,:,3:4]			

			error_map_it2 = tfmodel.get_error_map(pred[:,:,:,4:5],y)
			error_it2 = tfmodel.gt_compare(pred[:,:,:,4:5],y)				

			saver = tf.train.Saver()
			saver.restore(sess, args.model)
			print("Model restored")	

			accumulate_it1 = np.empty([0])
			accumulate_it2 = np.empty([0])
			accumulate_init = np.empty([0])
			
			while( dataloader.get_sample_index() <  dataloader.get_sample_size() ):				
				data,gt,gt_noc,sindex,o_height,o_width,name = dataloader.load_verify_sample();
				n_h = data.shape[1] - o_height;
				n_w = data.shape[2] - o_width;

				disp_p = np.copy(data[0,n_h:data.shape[1],n_w:data.shape[2],0 ])
				cpputils.write2png(disp_p.astype(np.float32),str("./validation/"+name))
				outp,outp1,ini_err,err_it1,err_it2,init_error_map,err_map_it1,err_map_it2, r1_i1,r2_i1,r3_i1, r1_i2,r2_i2,r3_i2 = sess.run([pred,out1,init_error,error_it1,error_it2,error_map,error_map_it1,error_map_it2,r1_it1,r2_it1,r3_it1,
																									r1_it2,r2_it2,r3_it2	], feed_dict={x: data,y:gt,y_noc:gt_noc,input_width:width,input_height:height,is_training:False,keep_prob:1})

				accumulate_it1 = np.append( accumulate_it1,[err_it1 ] )
				accumulate_it2 = np.append( accumulate_it2,[err_it2 ] )
				accumulate_init = np.append( accumulate_init,[ini_err])
				print "Sample index: " + str(dataloader.get_sample_index()) + " Init error: " + str(ini_err) + " it1 error: " + str(err_it1)+ " it2 error: " + str(err_it2)

				disp_init = np.copy(data[0, n_h:data.shape[1],n_w:data.shape[2],0 ])
				disp_p1 = np.copy(outp1[0, n_h:outp1.shape[1],n_w:outp1.shape[2],0 ])
				disp_p = np.copy(outp[0, n_h:outp.shape[1],n_w:outp.shape[2],4 ])


				ier = np.copy(init_error_map[0, n_h:init_error_map.shape[1],n_w:init_error_map.shape[2],0 ])
				it_er1 =  np.copy(err_map_it1[0, n_h:err_map_it1.shape[1],n_w:err_map_it1.shape[2],0 ])	
				it_er2 =  np.copy(err_map_it2[0, n_h:err_map_it2.shape[1],n_w:err_map_it2.shape[2],0 ])

				rit1= r1_i1[0,:,:,0]+r2_i1[0,:,:,0]+r3_i1[0,:,:,0]
				rit2= r1_i2[0,:,:,0]+r2_i2[0,:,:,0]+r3_i2[0,:,:,0]					
			mean_error_it2 = np.mean(accumulate_it2)
			mean_error_it1 = np.mean(accumulate_it1)
			mean_init_error = np.mean(accumulate_init)
			print('Init mean error: ' + str(mean_init_error) + ' It 1 mean error : '+ str(mean_error_it1) + ' It 2 mean error : '+ str(mean_error_it2))								
	else:
		with tf.Session() as sess:

			if loader_data["loader"] == "kitti":
				dataloader = ktdt(params)
			elif loader_data["loader"] == "kitti15":
				dataloader = kt2015dt(params)  
			else:
				dataloader = frdt(params)

			height,width,channels = dataloader.get_data_size();				

			x=tf.placeholder(tf.float32, [1,None,None,channels], name="x_p")

			input_height = tf.Variable(0, name="input_height",dtype=tf.int32)	
			input_width = tf.Variable(0, name="input_width",dtype=tf.int32)
			is_training = tf.Variable(False, name="is_training",dtype=tf.bool)
			keep_prob = tf.placeholder(tf.float32,name="keep_prob")

			pred = tf.to_float(tfmodel.h_net(x,input_height,input_width,is_training,reuse=False,keep_prob=keep_prob))
			pred = tf.concat([pred[:,:,:,4:5],x[:,:,:,1:2]],3)
			pred = tf.to_float(tfmodel.h_net(pred,input_height,input_width,is_training,reuse=True,keep_prob=keep_prob))			

			saver = tf.train.Saver()
			saver.restore(sess, args.model)
			print("Model restored")

			while( dataloader.get_sample_index() <  dataloader.get_sample_size() ):

				data,sindex,o_height,o_width,name = dataloader.load_test_sample();
				outp = sess.run(pred, feed_dict={x: data,input_width:width,input_height:height,is_training:False,keep_prob:1})

				n_h = outp.shape[1] - o_height;
				n_w = outp.shape[2] - o_width;

				disp_p = np.copy(outp[0, n_h:outp.shape[1],n_w:outp.shape[2],4 ])
				cpputils.write2png(disp_p,str("./test/"+name))
				print("Res saved at: "+ "./test/"+name )
					
					



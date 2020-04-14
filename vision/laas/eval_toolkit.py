import cv2
import os
import numpy as np
import statistics
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# from sklearn.utils.fixes import signature
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy import interp


def read_gt_simple(gt_file):
	"""
	:param gt_path:
	:param fps:
	:return:
	"""
	if os.path.exists(gt_file):
		gt = np.loadtxt(gt_file, comments="#", delimiter=" ", unpack=False)
		return gt.tolist()
	return []

def read_gt_list(val_txt, batch_size):
	f = open(val_txt, 'r')
	lines = f.readlines()
	num = (len(lines)//batch_size)*batch_size

	gt = []

	for i in range(num):
		path   = lines[i].split(' ')[0]
		label  = lines[i].split(' ')[-1]
		iframe = lines[i].split(' ')[1]
		gt.append(int(label))

	return gt


def read_gt_avdiar(gt_path, fps=25):
	"""
	:param gt_path:
	:param fps:
	:return:
	"""
	gt_speakers = []
	gt_faces = []

	with open(gt_path + '/GroundTruth/speakers.rttm', 'r') as ff:
		lines = ff.readlines()
		num = len(lines)
		for line in lines:
			words = line.split(' ')
			if 'INFO' in words[0]:
				continue
			start = float(words[3]) * fps
			durat = float(words[4]) * fps
			label = int(words[7][-1])
			gt_speakers.append([label, start, durat])

	# Read faces
	with open(gt_path + '/GroundTruth/face_bb.txt', 'r') as ff:
		lines = ff.readlines()
		num = len(lines)
		for line in lines:
			words = line.split(',')
			nums = list(map(float, words))
			nums[0] = int(nums[0])
			nums[1] = int(nums[1])
			gt_faces.append(nums)

	return gt_speakers, gt_faces


def isSpeaking(gt, id, frame):
	if 'avdiar' in gt[0]:
		for segment in gt[1:]:
			if segment[0] != id:
				continue
			if frame > segment[1] and frame < segment[1] + segment[2]:
				return True
	elif 'avasm' in gt[0]:
		return gt[frame+1] == 1
	return False


def evaluate(gt_path, rs_path):
	gt = read_gt_avdiar(gt_path)
	for i in range(500):
		print('Frame:', i, isSpeaking(gt, 1, i))


def evaluate_performance_multiclass(gt_labels, pred_labels, pred, save_path=None, display=True):
	"""
	:param pred:
	:param gt_labels:
	:param pred_labels:
	:param pred:
	:param save_path:
	:return:
	"""
	ap, auc_, f1, precision, recall = [0, 0, 0, 0, 0]

	if len(gt_labels) == 0:
		return ap, auc_, f1, precision, recall

	if save_path is not None and not os.path.exists(save_path):
		os.makedirs(save_path)
	# try:
	n_classes = pred.shape[1]

	if n_classes == 2:
		y_test = np.concatenate((label_binarize(np.asanyarray(gt_labels)+1, classes=range(pred.shape[1])),
		                         label_binarize(gt_labels, classes=range(pred.shape[1]))), axis=1)
		# y_test = np.asmatrix([[0, 1] if gt_labels[i] else [1, 0] for i in range(len(gt_labels))])
	else:
		y_test = label_binarize(gt_labels,   classes=range(pred.shape[1]))
	z_test = label_binarize(pred_labels, classes=range(pred.shape[1]))
	lw = 2

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	precision=dict()
	recall   = dict()
	f1       = dict()
	ap       = np.zeros(n_classes)
	auc_     = np.zeros(n_classes)

	for i in range(n_classes):
		if sum(y_test[:, i]) == 0:
			continue
		# calculate roc curve
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])
		# calculate AUC
		roc_auc[i] = auc(fpr[i], tpr[i])
		# calculate AUC
		auc_[i] = roc_auc_score(y_test[:, i], pred[:, i])
		# calculate precision-recall curve
		# precision[i], recall[i] = precision_recall_curve(y_test[:, i], pred[:, i])
		# calculate average precision score
		ap[i] = average_precision_score(y_test[:, i], pred[:, i])

	n_classes = fpr.__len__()

	# calculate F1 score
	f1['micro'] = f1_score(gt_labels, pred_labels, average='micro')
	f1['macro'] = f1_score(gt_labels, pred_labels, average='macro')
	f1['weighted'] = f1_score(gt_labels, pred_labels, average='weighted')
	f1['None']  = f1_score(gt_labels, pred_labels, average=None)

	# Compute macro-average ROC curve and ROC area

	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	# Plot all ROC curves
	plt.figure()
	plt.plot(fpr["micro"], tpr["micro"],
	         label='micro-average (area = {0:0.2f})'
	               ''.format(roc_auc["micro"]),
	         color='deeppink', linestyle=':', linewidth=4)

	plt.plot(fpr["macro"], tpr["macro"],
	         label='macro-average (area = {0:0.2f})'
	               ''.format(roc_auc["macro"]),
	         color='navy', linestyle=':', linewidth=4)

	colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'forestgreen', 'firebrick', 'indigo', 'gold', 'dodgerblue', 'chocolate', 'violet', 'darkturquoise', 'peru', 'crimson'])
	for i, color in zip(range(n_classes), colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=lw,
		         label='Class {0} (area = {1:0.2f})'
		               ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve - multi-class')
	plt.legend(loc="lower right")
	plt.grid(True)
	if display:
		plt.show()
	if save_path:
		plt.savefig(save_path + '/roc_curve.pdf', format='pdf')

	# plt.figure(2)
	# # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
	# step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
	# plt.step(recall, precision, color='b', alpha=0.2, where='post')
	# plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
	#
	# plt.xlabel('Recall')
	# plt.ylabel('Precision')
	# plt.ylim([0.0, 1.05])
	# plt.xlim([0.0, 1.0])
	# plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(ap))
	# if save_path:
	# 	plt.savefig(save_path + '/Precision-Recall.pdf', format='pdf')

	if save_path:
		curve_roc = np.array([fpr, tpr])
		dataile_id = open(save_path + '/roc.txt', 'w')
		dataile_id.write(str(fpr))
		dataile_id.write(str(tpr))

		# np.savetxt(dataile_id, curve_roc)
		# results = np.array([ap, auc, f1, precision, recall])
		# results = np.array([gt_labels, pred_labels, prob])
		# np.savetxt(save_path + '/gt_pred_labels.txt', [gt_labels, pred_labels], fmt='%d')
		# np.savetxt(save_path + '/pred_prob.txt', [gt_prob, pred_prob], fmt='%f')

		with open(save_path + '/performance.txt', 'w') as f:
			f.write("average_precision_score: \n")
			np.savetxt(f, ap)
			f.write("roc_auc_score: \n" )
			np.savetxt(f, auc_)
			f.write("f1_score: \n")
			f.write(str(f1))
			# np.savetxt(f, f1)
			# f.write("precision: %.4f\n" % statistics.mean(precision))
			# f.write("recall: %.4f\n" % statistics.mean(recall))
	# else:
	# 	plt.show()
	#
	# except:
	# 	# if save_path:
	# 		# 	np.savetxt(save_path + '/gt_pred_labels.txt', [gt_labels, pred_labels], fmt='%d')
	# 		# 	np.savetxt(save_path + '/gt_pred_prob.txt', [gt_prob, pred_prob], fmt='%f')
	# 	print('Error')
	plt.close('all')
	return ap, auc_, f1, precision, recall, fpr, tpr, roc_auc


def evaluate_performance(gt_labels, pred_labels, gt_prob, pred_prob, save_path=None):
	"""
	:param gt_labels:
	:param pred_labels:
	:param prob:
	:param save_path:
	:return:
	"""
	ap, auc, f1, precision, recall = [0, 0, 0, 0, 0]

	if len(gt_labels) == 0:
		return ap, auc, f1, precision, recall

	if save_path is not None and not os.path.exists(save_path):
		os.makedirs(save_path)
	try:
		# calculate AUC
		auc = roc_auc_score(gt_labels, gt_prob)
		# calculate precision-recall curve
		precision, recall, thresholds = precision_recall_curve(gt_labels, gt_prob)
		# calculate F1 score
		f1 = f1_score(gt_labels, pred_labels)
		# calculate average precision score
		ap = average_precision_score(gt_labels, gt_prob)

		# calculate roc curve
		fpr, tpr, thresholds = roc_curve(gt_labels, gt_prob)
		# # plot no skill
		# pyplot.plot([0, 1], [0, 1], linestyle='--')
		# # plot the roc curve for the model
		# pyplot.plot(fpr, tpr, marker='.')
		# # show the plot
		# pyplot.show()

		plt.plot(fpr, tpr, label='ROC curve: AUC={0:0.2f}'.format(auc))
		plt.xlabel('1-Specificity')
		plt.ylabel('Sensitivity')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.0])
		plt.grid(True)
		plt.title('ROC curve')
		plt.legend(loc="lower left")
		if save_path:
			plt.savefig(save_path + '/roc_curve.pdf', format='pdf')

		# plt.figure(2)
		# # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
		# step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
		# plt.step(recall, precision, color='b', alpha=0.2, where='post')
		# plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
		#
		# plt.xlabel('Recall')
		# plt.ylabel('Precision')
		# plt.ylim([0.0, 1.05])
		# plt.xlim([0.0, 1.0])
		# plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(ap))
		if save_path:
			plt.savefig(save_path + '/Precision-Recall.pdf', format='pdf')

		if save_path:
			curve_roc = np.array([fpr, tpr])
			dataile_id = open(save_path + '/roc.txt', 'w')
			np.savetxt(dataile_id, curve_roc)
			# results = np.array([ap, auc, f1, precision, recall])
			# results = np.array([gt_labels, pred_labels, prob])
			# np.savetxt(save_path + '/gt_pred_labels.txt', [gt_labels, pred_labels], fmt='%d')
			# np.savetxt(save_path + '/pred_prob.txt', [gt_prob, pred_prob], fmt='%f')

			with open(save_path + '/performance.txt', 'w') as f:
				f.write("average_precision_score: %.4f\n" % ap)
				f.write("roc_auc_score: %.4f\n" % auc)
				f.write("f1_score: %.4f\n" % f1)
				f.write("precision: %.4f\n" % statistics.mean(precision))
				f.write("recall: %.4f\n" % statistics.mean(recall))
		else:
			plt.show()

	except:
		# if save_path:
			# 	np.savetxt(save_path + '/gt_pred_labels.txt', [gt_labels, pred_labels], fmt='%d')
			# 	np.savetxt(save_path + '/gt_pred_prob.txt', [gt_prob, pred_prob], fmt='%f')
		print('Error')
	plt.close('all')
	return ap, auc, f1, precision, recall


def evaluate_saved_results(path):
	if os.path.isfile(path + 'gt_pred_labels.txt'):
		# labels = read_gt_simple(path + 'gt_pred_labels.txt')
		# prob = read_gt_simple(path + 'gt_pred_prob.txt')
		# evaluate_performance(labels[1], labels[0], prob[0], prob[1], path)
		labels = np.loadtxt(path + '/gt_pred_labels.txt', comments="#", delimiter=" ", unpack=False)
		# pred   = np.loadtxt(path + '/prediction.txt', comments="#", delimiter=" ", unpack=False)
		# [gt_labels, pred_labels] = labels
		# ap, auc, f1, precision, recall, fpr, tpr, roc_auc  = evaluate_performance_multiclass(gt_labels, pred_labels, pred, display=False)
		# # ap, auc, f1, precision, recall, fpr, tpr, roc_auc = evaluate_performance_multiclass(gt_labels, pred_labels, predictions, save_path)
		#
		# eval_folder = ''
		# i = 0
		# # Plot all ROC curves
		# plt.figure()
		# plt.plot(fpr["micro"], tpr["micro"], color='red', lw=2, label='{0} (area = {1:0.2f})' ''.format('names[i]', roc_auc["micro"]), linestyle=linestyles[i%4])
		#
		#
		# plt.xlim([0.0, 1.0])
		# plt.ylim([0.0, 1.05])
		# plt.xlabel('False Positive Rate')
		# plt.ylabel('True Positive Rate')
		# plt.title('ROC curve - Micro')
		# plt.legend(loc="lower right", prop={'size': 8})
		# plt.grid(True)
		# i = 0
		# save_name = path + '/' + eval_folder + '_roc_curve_micro.pdf'
		# while os.path.isfile(save_name):
		# 	save_name = path + '/' + eval_folder + '_roc_curve_micro{0}.pdf'.format(i)
		# 	i += 1
		# plt.savefig(save_name, format='pdf')
		# plt.show()
		#
		# # Plot all ROC curves
		# plt.figure()
		# for i, color, e in zip(range(xnames.__len__()), colors, xnames):
		# 	if not e in mfpr:
		# 		continue
		# 	plt.plot(mfpr[e], mtpr[e], color=color, lw=2,
		# 	         label='{0} (area = {1:0.2f})' ''.format(names[i], mroc_auc[e]), linestyle=linestyles[i % 4])
		#
		# plt.xlim([0.0, 1.0])
		# plt.ylim([0.0, 1.05])
		# plt.xlabel('False Positive Rate')
		# plt.ylabel('True Positive Rate')
		# plt.title('ROC curve - Macro')
		# plt.legend(loc="lower right", prop={'size': 8})
		# plt.grid(True)
		# i = 0
		# save_name = path + '/' + eval_folder + '_roc_curve_macro.pdf'
		# while os.path.isfile(save_name):
		# 	save_name = path + '/' + eval_folder + '_roc_curve_macro{0}.pdf'.format(i)
		# 	i += 1
		# plt.savefig(save_name, format='pdf')
		# plt.show()

	else:
		gfpr = dict()
		gtpr = dict()
		groc_auc = dict()
		mfpr = dict()
		mtpr = dict()
		mroc_auc = dict()
		# Iterate over all folders
		dirnames = [f for f in os.listdir(path) if os.path.isdir(path + '/' + f)]

		dirnames.sort()
		# for (dirpath, dirnames, filenames) in os.walk(path):
		for dirname in dirnames:
			dir = path + '/' + dirname
			eval_folder = os.listdir(dir)
			eval_folder.sort()
			eval_folder = eval_folder[-1]
			# eval_folder = 'C5E10'
			# eval_folder = ''
			test_path = dir + '/' + eval_folder + '/test/'
			labels = np.loadtxt(test_path + '/gt_pred_labels.txt', comments="#", delimiter=" ", unpack=False)
			pred   = np.loadtxt(test_path + '/prediction.txt', comments="#", delimiter=" ", unpack=False)
			[gt_labels, pred_labels] = labels
			ap, auc, f1, precision, recall, fpr, tpr, roc_auc  = evaluate_performance_multiclass(gt_labels, pred_labels, pred, display=False)
			gfpr[dirname] = fpr["micro"]
			gtpr[dirname] = tpr["micro"]
			groc_auc[dirname] = roc_auc["micro"]
			mfpr[dirname] = fpr["macro"]
			mtpr[dirname] = tpr["macro"]
			mroc_auc[dirname] = roc_auc["macro"]

		colors = cycle(
			['aqua', 'darkorange', 'cornflowerblue', 'forestgreen', 'firebrick', 'indigo', 'gold', 'dodgerblue',
			 'chocolate', 'violet', 'darkturquoise', 'peru', 'crimson', 'blue'])
		linestyles = ['-', '--', '-.', ':']

		xnames = ['resnet2D_concat', 'C3D', 'resnet3D_18', 'resnet3D_34',
		        'sC3D', 'sresnet3D_18', 'sresnet3D_34',
	            'C3D_resnet34s','resnet3D_18_resnet34s', 'resnet3D_34_resnet34s',
		        'sC3D_resnet34s', 'sresnet3D_18_resnet34s', 'sresnet3D_34_resnet34s',
		          'hyperface']


		names = ['R2D - RGB', 'C3D - RGB', 'R3D18 - RGB', 'R3D34 - RGB',
		         'C3D - RGB + OF',     'R3D18 - RGB + OF',     'R3D34 - RGB + OF',
		         'C3D - RGB + A',      'R3D18 - RGB + A',      'R3D34 - RGB + A',
		         'C3D - RGB + OF + A', 'R3D18 - RGB + OF + A', 'R3D34 - RGB + OF + A',
		         'HyperFace']

		# Plot all ROC curves
		plt.figure()
		# for i, color, fpr, tpr, roc_auc in zip(range(gfpr.__len__()), colors, gfpr.items(), gtpr.items(), groc_auc.items()):
		for i, color, e in zip(range(xnames.__len__()), colors, xnames):
			if not e in gfpr:
				continue
			plt.plot(gfpr[e], gtpr[e], color=color, lw=2, label='{0} (area = {1:0.2f})' ''.format(names[i], groc_auc[e]), linestyle=linestyles[i%4])

		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC curve - Micro')
		plt.legend(loc="lower right", prop={'size': 8})
		plt.grid(True)
		i=0
		save_name = path + '/' + eval_folder + '_roc_curve_micro.pdf'
		while os.path.isfile(save_name):
			save_name = path + '/' + eval_folder + '_roc_curve_micro{0}.pdf'.format(i)
			i += 1
		plt.savefig(save_name, format='pdf')
		plt.show()

		# Plot all ROC curves
		plt.figure()
		for i, color, e in zip(range(xnames.__len__()), colors, xnames):
			if not e in mfpr:
				continue
			plt.plot(mfpr[e], mtpr[e], color=color, lw=2, label='{0} (area = {1:0.2f})' ''.format(names[i], mroc_auc[e]), linestyle=linestyles[i%4])

		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC curve - Macro')
		plt.legend(loc="lower right", prop={'size': 8})
		plt.grid(True)
		i=0
		save_name = path + '/' + eval_folder + '_roc_curve_macro.pdf'
		while os.path.isfile(save_name):
			save_name = path + '/' + eval_folder + '_roc_curve_macro{0}.pdf'.format(i)
			i += 1
		plt.savefig(save_name, format='pdf')
		plt.show()



def load_model_weights(model_file, num_classes):
	import keras
	if 'resnet2D' in model_file:
		import train_resnet2d_concat as trainer
		generator_batch = trainer.generator_test_batch

		model = trainer.resnet50_model(num_classes, trainer.kernel_w, trainer.kernel_h, 3 * trainer.clip_size)


	elif 'C3D' in model_file:
		import train_c3d as trainer
		generator_batch = trainer.generator_test_batch

		model = trainer.c3d_model(num_classes, 112, 112)


	elif 'resnet3D_18' in model_file:
		import train_resnet3d as trainer
		generator_batch = trainer.generator_test_batch

		model = trainer.m.r3d_18(num_classes, trainer.kernel_w, trainer.kernel_h)


	elif 'resnet3D_34' in model_file:
		import train_resnet3d as trainer
		generator_batch = trainer.generator_test_batch

		model = trainer.m.r3d_34(num_classes, trainer.kernel_w, trainer.kernel_h)


	from keras.optimizers import SGD
	sgd = SGD(lr=0.005, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	model.summary()
	model.load_weights(model_file, by_name=True)
	return generator_batch, model


def evaluate_model(test_file, model, generator_batch, save_path, num_classes = 2, batch_size = 16):

	img_path = ''


	gt_labels = read_gt_list(test_file, 16)

	with open(test_file, 'r') as ff:
		lines = ff.readlines()
		ff.close()
		val_samples = len(lines)

	predictions = model.predict_generator(generator_batch(test_file, batch_size, num_classes, img_path),
								  steps=val_samples // batch_size, verbose=1)

	pred_labels = [np.argmax(predictions[i]) for i in range(len(predictions))]
	pred_probs  = [predictions[i, pred_labels[i]] for i in range(len(predictions))]
	gt_probs    = [predictions[i, gt_labels[i]]   for i in range(len(predictions))]

	binarize = False
	if binarize:
		tmp = [predictions[:, 0:5].sum(axis=1), predictions[:, 5:10].sum(axis=1)]
		predictions=np.asanyarray(tmp).transpose()
		pred_labels=[0 if k < 5 else 1 for k in pred_labels]

	if save_path is not None and not os.path.exists(save_path):
		os.makedirs(save_path)
	np.savetxt(save_path + '/gt_pred_labels.txt', [gt_labels, pred_labels], fmt='%d')
	np.savetxt(save_path + '/gt_pred_prob.txt', [gt_probs, pred_probs], fmt='%f')
	np.savetxt(save_path + '/prediction.txt', predictions, fmt='%f')
	# return  gt_labels, pred_labels, pred_probs, gt_probs, save_path, predictions

	if predictions.shape[1] > 0:
		ap, auc, f1, precision, recall, fpr, tpr, roc_auc = evaluate_performance_multiclass(gt_labels, pred_labels, predictions, save_path)
	else: # Originally, this method should do evaluate binary classification but results are not correct
		ap, auc, f1, precision, recall = evaluate_performance(gt_labels, pred_labels, predictions[:,0], pred_probs, save_path)

	return ap, auc, f1, precision, recall


def evaluate_trained_models(test_file, model_path, num_classes = 2, batch_size = 16):

	img_path = ''


	gt_labels = read_gt_list(test_file, 1)

	with open(test_file, 'r') as ff:
		lines = ff.readlines()
		ff.close()
		val_samples = len(lines)

	for (dirpath, dirnames, filenames) in os.walk(model_path):

		if len(filenames) == 0 and not 'weights.h5' in filenames or not 'resnet' in dirpath and not 'resnet' in dirpath:
			continue



		save_path = './results/resnet/' + dirpath.split('weights')[1]
		dirpath = dirpath.replace('resnet2D_concat', 'resnet3D_18')
		dirpath = dirpath.replace('resnet3D_34', 'resnet3D_18')
		model_file = dirpath + '/weights.h5'
		model = []
		generator_batch = None


		generator_batch, model = load_model_weights(model_file, num_classes)

		predictions = model.predict_generator(generator_batch(test_file, batch_size, num_classes, img_path),
									  steps=val_samples // batch_size, verbose=1)

		pred_labels = [np.argmax(predictions[i]) for i in range(len(predictions))]
		pred_probs  = [predictions[i, pred_labels[i]] for i in range(len(predictions))]
		gt_probs    = [predictions[i, gt_labels[i]]   for i in range(len(predictions))]

		ap, auc, f1, precision, recall = evaluate_performance(gt_labels, pred_labels, pred_probs, gt_probs, save_path)


if __name__ == '__main__':
	# ucf_dataset()
	# mvlrs_v1_dataset()
	# vid_dataset()
	# afew_dataset()
	# vid_dataset_slow()
	# get_lips('/data/jfmadrig/mvlrs_v1/pretrain_heads/', '/data/jfmadrig/mvlrs_v1/pretrain_lips/')
	# get_lips('/data/jfmadrig/VidTIMIT_heads/', '/data/jfmadrig/VidTIMIT_lips/')
	# evaluate('/data/jfmadrig/avdiar/Seq43-2P-S0M0/GroundTruth/speakers.rttm', '')
	# path = '/local/users/jfmadrig/LinTo/demo/visual_speech/results/ibug-avs/lr-vid-ict/lips/C3D/C2E10/global/'
	# evaluate_saved_results(path)
	#
	# test_file = 'lists/test_lips_list_lr-vid-ict.txt'
	# model_path = 'weights/lr-vid-ict/lips/'
	# evaluate_trained_models(test_file,model_path)

	# evaluate_saved_results('/local/users/jfmadrig/LinTo/demo/video_audio_speaker_detection/weights/amiOK')
	# evaluate_saved_results('/local/users/jfmadrig/video_audio_speaker_detection/weights/ami_ws_cv5/')
	# evaluate_saved_results('/local/bs14/users/jfmadrig/LinTo/demo/visual_speech/weights/ibugs/heads_angles/')
	# evaluate_saved_results('/local/users/jfmadrig/Programs/hyperface/results/ict3DHP/')
	evaluate_saved_results('./weights/bs14/ami_ws_cv5/')
	print("All Done!")


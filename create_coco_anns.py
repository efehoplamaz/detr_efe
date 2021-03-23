import json
import sys
import os
import argparse
import generate_spectrogram as gs
import contextlib
import wave

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--bigger_bbox')

args = parser.parse_args()

if args.dataset == 'train':
	#PATH = 'C:/Users/ehopl/Desktop/bat_data/annotations/BritishBatCalls_MartynCooke_2018_1_sec_train_expert.json'
	#AUDIO_PATH = 'C:/Users/ehopl/Desktop/bat_data/audio/mc_2018/audio/'
        PATH = '/home/s1764306/data/annotations/train.json'
        AUDIO_PATH = '/home/s1764306/data/audio/mc_2018/audio/'
        # PATH = 'C:/Users/ehopl/Desktop/bat_data/train.json'
        # AUDIO_PATH = 'C:/Users/ehopl/Desktop/bat_data/audio/mc_2018/audio/'
elif args.dataset == 'test':
	#PATH = 'C:/Users/ehopl/Desktop/bat_data/annotations/BritishBatCalls_MartynCooke_2019_1_sec_train_expert.json'
	#AUDIO_PATH = 'C:/Users/ehopl/Desktop/bat_data/audio/mc_2019/audio/'
        PATH = '/home/s1764306/data/annotations/test.json'
        AUDIO_PATH = '/home/s1764306/data/audio/mc_2019/audio/'
        # PATH = 'C:/Users/ehopl/Desktop/bat_data/test.json'
        # AUDIO_PATH = 'C:/Users/ehopl/Desktop/bat_data/audio/mc_2019/audio/'
else:
	print('No dataset like that!')
	sys.exit()

coco_data = {}
coco_data['info'] = {}
coco_data['licenses'] = []
coco_data['images'] = []
coco_data['categories'] = [{"supercategory" : "none", "id": 0, "name": "bat_call"}]
coco_data['annotations'] = []

f = open(PATH)
data = json.loads(f.read())

image_unq_id = 1
ann_unq_id = 1

for wav_f in os.listdir(AUDIO_PATH):
	spec, sampling_rate, spec_duration = gs.get_spectrogram_sampling_rate(AUDIO_PATH + wav_f)
	ann = []
	for d in data:
		if d['id'] == wav_f:
			ann = d
			break

	coco_data['images'].append({"file_name" : wav_f, "height": spec.shape[0], "width": spec.shape[1], "id": image_unq_id})

	fname = AUDIO_PATH + wav_f
	with contextlib.closing(wave.open(fname,'r')) as f:
	    frames = f.getnframes()
	    rate = f.getframerate()
	    duration = frames / float(rate)

	if ann:
		if ann['annotation']:	
			for a in ann['annotation']:
				lb_x = a['start_time'] * (spec.shape[1]/duration)
				lb_y = a['low_freq'] * (spec.shape[0]/120000)
				w = (a['end_time'] - a['start_time']) * (spec.shape[1]/duration)
				h = (a['high_freq'] - a['low_freq']) * (spec.shape[0]/120000)
				area = w * h
				if args.bigger_bbox == "True":
					coco_data['annotations'].append({"id": ann_unq_id, "bbox":  [(lb_x-(w/2)), (lb_y-(h/2)), 2*w, 2*h], "image_id": image_unq_id, "segmentation" : [], "area" : area, "category_id": 0, "iscrowd": 0})
				else:
					coco_data['annotations'].append({"id": ann_unq_id, "bbox":  [lb_x, lb_y, w, h], "image_id": image_unq_id, "segmentation" : [], "area" : area, "category_id": 0, "iscrowd": 0}) ## Left-bottom, width, height
				ann_unq_id += 1

	image_unq_id += 1

if args.dataset == 'train':
	if args.bigger_bbox == "True":
		with open('/home/s1764306/data/annotations/coco_v_train_b.json', 'w') as json_file:
		    json.dump(coco_data, json_file)
	else:
		with open('/home/s1764306/data/annotations/coco_v_train.json', 'w') as json_file:
		    json.dump(coco_data, json_file)
else:
	if args.bigger_bbox == "True":
		with open('/home/s1764306/data/annotations/coco_v_test_b.json', 'w') as json_file:
		    json.dump(coco_data, json_file)
	else:
		with open('/home/s1764306/data/annotations/coco_v_test.json', 'w') as json_file:
		    json.dump(coco_data, json_file)	







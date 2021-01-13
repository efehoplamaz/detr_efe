import json
import os

TR_JSON_PATH = './annotations/BritishBatCalls_MartynCooke_2018_1_sec_train_expert.json'
TST_JSON_PATH = './annotations/BritishBatCalls_MartynCooke_2019_1_sec_train_expert.json'

TR_AUD_PATH = './audio/mc_2018/audio/'
TST_AUD_PATH = './audio/mc_2019/audio/'

####################################################################################################

train_json = json.load(open(TR_JSON_PATH))
test_json = json.load(open(TST_JSON_PATH))

rmv_lst = os.listdir(TR_AUD_PATH)
rmv_lst_tst = os.listdir(TST_AUD_PATH)

if len(train_json) != len(os.listdir(TR_AUD_PATH)):
	if len(train_json) > len(os.listdir(TR_AUD_PATH)):
		for i, tr_ann in enumerate(train_json):
			if tr_ann['id'] not in os.listdir(TR_AUD_PATH):
				del tr_ann[i]
	else:
		for tr_ann in train_json:
			if tr_ann['id'] in rmv_lst:
				rmv_lst.remove(tr_ann['id'])
		for wav_file in rmv_lst:
			os.remove(TR_AUD_PATH + wav_file)
			print(wav_file + ' removed!')
		with open('train_wav_files_missing_annotation.txt', 'w') as f:
			for item in rmv_lst:
				f.write("%s\n" % item)


if len(test_json) != len(os.listdir(TST_AUD_PATH)):
	if len(test_json) > len(os.listdir(TST_AUD_PATH)):
		for i, tst_ann in enumerate(test_json):
			if tst_ann['id'] not in os.listdir(TST_AUD_PATH):
				del tst_ann[i]
	else:
		for tst_ann in test_json:
			if tst_ann['id'] in rmv_lst_tst:
				rmv_lst_tst.remove(tst_ann['id'])
		for wav_file in rmv_lst_tst:
			os.remove(TST_AUD_PATH + wav_file)
			print(wav_file + ' removed!')
		with open('test_wav_files_missing_annotation.txt', 'w') as c:
			for item in rmv_lst_tst:
				c.write("%s\n" % item)

print('Train JSON annotation size is {} and total train wav files is {}.'.format(len(train_json), len(os.listdir(TR_AUD_PATH))))
print('Test JSON annoation size is {} and total test wav files is {}.'.format(len(test_json), len(os.listdir(TST_AUD_PATH))))


####################################################################################################

### QUICK CHECK

bool_lst_train = []
bool_lst_test = []

for wav_file in os.listdir(TR_AUD_PATH):
	if any(d['id'] == wav_file for d in train_json):
		bool_lst_train.append(True)
	else:
		bool_lst_train.append(False)

for wav_file in os.listdir(TST_AUD_PATH):
	if any(d['id'] == wav_file for d in test_json):
		bool_lst_test.append(True)
	else:
		bool_lst_test.append(False)

print('All files and annotations match for training set: {}'.format(all(bool_lst_train)))
print('All files and annotations match for test set: {}'.format(all(bool_lst_test)))

####################################################################################################

### MOVING FROM TEST TO TRAIN

train_all_species = [] 
test_all_species = []

difference = []

for tr_ann in train_json:
	if tr_ann['class_name'] not in train_all_species:
		train_all_species.append(tr_ann['class_name'])

for tst_ann in test_json:
	if tst_ann['class_name'] not in test_all_species:
		test_all_species.append(tst_ann['class_name'])

for sp in test_all_species:
	if sp not in train_all_species:
		difference.append(sp)

print(difference)

for i, ann in enumerate(test_json):
	if ann['class_name'] in difference:
		train_json.append(ann)
		source = TST_AUD_PATH + ann['id']
		destination = TR_AUD_PATH + ann['id']
		os.rename(source, destination)
		del test_json[i]

print('New train JSON annotation size is {} and total train wav files is {}.'.format(len(train_json), len(os.listdir(TR_AUD_PATH))))
print('New test JSON annoation size is {} and total test wav files is {}.'.format(len(test_json), len(os.listdir(TST_AUD_PATH))))

####################################################################################################

##### QUICK CHECK

bool_lst_train_v2 = []
bool_lst_test_v2 = []

for wav_file in os.listdir(TR_AUD_PATH):
	if any(d['id'] == wav_file for d in train_json):
		bool_lst_train_v2.append(True)
	else:
		bool_lst_train_v2.append(False)

for wav_file in os.listdir(TST_AUD_PATH):
	if any(d['id'] == wav_file for d in test_json):
		bool_lst_test_v2.append(True)
	else:
		bool_lst_test_v2.append(False)

print('All files and annotations match for new training set: {}'.format(all(bool_lst_train)))
print('All files and annotations match for new test set: {}'.format(all(bool_lst_test)))

with open('./annotations/train_val.json', 'w') as tr_outfile:
    json.dump(train_json, tr_outfile)
with open('./annotations/test.json', 'w') as ts_outfile:
    json.dump(test_json, ts_outfile)